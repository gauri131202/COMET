from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.common.paths import MODELS_DIR
from src.db import get_connection, get_dataset, init_db, insert_model_metadata
from src.longform.data import (
    FEATURE_COLUMNS,
    LongFormConfig,
    build_windows,
    fit_normalization,
    load_readings,
    numeric_series,
)
from src.modeling.lstm_autoencoder import LSTMAutoencoder, reconstruction_loss
from src.modeling.train_model import (
    _next_model_path,
    _percentile,
    _reconstruction_errors,
    load_training_config,
    resolve_training_config_path,
)


def train_longform(
    dataset_id: int,
    config_path: str | Path | None = None,
    *,
    long_config: LongFormConfig | None = None,
    db_path: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, Any]:
    """Train an LSTM autoencoder on per-series windows from `samples` (same flow as train_model)."""
    init_db(db_path)
    if get_dataset(dataset_id, db_path=db_path) is None:
        raise ValueError(f"Dataset {dataset_id} does not exist.")

    long_config = long_config or LongFormConfig()
    resolved_config_path = resolve_training_config_path(config_path)
    training_config = load_training_config(resolved_config_path)
    window_size, stride = int(training_config["window_size"]), int(training_config["stride"])
    if long_config.min_readings < window_size:
        raise ValueError(f"min_readings must be at least window_size={window_size}.")

    parts: list[torch.Tensor] = []
    series_stats: dict[str, dict[str, Any]] = {}
    total_windows = 0
    with get_connection(db_path) as conn:
        for series_id, key in numeric_series(conn, dataset_id, long_config):
            readings = load_readings(conn, dataset_id, series_id, key, long_config)
            if readings is None:
                continue
            mean, std = fit_normalization(readings, long_config.normalize_fraction)
            windows = build_windows(readings, mean, std, long_config, window_size, stride)
            if windows is None:
                continue
            total_windows += windows.size(0)
            # train on the first part of each series (in time), like train_model's split
            split_index = max(1, int(windows.size(0) * float(training_config["train_test_split"])))
            train_part = windows[:split_index]
            if train_part.size(0) > long_config.max_windows_per_series:
                keep = torch.linspace(0, train_part.size(0) - 1, long_config.max_windows_per_series).long()
                train_part = train_part[keep]
            parts.append(train_part)
            series_stats[key] = {"series_id": series_id, "mean": mean, "std": std}
    if not parts:
        raise ValueError("No series had enough readings to create training windows.")
    train_windows = torch.cat(parts)

    model = LSTMAutoencoder(
        num_features=len(FEATURE_COLUMNS),
        hidden_size=int(training_config["hidden_size"]),
        num_layers=int(training_config["num_layers"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    loader = DataLoader(
        TensorDataset(train_windows), batch_size=int(training_config["batch_size"]), shuffle=True
    )

    epoch_losses: list[float] = []
    model.train()
    for _ in range(int(training_config["epochs"])):
        batch_losses: list[float] = []
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = reconstruction_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    reconstruction_errors = _reconstruction_errors(model, train_windows)
    threshold_percentile = float(training_config["threshold_percentile"])
    threshold = _percentile(reconstruction_errors, threshold_percentile)

    model_path = _next_model_path(Path(models_dir), dataset_id)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_id": dataset_id,
            "feature_columns": FEATURE_COLUMNS,
            "num_features": len(FEATURE_COLUMNS),
            "config": training_config,
            "long_config": long_config.__dict__,
            "series_stats": series_stats,  # mean and std per series, to scale new data the same way
            "threshold": threshold,
            "threshold_method": "training_reconstruction_error_percentile",
            "threshold_percentile": threshold_percentile,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        model_path,
    )

    model_id = insert_model_metadata(
        dataset_id,
        model_path,
        config_path=resolved_config_path,
        threshold=threshold,
        threshold_method="training_reconstruction_error_percentile",
        threshold_percentile=threshold_percentile,
        training_summary={
            "epochs": int(training_config["epochs"]),
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "epoch_losses": epoch_losses,
            "training_window_count": int(train_windows.size(0)),
            "total_window_count": total_windows,
        },
        metadata={
            "pipeline": "long_form",
            "feature_columns": FEATURE_COLUMNS,
            "num_features": len(FEATURE_COLUMNS),
            "config": training_config,
            "long_config": long_config.__dict__,
            "series_trained": len(series_stats),
        },
        db_path=db_path,
    )
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model_path": str(model_path),
        "config_path": str(resolved_config_path),
        "threshold": threshold,
        "series_trained": len(series_stats),
        "training_window_count": int(train_windows.size(0)),
    }
