from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on local environment.
    yaml = None

from src.common.paths import MODEL_CONFIG_DIR, MODELS_DIR
from src.db import get_connection, get_dataset, init_db, insert_model_metadata, table_exists
from src.modeling.lstm_autoencoder import (
    LSTMAutoencoder,
    create_sliding_windows,
    reconstruction_loss,
)


DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "window_size": 100,
    "stride": 1,
    "hidden_size": 64,
    "num_layers": 1,
    "batch_size": 32,
    "epochs": 20,
    "train_test_split": 0.8,
    "threshold_percentile": 95,
    "learning_rate": 0.001,
}
DEFAULT_TRAINING_CONFIG_PATH = MODEL_CONFIG_DIR / "default.yaml"


@dataclass(frozen=True)
class TrainingData:
    feature_columns: list[str]
    values: torch.Tensor


def train_model(
    dataset_id: int,
    config_path: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, Any]:
    """Train an LSTM autoencoder, using the default training config when no path is provided."""
    init_db(db_path)
    dataset = get_dataset(dataset_id, db_path=db_path)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} does not exist.")

    model_ready_table = f"model_ready_dataset_{dataset_id}"
    resolved_config_path = resolve_training_config_path(config_path)
    training_config = load_training_config(resolved_config_path)
    training_data = _load_training_data(dataset_id, model_ready_table, db_path=db_path)
    windows = create_sliding_windows(
        training_data.values,
        window_size=int(training_config["window_size"]),
        stride=int(training_config["stride"]),
    )
    if windows.size(0) == 0:
        raise ValueError(
            "Not enough rows to create training windows. "
            f"Need at least window_size={training_config['window_size']} rows."
        )

    split_index = max(1, int(windows.size(0) * float(training_config["train_test_split"])))
    split_index = min(split_index, windows.size(0))
    train_windows = windows[:split_index]

    model = LSTMAutoencoder(
        num_features=len(training_data.feature_columns),
        hidden_size=int(training_config["hidden_size"]),
        num_layers=int(training_config["num_layers"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    loader = DataLoader(
        TensorDataset(train_windows),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
    )

    epoch_losses: list[float] = []
    model.train()
    for _ in range(int(training_config["epochs"])):
        batch_losses: list[float] = []
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = reconstruction_loss(reconstructed, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    reconstruction_errors = _reconstruction_errors(model, train_windows)
    threshold_percentile = float(training_config["threshold_percentile"])
    threshold = _percentile(reconstruction_errors, threshold_percentile)

    model_path = _next_model_path(Path(models_dir), dataset_id)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "dataset_id": dataset_id,
        "model_ready_table": model_ready_table,
        "feature_columns": training_data.feature_columns,
        "num_features": len(training_data.feature_columns),
        "config": training_config,
        "threshold": threshold,
        "threshold_method": "training_reconstruction_error_percentile",
        "threshold_percentile": threshold_percentile,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)

    training_summary = {
        "epochs": int(training_config["epochs"]),
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "training_window_count": int(train_windows.size(0)),
        "total_window_count": int(windows.size(0)),
        "reconstruction_error_min": min(reconstruction_errors),
        "reconstruction_error_max": max(reconstruction_errors),
        "reconstruction_error_mean": sum(reconstruction_errors) / len(reconstruction_errors),
    }
    metadata = {
        "model_ready_table": model_ready_table,
        "feature_columns": training_data.feature_columns,
        "num_features": len(training_data.feature_columns),
        "config": training_config,
    }
    model_id = insert_model_metadata(
        dataset_id,
        model_path,
        config_path=resolved_config_path,
        threshold=threshold,
        threshold_method="training_reconstruction_error_percentile",
        threshold_percentile=threshold_percentile,
        training_summary=training_summary,
        metadata=metadata,
        db_path=db_path,
    )

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model_path": str(model_path),
        "config_path": str(resolved_config_path),
        "threshold": threshold,
        "feature_count": len(training_data.feature_columns),
        "training_window_count": int(train_windows.size(0)),
    }


def resolve_training_config_path(config_path: str | Path | None = None) -> Path:
    """Return the explicit training config path or config/model_configs/default.yaml."""
    if config_path is not None:
        return Path(config_path)
    return DEFAULT_TRAINING_CONFIG_PATH


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    """Load a training config file when present and merge it with built-in defaults."""
    config = DEFAULT_TRAINING_CONFIG.copy()
    path = Path(config_path)
    if path.exists():
        parsed = _load_config_file(path)
        config.update({key: value for key, value in parsed.items() if value is not None})

    _validate_training_config(config)
    return config


def _load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        parsed = yaml.safe_load(text) or {}
        if not isinstance(parsed, dict):
            raise ValueError(f"Training config must be a mapping: {path}")
        return parsed
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _parse_simple_yaml_mapping(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return parsed


def _parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parsed[key.strip()] = _parse_scalar(value.strip())
    return parsed


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _validate_training_config(config: dict[str, Any]) -> None:
    positive_ints = ["window_size", "stride", "hidden_size", "num_layers", "batch_size", "epochs"]
    for key in positive_ints:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if not 0 < float(config["train_test_split"]) <= 1:
        raise ValueError("train_test_split must be > 0 and <= 1.")
    if not 0 < float(config["threshold_percentile"]) <= 100:
        raise ValueError("threshold_percentile must be > 0 and <= 100.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive.")


def _load_training_data(
    dataset_id: int,
    table_name: str,
    *,
    db_path: str | Path | None,
) -> TrainingData:
    with get_connection(db_path) as conn:
        if not table_exists(conn, table_name):
            raise ValueError(f"Required table {table_name} is missing. Run prepare first.")

        column_info = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
        feature_columns = [
            row["name"]
            for row in column_info
            if row["name"] not in {"row_id", "dataset_id", "timestamp"}
            and row["type"].upper() in {"REAL", "INTEGER", "NUMERIC", "FLOAT", "DOUBLE"}
        ]
        if not feature_columns:
            raise ValueError(f"No numeric feature columns found in {table_name}.")

        rows = conn.execute(
            f"""
            SELECT {", ".join(_quote_identifier(column) for column in feature_columns)}
            FROM {_quote_identifier(table_name)}
            WHERE dataset_id = ?
            ORDER BY row_id
            """,
            (dataset_id,),
        ).fetchall()

    if not rows:
        raise ValueError(f"No rows found in {table_name}.")

    values = torch.tensor(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=torch.float32,
    )
    return TrainingData(feature_columns=feature_columns, values=values)


def _reconstruction_errors(model: LSTMAutoencoder, windows: torch.Tensor) -> list[float]:
    model.eval()
    with torch.no_grad():
        reconstructed = model(windows)
        errors = torch.mean((reconstructed - windows) ** 2, dim=(1, 2))
    return [float(value) for value in errors.tolist()]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile for empty values.")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _next_model_path(models_dir: Path, dataset_id: int) -> Path:
    date_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    pattern = re.compile(rf"model_{date_stamp}_dataset{dataset_id}_v(\d+)\.pt$")
    versions = []
    if models_dir.exists():
        for path in models_dir.glob(f"model_{date_stamp}_dataset{dataset_id}_v*.pt"):
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))
    version = max(versions, default=0) + 1
    return models_dir / f"model_{date_stamp}_dataset{dataset_id}_v{version}.pt"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
