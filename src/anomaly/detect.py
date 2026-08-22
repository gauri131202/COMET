from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.db import get_connection, get_dataset, get_model, init_db, insert_anomaly, table_exists
from src.modeling.lstm_autoencoder import LSTMAutoencoder, create_sliding_windows


def detect_anomalies(
    dataset_id: int,
    model_id: int,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score prepared telemetry windows with a trained model and store detected anomalies."""
    init_db(db_path)

    # Checks
    dataset = get_dataset(dataset_id, db_path=db_path)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} does not exist.")

    model_row = get_model(model_id, db_path=db_path)
    if model_row is None:
        raise ValueError(f"Model {model_id} does not exist.")
    if int(model_row["dataset_id"]) != dataset_id:
        raise ValueError(f"Model {model_id} was trained for dataset {model_row['dataset_id']}, not {dataset_id}.")

    model_path = Path(model_row["model_path"])
    if not model_path.exists():
        raise ValueError(f"Model file does not exist: {model_path}")

    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    model_config = checkpoint.get("config") or _json_dict(model_row["metadata_json"]).get("config") or {}
    feature_columns = checkpoint.get("feature_columns") or _json_dict(model_row["metadata_json"]).get("feature_columns")
    if not feature_columns:
        raise ValueError(f"Model {model_id} is missing feature column metadata.")

    threshold = float(model_row["threshold"])
    model_ready_table = checkpoint.get("model_ready_table") or f"model_ready_dataset_{dataset_id}"
    rows, values = _load_model_ready_rows(
        dataset_id,
        model_ready_table,
        feature_columns,
        db_path=db_path,
    )
    window_size = int(model_config.get("window_size", 100))
    stride = int(model_config.get("stride", 1))
    windows = create_sliding_windows(values, window_size=window_size, stride=stride)
    if windows.size(0) == 0:
        raise ValueError(
            "Not enough rows to create detection windows. "
            f"Need at least window_size={window_size} rows."
        )

    model = LSTMAutoencoder(
        num_features=len(feature_columns),
        hidden_size=int(model_config.get("hidden_size", 64)),
        num_layers=int(model_config.get("num_layers", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        reconstructed = model(windows)
        per_window_errors = torch.mean((reconstructed - windows) ** 2, dim=(1, 2))
        per_feature_errors = torch.mean((reconstructed - windows) ** 2, dim=1)

    anomalies = []
    for window_index, error in enumerate(per_window_errors.tolist()):
        anomaly_score = float(error)
        if anomaly_score <= threshold:
            continue

        start_offset = window_index * stride
        end_offset = start_offset + window_size - 1
        start_row = rows[start_offset]
        end_row = rows[end_offset]
        top_features, contribution_method = _top_contributing_features(
            feature_columns=feature_columns,
            feature_errors=per_feature_errors[window_index],
            window=windows[window_index],
        )
        anomaly_id = f"anomaly_dataset{dataset_id}_model{model_id}_window{window_index}"
        packet = {
            "anomaly_id": anomaly_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "window_index": window_index,
            "start_row_id": int(start_row["row_id"]),
            "end_row_id": int(end_row["row_id"]),
            "start_timestamp": start_row.get("timestamp"),
            "end_timestamp": end_row.get("timestamp"),
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "top_contributing_features": top_features,
            "contribution_method": contribution_method,
        }

        # Write out anomaly to database
        insert_anomaly(
            anomaly_id=anomaly_id,
            dataset_id=dataset_id,
            model_id=model_id,
            window_index=window_index,
            start_row_id=int(start_row["row_id"]),
            end_row_id=int(end_row["row_id"]),
            start_timestamp=start_row.get("timestamp"),
            end_timestamp=end_row.get("timestamp"),
            anomaly_score=anomaly_score,
            threshold=threshold,
            contributing_features=top_features,
            packet=packet,
            db_path=db_path,
        )
        anomalies.append(packet)

    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "windows_scored": int(windows.size(0)),
        "anomalies_detected": len(anomalies),
        "threshold": threshold,
        "anomalies": anomalies,
    }


def _load_model_ready_rows(
    dataset_id: int,
    table_name: str,
    feature_columns: list[str],
    *,
    db_path: str | Path | None,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Load ordered model-ready rows and tensor values for the model feature columns."""
    with get_connection(db_path) as conn:
        if not table_exists(conn, table_name):
            raise ValueError(f"Required table {table_name} is missing. Run prepare first.")
        existing_columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
        }
        missing_features = [column for column in feature_columns if column not in existing_columns]
        if missing_features:
            raise ValueError(f"Model features are missing from {table_name}: {missing_features}")

        select_columns = ["row_id", *feature_columns]
        if "timestamp" in existing_columns:
            select_columns.insert(1, "timestamp")
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT {", ".join(_quote_identifier(column) for column in select_columns)}
                FROM {_quote_identifier(table_name)}
                WHERE dataset_id = ?
                ORDER BY row_id
                """,
                (dataset_id,),
            ).fetchall()
        ]

    if not rows:
        raise ValueError(f"No rows found in {table_name}.")
    values = torch.tensor(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=torch.float32,
    )
    return rows, values


def _top_contributing_features(
    *,
    feature_columns: list[str],
    feature_errors: torch.Tensor,
    window: torch.Tensor,
    limit: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """Rank features by reconstruction error, falling back to feature change when needed."""
    scores = [float(value) for value in feature_errors.tolist()]
    method = "per_feature_reconstruction_error"
    if not any(score > 0 for score in scores):
        scores = [float(abs(window[-1, index] - window[0, index])) for index in range(window.size(1))]
        method = "absolute_feature_change"

    ranked = sorted(
        zip(feature_columns, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": feature, "score": score, "rank": rank}
        for rank, (feature, score) in enumerate(ranked[:limit], start=1)
    ], method


def _json_dict(payload: str | None) -> dict[str, Any]:
    """Parse a JSON object payload and return an empty dict for missing payloads."""
    if not payload:
        return {}
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, dict) else {}


def _quote_identifier(identifier: str) -> str:
    """Return a SQLite-quoted identifier safe for generated SQL."""
    return '"' + identifier.replace('"', '""') + '"'
