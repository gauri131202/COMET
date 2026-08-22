from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.db import get_anomaly, get_connection, get_model, init_db, table_exists


def build_anomaly_packet(
    anomaly_id: str,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    init_db(db_path)
    anomaly = get_anomaly(anomaly_id, db_path=db_path)
    if anomaly is None:
        raise ValueError(f"Anomaly {anomaly_id} does not exist.")

    dataset_id = int(anomaly["dataset_id"])
    model_id = int(anomaly["model_id"])
    model = get_model(model_id, db_path=db_path)
    if model is None:
        raise ValueError(f"Model {model_id} does not exist.")

    model_metadata = _json_dict(model["metadata_json"])
    model_config = model_metadata.get("config") or {}
    features_used = model_metadata.get("feature_columns") or []
    model_ready_table = model_metadata.get("model_ready_table") or f"model_ready_dataset_{dataset_id}"
    context_ready_table = f"context_ready_dataset_{dataset_id}"
    raw_table = f"raw_dataset_{dataset_id}"
    start_row_id = int(anomaly["start_row_id"])
    end_row_id = int(anomaly["end_row_id"])
    top_contributing_features = _json_list(anomaly["contributing_features_json"])

    model_rows = _get_rows(model_ready_table, dataset_id, start_row_id, end_row_id, db_path=db_path)
    context_rows = _get_rows_if_exists(context_ready_table, dataset_id, start_row_id, end_row_id, db_path=db_path)
    raw_exists = _table_exists(raw_table, db_path=db_path)

    top_feature_names = [
        str(item["feature"])
        for item in top_contributing_features
        if isinstance(item, dict) and item.get("feature")
    ]
    numeric_summary = _numeric_summary(model_rows, top_feature_names)
    categorical_changes = _categorical_changes(context_rows)

    return {
        "anomaly_id": anomaly_id,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "time_window": {
            "window_index": int(anomaly["window_index"]),
            "start_row_id": start_row_id,
            "end_row_id": end_row_id,
            "start_timestamp": anomaly["start_timestamp"],
            "end_timestamp": anomaly["end_timestamp"],
        },
        "anomaly_score": float(anomaly["anomaly_score"]),
        "threshold": float(anomaly["threshold"]),
        "model_context": {
            "features_used": features_used,
            "window_size": model_config.get("window_size"),
        },
        "numeric_summary": numeric_summary,
        "categorical_changes": categorical_changes,
        "top_contributing_features": top_contributing_features,
        "raw_data_reference": {
            "table": raw_table,
            "available": raw_exists,
            "query": {
                "start_row_id": start_row_id,
                "end_row_id": end_row_id,
            },
        },
    }


def _get_rows(
    table_name: str,
    dataset_id: int,
    start_row_id: int,
    end_row_id: int,
    *,
    db_path: str | Path | None,
) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        if not table_exists(conn, table_name):
            raise ValueError(f"Required table {table_name} does not exist.")
        rows = conn.execute(
            f"""
            SELECT *
            FROM {_quote_identifier(table_name)}
            WHERE dataset_id = ?
              AND row_id BETWEEN ? AND ?
            ORDER BY row_id
            """,
            (dataset_id, start_row_id, end_row_id),
        ).fetchall()
    return [_json_ready_row(dict(row)) for row in rows]


def _get_rows_if_exists(
    table_name: str,
    dataset_id: int,
    start_row_id: int,
    end_row_id: int,
    *,
    db_path: str | Path | None,
) -> list[dict[str, Any]]:
    if not _table_exists(table_name, db_path=db_path):
        return []
    return _get_rows(
        table_name,
        dataset_id,
        start_row_id,
        end_row_id,
        db_path=db_path,
    )


def _numeric_summary(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    if not rows:
        return summary

    available = set(rows[0].keys())
    for feature in feature_names:
        if feature not in available:
            continue
        values = [_to_float(row.get(feature)) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            continue
        start = values[0]
        end = values[-1]
        delta = end - start
        percent_change = None if start == 0 else (delta / abs(start)) * 100
        summary[feature] = {
            "start": start,
            "end": end,
            "min": min(values),
            "max": max(values),
            "delta": delta,
            "percent_change": percent_change,
        }
    return summary


def _categorical_changes(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    changes: dict[str, list[Any]] = {}
    if not rows:
        return changes

    excluded = {"row_id", "dataset_id", "timestamp"}
    for column in rows[0].keys():
        if column in excluded:
            continue
        values = [row.get(column) for row in rows]
        compact_values = []
        seen = set()
        for value in values:
            marker = json.dumps(value, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            compact_values.append(value)
        if len(compact_values) > 1:
            changes[column] = compact_values
    return changes


def _table_exists(table_name: str, *, db_path: str | Path | None) -> bool:
    with get_connection(db_path) as conn:
        return table_exists(conn, table_name)


def _json_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, dict) else {}


def _json_list(payload: str | None) -> list[Any]:
    if not payload:
        return []
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, list) else []


def _json_ready_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_ready_value(value) for key, value in row.items()}


def _json_ready_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
