from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.db import get_anomaly, get_connection, get_dataset, init_db, table_exists


def get_raw_window(
    dataset_id: int,
    start_row_id: int,
    end_row_id: int,
    *,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]] | dict[str, str]:
    validation = _validate_window_request(
        dataset_id,
        start_row_id,
        end_row_id,
        f"raw_dataset_{dataset_id}",
        db_path=db_path,
    )
    if validation.get("error"):
        return validation
    return _select_window(f"raw_dataset_{dataset_id}", dataset_id, start_row_id, end_row_id, db_path=db_path)


def get_context_window(
    dataset_id: int,
    start_row_id: int,
    end_row_id: int,
    *,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]] | dict[str, str]:
    validation = _validate_window_request(
        dataset_id,
        start_row_id,
        end_row_id,
        f"context_ready_dataset_{dataset_id}",
        db_path=db_path,
    )
    if validation.get("error"):
        return validation
    return _select_window(
        f"context_ready_dataset_{dataset_id}",
        dataset_id,
        start_row_id,
        end_row_id,
        db_path=db_path,
    )


def get_column_summary(
    dataset_id: int,
    column_name: str,
    start_row_id: int,
    end_row_id: int,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    table_name = f"model_ready_dataset_{dataset_id}"
    validation = _validate_window_request(
        dataset_id,
        start_row_id,
        end_row_id,
        table_name,
        db_path=db_path,
    )
    if validation.get("error"):
        return validation
    column_validation = _validate_column(table_name, column_name, db_path=db_path)
    if column_validation.get("error"):
        return column_validation

    rows = _select_window(table_name, dataset_id, start_row_id, end_row_id, db_path=db_path)
    if isinstance(rows, dict):
        return rows

    values = [_to_float(row.get(column_name)) for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        return {
            "dataset_id": dataset_id,
            "column_name": column_name,
            "start_row_id": start_row_id,
            "end_row_id": end_row_id,
            "count": len(values),
            "missing_count": len(values),
            "min": None,
            "max": None,
            "mean": None,
            "first": None,
            "last": None,
            "delta": None,
        }

    first = next((value for value in values if value is not None), None)
    last = next((value for value in reversed(values) if value is not None), None)
    return {
        "dataset_id": dataset_id,
        "column_name": column_name,
        "start_row_id": start_row_id,
        "end_row_id": end_row_id,
        "count": len(values),
        "missing_count": len(values) - len(present),
        "min": min(present),
        "max": max(present),
        "mean": sum(present) / len(present),
        "first": first,
        "last": last,
        "delta": None if first is None or last is None else last - first,
    }


def get_rows_around_peak(
    dataset_id: int,
    anomaly_id: str,
    *,
    radius: int = 2,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]] | dict[str, str]:
    init_db(db_path)
    if not _valid_positive_int(dataset_id):
        return _error("invalid_dataset_id", "dataset_id must be a positive integer.")
    if not isinstance(anomaly_id, str) or not anomaly_id.strip():
        return _error("invalid_anomaly_id", "anomaly_id must be a non-empty string.")
    if radius < 0:
        return _error("invalid_radius", "radius must be non-negative.")
    if get_dataset(dataset_id, db_path=db_path) is None:
        return _error("dataset_not_found", f"Dataset {dataset_id} does not exist.")

    anomaly = get_anomaly(anomaly_id, db_path=db_path)
    if anomaly is None:
        return _error("anomaly_not_found", f"Anomaly {anomaly_id} does not exist.")
    if int(anomaly["dataset_id"]) != dataset_id:
        return _error(
            "dataset_mismatch",
            f"Anomaly {anomaly_id} belongs to dataset {anomaly['dataset_id']}, not {dataset_id}.",
        )

    packet = _json_dict(anomaly["packet_json"])
    peak_row_id = _peak_row_from_packet(packet)
    if peak_row_id is None:
        peak_row_id = (int(anomaly["start_row_id"]) + int(anomaly["end_row_id"])) // 2

    start_row_id = max(1, peak_row_id - radius)
    end_row_id = peak_row_id + radius
    validation = _validate_window_request(
        dataset_id,
        start_row_id,
        end_row_id,
        f"context_ready_dataset_{dataset_id}",
        db_path=db_path,
    )
    if validation.get("error"):
        return validation
    return _select_window(
        f"context_ready_dataset_{dataset_id}",
        dataset_id,
        start_row_id,
        end_row_id,
        db_path=db_path,
    )


def _validate_window_request(
    dataset_id: int,
    start_row_id: int,
    end_row_id: int,
    table_name: str,
    *,
    db_path: str | Path | None,
) -> dict[str, str]:
    init_db(db_path)
    if not _valid_positive_int(dataset_id):
        return _error("invalid_dataset_id", "dataset_id must be a positive integer.")
    if not _valid_positive_int(start_row_id) or not _valid_positive_int(end_row_id):
        return _error("invalid_row_range", "row IDs must be positive integers.")
    if start_row_id > end_row_id:
        return _error("invalid_row_range", "start_row_id must be <= end_row_id.")
    if get_dataset(dataset_id, db_path=db_path) is None:
        return _error("dataset_not_found", f"Dataset {dataset_id} does not exist.")
    with get_connection(db_path) as conn:
        if not table_exists(conn, table_name):
            return _error("table_not_found", f"Required table {table_name} does not exist.")
    return {}


def _validate_column(
    table_name: str,
    column_name: str,
    *,
    db_path: str | Path | None,
) -> dict[str, str]:
    if not isinstance(column_name, str) or not column_name.strip():
        return _error("invalid_column", "column_name must be a non-empty string.")
    with get_connection(db_path) as conn:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
        }
    if column_name not in columns:
        return _error("unknown_column", f"Column {column_name} does not exist in {table_name}.")
    return {}


def _select_window(
    table_name: str,
    dataset_id: int,
    start_row_id: int,
    end_row_id: int,
    *,
    db_path: str | Path | None,
) -> list[dict[str, Any]] | dict[str, str]:
    with get_connection(db_path) as conn:
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


def _peak_row_from_packet(packet: dict[str, Any]) -> int | None:
    for key in ("peak_row_id", "peak_row"):
        value = packet.get(key)
        if _valid_positive_int(value):
            return int(value)
    return None


def _valid_positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _error(code: str, message: str) -> dict[str, str]:
    return {"error": code, "message": message}


def _json_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
