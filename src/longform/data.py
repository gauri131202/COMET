from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch

from src.modeling.lstm_autoencoder import create_sliding_windows

# Long-form input for the autoencoder: windows of consecutive readings of ONE series, read from
# `samples`. Each step is (z-scored value, scaled log time since the previous reading).
# Window size, stride and the training settings come from the usual training config.

FEATURE_COLUMNS = ["value", "time_since_previous"]
EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class LongFormConfig:
    value: str = "eu_else_dn"  # eu_else_dn | dn | eu
    series: list[str] | None = None
    start: str | None = None
    end: str | None = None
    min_readings: int = 200
    max_gap_seconds: float = 3600.0
    normalize_fraction: float = 0.2
    outlier_factor: float = 1e6
    max_windows_per_series: int = 500


@dataclass
class SeriesReadings:
    series_id: int
    key: str
    ts: list[str]
    seconds: list[float]
    values: array


def numeric_series(conn, dataset_id: int, config: LongFormConfig) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT series_id, series_key FROM series_catalog "
        "WHERE dataset_id = ? AND kind = 'numeric' ORDER BY series_id",
        (dataset_id,),
    ).fetchall()
    if config.series is not None:
        missing = sorted(set(config.series) - {r["series_key"] for r in rows})
        if missing:
            raise ValueError(f"Numeric series not in the catalog for dataset {dataset_id}: {missing[:10]}")
        rows = [r for r in rows if r["series_key"] in config.series]
    if not rows:
        raise ValueError(f"No numeric series for dataset {dataset_id}.")
    return [(r["series_id"], r["series_key"]) for r in rows]


def load_readings(conn, dataset_id: int, series_id: int, key: str, config: LongFormConfig) -> SeriesReadings | None:
    """One series in time order: same-time duplicates keep the last reading, absurd outliers dropped."""
    sql = "SELECT ts, value_eu, value_dn FROM samples WHERE dataset_id = ? AND series_id = ?"
    params: list[Any] = [dataset_id, series_id]
    if config.start:
        sql += " AND ts >= ?"
        params.append(_iso(config.start))
    if config.end:
        sql += " AND ts <= ?"
        params.append(_iso(config.end))
    rows = conn.execute(sql + " ORDER BY ts, sample_id", params).fetchall()

    use_eu = config.value == "eu" or (
        config.value == "eu_else_dn" and rows and all(r["value_eu"] is not None for r in rows)
    )
    column = "value_eu" if use_eu else "value_dn"
    last: dict[str, float] = {}
    for row in rows:
        if row[column] is not None and math.isfinite(row[column]):
            last[row["ts"]] = row[column]
    if not last:
        return None

    stamps, values = list(last), list(last.values())
    low, high = _outlier_bounds(values, config.outlier_factor)
    kept = [(t, v) for t, v in zip(stamps, values) if low <= v <= high]
    if len(kept) < config.min_readings:
        return None
    ts = [t for t, _ in kept]
    return SeriesReadings(series_id, key, ts, [_seconds(t) for t in ts], array("d", [v for _, v in kept]))


def fit_normalization(readings: SeriesReadings, fraction: float) -> tuple[float, float]:
    head = readings.values[: max(2, math.ceil(fraction * len(readings.values)))]
    mean = sum(head) / len(head)
    std = math.sqrt(sum((v - mean) ** 2 for v in head) / len(head))
    return mean, (std if std > 0 else 1.0)


def build_windows(
    readings: SeriesReadings, mean: float, std: float, config: LongFormConfig, window_size: int, stride: int
) -> torch.Tensor | None:
    """Sliding windows over each gap-free run of readings, so a window never crosses a long gap."""
    scale = math.log1p(config.max_gap_seconds)
    steps, runs, run_start = [], [], 0
    for i, value in enumerate(readings.values):
        dt = readings.seconds[i] - readings.seconds[i - 1] if i else 0.0
        if dt > config.max_gap_seconds:
            runs.append((run_start, i))
            run_start = i
        steps.append([(value - mean) / std, math.log1p(min(dt, config.max_gap_seconds)) / scale])
    runs.append((run_start, len(steps)))

    data = torch.tensor(steps, dtype=torch.float32)
    windows = [create_sliding_windows(data[a:b], window_size, stride) for a, b in runs]
    windows = torch.cat([w for w in windows if w.size(0)]) if any(w.size(0) for w in windows) else None
    if windows is not None:
        windows[:, 0, 1] = 0.0  # the first step of a window has no predecessor inside it
    return windows


def _outlier_bounds(values: list[float], factor: float) -> tuple[float, float]:
    """Wide bounds around the median: only absurd readings (1e30 on a channel near 1) fall outside."""
    ordered = sorted(values)
    last = len(ordered) - 1
    median = ordered[last // 2]
    spread = ordered[int(0.99 * last)] - ordered[int(0.01 * last)]
    if spread <= 0:
        spread = max(1.0, 0.001 * abs(median))
    return median - factor * spread, median + factor * spread


def _iso(text: str) -> str:
    return datetime.fromisoformat(text).isoformat(timespec="microseconds")


def _seconds(ts: str) -> float:
    return (datetime.fromisoformat(ts) - EPOCH).total_seconds()
