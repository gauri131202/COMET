from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LAYOUTS = {"wide", "long"}
FILTER_OPS = {"eq", "ne", "in", "not_in", "not_empty", "empty"}
DEFAULT_CHUNK_ROWS = 100_000


@dataclass(frozen=True)
class TimeSpec:
    column: str
    format: str


@dataclass(frozen=True)
class RowFilter:
    column: str
    op: str
    value: Any = None


@dataclass(frozen=True)
class LongColumns:
    """Column names in a long-format file. Only `series` is required."""

    series: str
    value_eu: str | None = None
    value_dn: str | None = None
    status: str | None = None
    dn_alarm: str | None = None
    eu_alarm: str | None = None
    session: str | None = None
    kind: str | None = None


@dataclass(frozen=True)
class SourceMapping:
    name: str
    layout: str
    files: str
    time: TimeSpec
    long: LongColumns | None = None
    series_columns: list[str] | None = None
    filters: list[RowFilter] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)
    context_columns: list[str] = field(default_factory=list)
    chunk_rows: int = DEFAULT_CHUNK_ROWS


def load_mapping(path: str | Path) -> SourceMapping:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Mapping file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_mapping(raw, source=str(path))


def parse_mapping(raw: Any, *, source: str = "mapping") -> SourceMapping:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top level must be a mapping")
    _reject_unknown(
        raw,
        {
            "name", "layout", "files", "time", "long", "series_columns",
            "filters", "exclude_columns", "context_columns", "chunk_rows",
        },
        source,
    )

    name = _require_str(raw, "name", source)
    layout = _require_str(raw, "layout", source)
    if layout not in LAYOUTS:
        raise ValueError(f"{source}: layout must be one of {sorted(LAYOUTS)}, got {layout!r}")

    time = _parse_time(raw.get("time"), source)
    long_columns = _parse_long(raw.get("long"), source)
    series_columns = _str_list(raw.get("series_columns"), "series_columns", source)

    if layout == "long":
        if long_columns is None:
            raise ValueError(f"{source}: layout 'long' requires a 'long' section")
        if series_columns is not None:
            raise ValueError(f"{source}: 'series_columns' applies only to layout 'wide'")
        if not (long_columns.value_eu or long_columns.value_dn or long_columns.status):
            raise ValueError(
                f"{source}: 'long' needs at least one of value_eu, value_dn or status"
            )
    else:
        if long_columns is not None:
            raise ValueError(f"{source}: 'long' section applies only to layout 'long'")

    exclude = _str_list(raw.get("exclude_columns"), "exclude_columns", source) or []
    context = _str_list(raw.get("context_columns"), "context_columns", source) or []
    overlap = set(exclude) & set(context)
    if overlap:
        raise ValueError(
            f"{source}: columns in both exclude_columns and context_columns: {sorted(overlap)}"
        )

    chunk_rows = raw.get("chunk_rows", DEFAULT_CHUNK_ROWS)
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError(f"{source}: chunk_rows must be a positive integer")

    return SourceMapping(
        name=name,
        layout=layout,
        files=_require_str(raw, "files", source),
        time=time,
        long=long_columns,
        series_columns=series_columns,
        filters=_parse_filters(raw.get("filters"), source),
        exclude_columns=exclude,
        context_columns=context,
        chunk_rows=chunk_rows,
    )


def _parse_time(raw: Any, source: str) -> TimeSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: 'time' section is required")
    _reject_unknown(raw, {"column", "format"}, f"{source}: time")
    return TimeSpec(
        column=_require_str(raw, "column", f"{source}: time"),
        format=_require_str(raw, "format", f"{source}: time"),
    )


def _parse_long(raw: Any, source: str) -> LongColumns | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: 'long' must be a mapping")
    keys = {"series", "value_eu", "value_dn", "status", "dn_alarm", "eu_alarm", "session", "kind"}
    _reject_unknown(raw, keys, f"{source}: long")
    values = {key: _optional_str(raw, key, f"{source}: long") for key in keys}
    if not values["series"]:
        raise ValueError(f"{source}: long.series is required")
    return LongColumns(**values)


def _parse_filters(raw: Any, source: str) -> list[RowFilter]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{source}: 'filters' must be a list")
    filters = []
    for index, item in enumerate(raw):
        where = f"{source}: filters[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be a mapping")
        _reject_unknown(item, {"column", "op", "value"}, where)
        op = _require_str(item, "op", where)
        if op not in FILTER_OPS:
            raise ValueError(f"{where}: op must be one of {sorted(FILTER_OPS)}, got {op!r}")
        value = item.get("value")
        if op in {"in", "not_in"} and not isinstance(value, list):
            raise ValueError(f"{where}: op {op!r} needs a list value")
        if op in {"eq", "ne"} and value is None:
            raise ValueError(f"{where}: op {op!r} needs a value")
        filters.append(RowFilter(column=_require_str(item, "column", where), op=op, value=value))
    return filters


def _reject_unknown(raw: dict, allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")


def _require_str(raw: dict, key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' is required and must be a non-empty string")
    return value.strip()


def _optional_str(raw: dict, key: str, where: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' must be a non-empty string when set")
    return value.strip()


def _str_list(value: Any, key: str, source: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{source}: '{key}' must be a list of non-empty strings")
    return [v.strip() for v in value]
