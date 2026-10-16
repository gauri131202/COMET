from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.db import get_connection, init_db
from src.ingestion.adapter_validation import SourceMapping
from src.ingestion.dictionary import TelemetryDictionary


@dataclass(frozen=True)
class ObservedSeries:
    """What the data files say about one series."""

    source_type: str | None = None
    names: list[str] = field(default_factory=list)


def subsystem_from_key(series_key: str) -> str | None:
    """Text before the first '-' in the series key, e.g. 'PEPS-0014' -> 'PEPS'."""
    prefix, dash, _ = series_key.partition("-")
    return prefix if dash and prefix else None


def build_catalog_rows(
    mapping: SourceMapping,
    dictionary: TelemetryDictionary | None,
    observed: dict[str, ObservedSeries],
) -> list[dict[str, Any]]:
    """One row per series: every dictionary channel plus every observed series.

    Kind comes from the data's source type first, then the dictionary's type.
    """
    entries = dictionary.entries if dictionary else {}
    keys = list(entries) + [key for key in observed if key not in entries]

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for key in keys:
        entry = entries.get(key)
        seen = observed.get(key)
        source_type = seen.source_type if seen else None
        type_for_kind = source_type or (entry.dictionary_type if entry else None)
        kind = _kind(mapping.kind_map, type_for_kind)
        if kind is None:
            problems.append(f"{key} (type {type_for_kind!r})")
            continue

        csv_names = list(dict.fromkeys(seen.names)) if seen else []
        extra: dict[str, Any] = {"in_dictionary": entry is not None}
        if csv_names:
            extra["csv_names"] = csv_names
        if entry:
            if entry.dictionary_type:
                extra["dictionary_type"] = entry.dictionary_type
            extra.update(entry.extra)

        rows.append(
            {
                "series_key": key,
                "display_name": (entry.name if entry and entry.name else None)
                or (csv_names[0] if csv_names else key),
                "source_type": source_type,
                "kind": kind,
                "subsystem": subsystem_from_key(key),
                "description": entry.description if entry else None,
                "units_eu": entry.units_eu if entry else None,
                "units_dn": entry.units_dn if entry else None,
                "extra_json": json.dumps(extra, sort_keys=True),
            }
        )

    if problems:
        shown = ", ".join(problems[:10])
        raise ValueError(
            f"Cannot determine kind for {len(problems)} series (add the type to kind_map): {shown}"
        )
    return rows


def write_catalog(
    dataset_id: int, rows: list[dict[str, Any]], db_path: str | Path | None = None
) -> dict[str, int]:
    """Insert or update catalog rows for a dataset. Returns series_key -> series_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO series_catalog (
                dataset_id, series_key, display_name, source_type, kind, subsystem,
                description, units_eu, units_dn, extra_json
            ) VALUES (
                :dataset_id, :series_key, :display_name, :source_type, :kind, :subsystem,
                :description, :units_eu, :units_dn, :extra_json
            )
            ON CONFLICT(dataset_id, series_key) DO UPDATE SET
                display_name = excluded.display_name,
                source_type = excluded.source_type,
                kind = excluded.kind,
                subsystem = excluded.subsystem,
                description = excluded.description,
                units_eu = excluded.units_eu,
                units_dn = excluded.units_dn,
                extra_json = excluded.extra_json
            """,
            [{**row, "dataset_id": dataset_id} for row in rows],
        )
        ids = {
            r["series_key"]: r["series_id"]
            for r in conn.execute(
                "SELECT series_id, series_key FROM series_catalog WHERE dataset_id = ?",
                (dataset_id,),
            )
        }
        conn.commit()
    return ids


def _kind(kind_map: dict[str, str], source_type: str | None) -> str | None:
    if source_type is None:
        return None
    return kind_map.get(source_type.strip().upper())
