from __future__ import annotations

import csv
import json
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from src.common.paths import DATA_DIR, LOGS_DIR
from src.db import get_connection, init_db, insert_dataset, update_dataset_row_count, utc_now
from src.ingestion.adapter_validation import SourceMapping
from src.ingestion.catalog import ObservedSeries, build_catalog_rows, write_catalog
from src.ingestion.csv_ingestion import _build_logger, _file_metadata
from src.ingestion.dictionary import TelemetryDictionary, read_telemetry_dictionary
from src.ingestion.values import parse_number

_ALREADY_DONE = object()

INSERT_SAMPLE = """
    INSERT INTO samples (
        dataset_id, source_file_id, series_id, session_id, line_no, ts,
        value_eu, value_dn, status, dn_alarm, eu_alarm, extra_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass(frozen=True)
class _Columns:
    """Positions of the mapped columns in one file's header."""

    time: int
    series: int
    value_eu: int | None
    value_dn: int | None
    status: int | None
    dn_alarm: int | None
    eu_alarm: int | None
    session: int | None
    kind: int | None
    name: int | None
    context: dict[str, int]

    @classmethod
    def from_header(cls, header: list[str], mapping: SourceMapping) -> _Columns:
        long = mapping.long
        position = {name: index for index, name in enumerate(header)}
        wanted = [mapping.time.column, long.series, *mapping.context_columns]
        wanted += [
            column
            for column in (
                long.value_eu, long.value_dn, long.status, long.dn_alarm,
                long.eu_alarm, long.session, long.kind, long.name,
            )
            if column
        ]
        missing = sorted({column for column in wanted if column not in position})
        if missing:
            raise ValueError(f"missing columns {missing}")

        def at(column: str | None) -> int | None:
            return position[column] if column else None

        return cls(
            time=position[mapping.time.column],
            series=position[long.series],
            value_eu=at(long.value_eu),
            value_dn=at(long.value_dn),
            status=at(long.status),
            dn_alarm=at(long.dn_alarm),
            eu_alarm=at(long.eu_alarm),
            session=at(long.session),
            kind=at(long.kind),
            name=at(long.name),
            context={column: position[column] for column in mapping.context_columns},
        )


@dataclass
class _FileStats:
    rows_read: int = 0
    rows_loaded: int = 0
    rows_bad: int = 0
    sessions: set[str] = field(default_factory=set)


def ingest_long(
    mapping: SourceMapping,
    *,
    data_dir: str | Path = DATA_DIR,
    db_path: str | Path | None = None,
    logs_dir: str | Path = LOGS_DIR,
    dataset_id: int | None = None,
) -> dict[str, int]:
    """Load every file matching mapping.files into one dataset, as canonical samples.

    Each file is loaded in a single transaction, so a failed file leaves nothing behind.
    Pass dataset_id to continue an earlier run: files already ingested there are skipped.
    """
    if mapping.layout != "long" or mapping.long is None:
        raise ValueError(f"{mapping.name}: ingest_long needs a mapping with layout 'long'")

    data_path = Path(data_dir)
    files = sorted(path for path in data_path.glob(mapping.files) if path.is_file())
    if not files:
        raise ValueError(f"No files match {data_path / mapping.files}")

    init_db(db_path)
    logger = _build_logger(Path(logs_dir))
    dictionary = _read_dictionary(mapping, data_path)

    observed = observe_series(files, mapping, logger)
    rows = build_catalog_rows(mapping, dictionary, observed)

    if dataset_id is None:
        dataset_id = insert_dataset(
            mapping.name, status="ingesting", metadata={"mapping": mapping.name}, db_path=db_path
        )
    series_ids = write_catalog(dataset_id, rows, db_path)

    summary = {
        "dataset_id": dataset_id,
        "series": len(series_ids),
        "files_ingested": 0,
        "files_skipped_empty": 0,
        "files_skipped_done": 0,
        "files_failed": 0,
        "rows_loaded": 0,
        "rows_bad": 0,
    }
    for path in files:
        try:
            stats = _ingest_file(path, mapping, dataset_id, series_ids, db_path, logger)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path.name, exc)
            summary["files_failed"] += 1
            continue
        if stats is None:
            summary["files_skipped_empty"] += 1
        elif stats is _ALREADY_DONE:
            summary["files_skipped_done"] += 1
        else:
            summary["files_ingested"] += 1
            summary["rows_loaded"] += stats.rows_loaded
            summary["rows_bad"] += stats.rows_bad

    with closing(get_connection(db_path)) as conn:
        total = conn.execute(
            "SELECT COALESCE(SUM(row_count), 0) FROM dataset_files "
            "WHERE dataset_id = ? AND status = 'ingested'",
            (dataset_id,),
        ).fetchone()[0]
    status = "partial" if summary["files_failed"] else "ingested"
    update_dataset_row_count(dataset_id, int(total), status=status, db_path=db_path)
    logger.info("Ingested dataset_id=%s (%s): %s", dataset_id, status, summary)
    return summary


def observe_series(files: list[Path], mapping: SourceMapping, logger) -> dict[str, ObservedSeries]:
    """First pass: the type and name spellings each series shows across the files."""
    types: dict[str, str | None] = {}
    names: dict[str, dict[str, None]] = {}
    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header is None:
                    continue
                cols = _Columns.from_header(header, mapping)
                for row in reader:
                    try:
                        key = row[cols.series].strip()
                        if not key:
                            continue
                        source_type = row[cols.kind].strip() if cols.kind is not None else ""
                        name = row[cols.name].strip() if cols.name is not None else ""
                    except IndexError:
                        continue
                    if types.get(key) is None:
                        types[key] = source_type or None
                    spellings = names.setdefault(key, {})
                    if name:
                        spellings[name] = None
        except ValueError as exc:
            logger.warning("Skipped %s while observing series: %s", path.name, exc)
    return {
        key: ObservedSeries(source_type=types[key], names=list(names[key])) for key in types
    }


def _ingest_file(
    path: Path,
    mapping: SourceMapping,
    dataset_id: int,
    series_ids: dict[str, int],
    db_path: str | Path | None,
    logger,
):
    """Returns None for an empty file, _ALREADY_DONE when already loaded, else the stats."""
    if path.stat().st_size == 0:
        logger.warning("Skipped empty file %s", path.name)
        return None
    metadata = _file_metadata(path)

    with closing(get_connection(db_path)) as conn:
        done = conn.execute(
            "SELECT 1 FROM dataset_files WHERE dataset_id = ? AND file_path = ? "
            "AND content_hash = ? AND status = 'ingested'",
            (dataset_id, metadata.file_path, metadata.content_hash),
        ).fetchone()
        if done:
            logger.info("Skipped %s: already ingested in dataset_id=%s", path.name, dataset_id)
            return _ALREADY_DONE

        try:
            conn.execute("BEGIN")
            stats = _load_file(conn, path, metadata, mapping, dataset_id, series_ids)
            if stats is None:
                conn.rollback()
                logger.warning("Skipped %s: no header row", path.name)
                return None
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    logger.info(
        "Ingested %s: loaded=%s bad=%s", path.name, stats.rows_loaded, stats.rows_bad,
    )
    return stats


def _load_file(conn, path, metadata, mapping, dataset_id, series_ids) -> _FileStats | None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return None
        try:
            cols = _Columns.from_header(header, mapping)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc

        conn.execute(
            "DELETE FROM samples WHERE source_file_id IN ("
            "SELECT source_file_id FROM dataset_files WHERE dataset_id = ? AND file_path = ?)",
            (dataset_id, metadata.file_path),
        )
        conn.execute(
            "DELETE FROM dataset_files WHERE dataset_id = ? AND file_path = ?",
            (dataset_id, metadata.file_path),
        )
        source_file_id = conn.execute(
            """
            INSERT INTO dataset_files (
                dataset_id, file_name, file_path, file_size, modified_time,
                content_hash, row_count, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 'registered', ?)
            """,
            (
                dataset_id, metadata.file_name, metadata.file_path, metadata.file_size,
                metadata.modified_time, metadata.content_hash, utc_now(),
            ),
        ).lastrowid

        stats = _FileStats()
        samples = _sample_rows(reader, cols, mapping, dataset_id, source_file_id, series_ids, stats)
        while chunk := list(islice(samples, mapping.chunk_rows)):
            conn.executemany(INSERT_SAMPLE, chunk)

    file_metadata = {
        "mapping": mapping.name,
        "rows_read": stats.rows_read,
        "rows_loaded": stats.rows_loaded,
        "rows_bad": stats.rows_bad,
        "sessions": sorted(stats.sessions),
    }
    conn.execute(
        "UPDATE dataset_files SET row_count = ?, status = 'ingested', file_metadata_json = ? "
        "WHERE source_file_id = ?",
        (stats.rows_loaded, json.dumps(file_metadata, sort_keys=True), source_file_id),
    )
    return stats


def _sample_rows(
    reader, cols: _Columns, mapping: SourceMapping, dataset_id: int, source_file_id: int,
    series_ids: dict[str, int], stats: _FileStats,
) -> Iterator[tuple]:
    time_format = mapping.time.format
    for row in reader:
        stats.rows_read += 1
        line_no = reader.line_num
        try:
            key = row[cols.series].strip()
            ts = datetime.strptime(row[cols.time].strip(), time_format).isoformat(
                timespec="microseconds"
            )
            if not key:
                raise ValueError("empty series")
            extra: dict[str, Any] = {}
            for column, index in cols.context.items():
                if row[index].strip():
                    extra[column] = row[index].strip()
            value_eu = _number(row, cols.value_eu, "eu", extra)
            value_dn = _number(row, cols.value_dn, "dn", extra)
            session = _text(row, cols.session)
            sample = (
                dataset_id,
                source_file_id,
                series_ids[key],
                session,
                line_no,
                ts,
                value_eu,
                value_dn,
                _text(row, cols.status),
                _text(row, cols.dn_alarm),
                _text(row, cols.eu_alarm),
                json.dumps(extra, sort_keys=True) if extra else "{}",
            )
        except (IndexError, ValueError):  # short row, unreadable time or empty series
            stats.rows_bad += 1
            continue
        if session is not None:
            stats.sessions.add(session)
        stats.rows_loaded += 1
        yield sample


def _number(row: list[str], index: int | None, label: str, extra: dict[str, Any]) -> float | None:
    """Value as a float. Text that is not a finite number is kept in extra and the value is NULL."""
    if index is None:
        return None
    text = row[index]
    value = parse_number(text)
    if value is None and text.strip():
        extra[f"{label}_text"] = text.strip()
    return value


def _text(row: list[str], index: int | None) -> str | None:
    if index is None:
        return None
    text = row[index].strip()
    return text or None


def _read_dictionary(mapping: SourceMapping, data_path: Path) -> TelemetryDictionary | None:
    spec = mapping.dictionary
    if spec is None:
        return None
    alarms = data_path / spec.alarms if spec.alarms else None
    return read_telemetry_dictionary(data_path / spec.channels, alarms)
