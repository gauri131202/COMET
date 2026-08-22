from __future__ import annotations

import csv
import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.common.paths import DATA_DIR, LOGS_DIR
from src.db import (
    get_connection,
    get_duplicate_dataset_file,
    init_db,
    insert_dataset,
    insert_dataset_file,
    update_dataset_row_count,
    utc_now,
)
from src.telemetry.csv_adapter import csv_rows_to_telemetry_records


@dataclass(frozen=True)
class CsvFileMetadata:
    file_name: str
    file_path: str
    file_size: int
    modified_time: str
    content_hash: str


@dataclass(frozen=True)
class CsvReadResult:
    headers: list[str]
    sanitized_headers: list[str]
    rows: list[list[str]]


def ingest_csvs(
    data_dir: str | Path = DATA_DIR,
    *,
    db_path: str | Path | None = None,
    logs_dir: str | Path = LOGS_DIR,
) -> dict[str, int]:
    init_db(db_path)
    logger = _build_logger(Path(logs_dir))
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    summary = {"ingested": 0, "skipped_duplicates": 0, "failed": 0}
    csv_files = sorted(path for path in data_path.iterdir() if path.is_file() and path.suffix.lower() == ".csv")

    if not csv_files:
        logger.info("No CSV files found in %s", data_path)
        return summary

    for csv_path in csv_files:
        try:
            metadata = _file_metadata(csv_path)
            duplicate = get_duplicate_dataset_file(
                file_name=metadata.file_name,
                file_path=metadata.file_path,
                file_size=metadata.file_size,
                modified_time=metadata.modified_time,
                content_hash=metadata.content_hash,
                db_path=db_path,
            )
            if duplicate is not None:
                logger.info(
                    "Skipped duplicate CSV %s; existing dataset_id=%s source_file_id=%s",
                    csv_path.name,
                    duplicate["dataset_id"],
                    duplicate["source_file_id"],
                )
                summary["skipped_duplicates"] += 1
                continue

            csv_data = _read_and_validate_csv(csv_path)
            dataset_id = insert_dataset(
                csv_path.stem,
                status="ingesting",
                row_count=0,
                metadata={"source_file": metadata.file_name},
                db_path=db_path,
            )
            source_file_id = insert_dataset_file(
                dataset_id,
                metadata.file_path,
                file_size=metadata.file_size,
                modified_time=metadata.modified_time,
                content_hash=metadata.content_hash,
                row_count=len(csv_data.rows),
                status="ingested",
                db_path=db_path,
            )
            _create_raw_table(
                dataset_id,
                source_file_id,
                csv_data,
                source_file=metadata.file_path,
                db_path=db_path,
            )
            update_dataset_row_count(dataset_id, len(csv_data.rows), status="ingested", db_path=db_path)

            logger.info(
                "Ingested CSV %s as dataset_id=%s source_file_id=%s rows=%s",
                csv_path.name,
                dataset_id,
                source_file_id,
                len(csv_data.rows),
            )
            summary["ingested"] += 1
        except Exception as exc:
            logger.error("Failed to ingest CSV %s: %s", csv_path.name, exc)
            summary["failed"] += 1

    return summary


def sanitize_column_names(headers: Iterable[str]) -> list[str]:
    sanitized: list[str] = []
    seen: dict[str, int] = {}
    reserved = {"row_id", "dataset_id", "source_file_id"}

    for index, header in enumerate(headers, start=1):
        name = re.sub(r"\W+", "_", header.strip().lower()).strip("_")
        if not name:
            name = f"column_{index}"
        if name[0].isdigit():
            name = f"column_{name}"
        if name in reserved:
            name = f"csv_{name}"

        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{name}_{count + 1}"
        sanitized.append(name)

    return sanitized


def _build_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("satellite_operator.ingestion")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = (logs_dir / "ingestion.log").resolve()
    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    return logger


def _file_metadata(csv_path: Path) -> CsvFileMetadata:
    stat = csv_path.stat()
    return CsvFileMetadata(
        file_name=csv_path.name,
        file_path=str(csv_path.resolve()),
        file_size=stat.st_size,
        modified_time=str(stat.st_mtime_ns),
        content_hash=_sha256(csv_path),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_and_validate_csv(csv_path: Path) -> CsvReadResult:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, strict=True)
            try:
                headers = next(reader)
            except StopIteration as exc:
                raise ValueError("CSV is empty") from exc

            if not headers or all(not header.strip() for header in headers):
                raise ValueError("CSV is missing headers")
            if any(not header.strip() for header in headers):
                raise ValueError("CSV contains blank header names")

            rows: list[list[str]] = []
            expected_length = len(headers)
            for line_number, row in enumerate(reader, start=2):
                if len(row) != expected_length:
                    raise ValueError(
                        f"row {line_number} has {len(row)} values; expected {expected_length}"
                    )
                rows.append(row)

    except csv.Error as exc:
        raise ValueError(f"CSV parse error: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV is not readable as UTF-8 text: {exc}") from exc

    if not rows:
        raise ValueError("CSV has headers but no data rows")

    return CsvReadResult(
        headers=headers,
        sanitized_headers=sanitize_column_names(headers),
        rows=rows,
    )


def _create_raw_table(
    dataset_id: int,
    source_file_id: int,
    csv_data: CsvReadResult,
    *,
    source_file: str | Path,
    db_path: str | Path | None,
) -> None:
    table_name = f"raw_dataset_{dataset_id}"
    quoted_table = _quote_identifier(table_name)
    csv_columns_sql = ",\n                ".join(
        f"{_quote_identifier(column_name)} TEXT" for column_name in csv_data.sanitized_headers
    )
    insert_columns = ["dataset_id", "source_file_id", *csv_data.sanitized_headers]
    quoted_insert_columns = ", ".join(_quote_identifier(column_name) for column_name in insert_columns)
    placeholders = ", ".join("?" for _ in insert_columns)
    records = csv_rows_to_telemetry_records(
        headers=csv_data.headers,
        sanitized_headers=csv_data.sanitized_headers,
        rows=csv_data.rows,
        source_file=source_file,
        dataset_id=dataset_id,
        source_file_id=source_file_id,
        received_at=utc_now(),
    )

    with get_connection(db_path) as conn:
        try:
            conn.execute("BEGIN")
            conn.execute(
                f"""
                CREATE TABLE {quoted_table} (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    source_file_id INTEGER NOT NULL,
                    {csv_columns_sql}
                )
                """
            )
            conn.executemany(
                f"INSERT INTO {quoted_table} ({quoted_insert_columns}) VALUES ({placeholders})",
                (
                    [
                        dataset_id,
                        source_file_id,
                        *(record.telemetry[column] for column in csv_data.sanitized_headers),
                    ]
                    for record in records
                ),
            )
            _insert_column_metadata(conn, dataset_id, csv_data)
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise


def _insert_column_metadata(
    conn: sqlite3.Connection,
    dataset_id: int,
    csv_data: CsvReadResult,
) -> None:
    now = utc_now()
    conn.executemany(
        """
        INSERT INTO dataset_columns (
            dataset_id, column_name, sanitized_name, inferred_type, role,
            created_at, updated_at, reasoning
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                dataset_id,
                original,
                sanitized,
                "unknown",
                "unassigned",
                now,
                now,
                "Recorded during raw CSV ingestion; profiling not yet run.",
            )
            for original, sanitized in zip(csv_data.headers, csv_data.sanitized_headers, strict=True)
        ),
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
