# src/ingestion

CSV ingestion code.

Overview:
- `csv_ingestion.py`: reads CSV files from `data/`, registers them in SQLite, and creates raw dataset tables

Place different input files in `data/` before running `python -m src.cli ingest`.