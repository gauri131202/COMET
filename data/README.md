# data

Put telemetry CSV files here before running ingestion.

Overview:

- Input files: COMET reads every `.csv` file directly under this directory when you run `python -m src.cli ingest`.
- Data source: files can be CATs telemetry or real mission telemetry.
- Repository rule: do not commit mission data.
