from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.paths import DATA_DIR
from src.cleaning.prepare_dataset import prepare_dataset
from src.db import create_samples_index, get_connection, init_db
from src.ingestion.adapter_validation import load_mapping
from src.ingestion.csv_ingestion import ingest_csvs
from src.ingestion.long_adapter import ingest_long
from src.longform.data import LongFormConfig
from src.longform.train import train_longform
from src.modeling.train_model import train_model
from src.profiling.dataset_profile import profile_dataset


def run_pipeline(
    mapping_path: str | Path,
    train_config: str | Path | None = None,
    *,
    long_config: LongFormConfig | None = None,
    data_dir: str | Path = DATA_DIR,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """CSV files to a trained autoencoder. The mapping YAML's `layout` picks the path.

    wide: ingest, profile, prepare, train (the existing steps, unchanged).
    long: load into samples, index, train on per-series windows.
    """
    mapping = load_mapping(mapping_path)
    init_db(db_path)

    if mapping.layout == "wide":
        dataset_ids = ingest_csvs(data_dir, db_path=db_path)["dataset_ids"]
        dataset_id = dataset_ids.get(Path(mapping.files).name)
        if dataset_id is None:
            raise ValueError(f"{mapping.files} was not ingested (see the ingestion log).")
        profile_dataset(dataset_id, db_path=db_path)
        prepare_dataset(dataset_id, db_path=db_path)
        result = train_model(dataset_id, train_config, db_path=db_path)
    else:
        # continue this source's earlier dataset when there is one, so files already loaded are skipped
        dataset_id = ingest_long(
            mapping, data_dir=data_dir, db_path=db_path, dataset_id=_existing_long_dataset(mapping.name, db_path)
        )["dataset_id"]
        create_samples_index(db_path)
        result = train_longform(dataset_id, train_config, long_config=long_config, db_path=db_path)
    return {"layout": mapping.layout, **result}


def _existing_long_dataset(name: str, db_path) -> int | None:
    """Newest dataset with this name that was loaded into samples (it has catalog rows)."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT dataset_id FROM datasets WHERE name = ? AND dataset_id IN "
            "(SELECT dataset_id FROM series_catalog) ORDER BY dataset_id DESC LIMIT 1",
            (name,),
        ).fetchone()
    return int(row["dataset_id"]) if row else None


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.pipeline", description=run_pipeline.__doc__)
    parser.add_argument("mapping", help="Mapping YAML, for example config/ingestion/cats.yaml")
    parser.add_argument("--train-config", help="Training config. Defaults to config/model_configs/default.yaml.")
    args = parser.parse_args()
    try:
        result = run_pipeline(args.mapping, args.train_config)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
