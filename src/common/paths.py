from __future__ import annotations

from pathlib import Path

# I should probably put this in .env file
V2_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = V2_ROOT / "data"
DB_DIR = V2_ROOT / "db"
DB_PATH = DB_DIR / "satellite_operator.db"
MODELS_DIR = V2_ROOT / "models"
CONFIG_DIR = V2_ROOT / "config"
GENERATED_CONFIG_DIR = CONFIG_DIR / "generated"
MODEL_CONFIG_DIR = CONFIG_DIR / "model_configs"
RAG_DIR = V2_ROOT / "rag"
RAG_DOCS_DIR = RAG_DIR / "documents"
RAG_VECTOR_DIR = RAG_DIR / "vector_store"
LOGS_DIR = V2_ROOT / "logs"
DIAGNOSTICS_DIR = V2_ROOT / "diagnostics"
EVALUATION_DIR = V2_ROOT / "evaluation"


def ensure_base_directories() -> None:
    for p in [
        DATA_DIR,
        DB_DIR,
        MODELS_DIR,
        GENERATED_CONFIG_DIR,
        MODEL_CONFIG_DIR,
        RAG_DOCS_DIR,
        RAG_VECTOR_DIR,
        LOGS_DIR,
        DIAGNOSTICS_DIR,
        EVALUATION_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
