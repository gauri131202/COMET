from __future__ import annotations

import logging
from pathlib import Path
from src.common.paths import LOGS_DIR


def get_app_logger(logs_dir: str | Path = LOGS_DIR) -> logging.Logger:
    log_path = Path(logs_dir) / "satellite_operator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("satellite_operator")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved_log_path = log_path.resolve()
    has_handler = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == resolved_log_path
        for handler in logger.handlers
    )
    if not has_handler:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    return logger
