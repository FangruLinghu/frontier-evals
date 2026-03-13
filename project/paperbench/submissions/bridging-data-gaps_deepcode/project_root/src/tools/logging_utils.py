"""
Lightweight logging utilities for experiments.

Exposes:
- get_logger(name, log_dir='logs', level=logging.INFO, console=True): returns a configured Python logger writing to a file and optionally to stdout.
- log_hparams(hparams, path=None): saves hyperparameters as JSON to path (defaults to hparams.json in cwd).
"""

import os
import json
import logging
import sys
from typing import Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO, console: bool = True) -> logging.Logger:
    """
    Create or retrieve a module-level logger with file and optional console output.

    - Logs are written to a timestamped file under log_dir
    - Optionally also prints to stdout
    - Prevents duplicate handlers if called multiple times for the same name
    """
    _ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        # Avoid propagating to root logger to prevent duplicate logs
        logger.propagate = False

        log_file = os.path.join(log_dir, f"{name}_{_timestamp()}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

    return logger


def log_hparams(hparams: dict, path: Optional[str] = None) -> str:
    """
    Persist hyperparameters dictionary to disk as JSON.

    Returns the path where the JSON was written.
    """
    if path is None:
        path = "hparams.json"
    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)
    return path


__all__ = ["get_logger", "log_hparams"]
