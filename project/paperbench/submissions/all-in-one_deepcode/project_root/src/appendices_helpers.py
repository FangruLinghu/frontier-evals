# Appendix utilities: lightweight helpers for Appendix-like experiments
"""appendices_helpers.py

This module provides small, deterministic utilities to support Appendix-style
experiments used in the diffusion-score project. The helpers are intentionally
minimal and dependency-light, focusing on reproducibility, configuration handling
and small data-processing aids commonly required when running supplementary experiments.

Public API:
- seed_everything(seed: int) -> None
- ensure_dir(path: str) -> None
- update_dict(d: dict, updates: dict) -> None
- load_yaml(path: str) -> Any | None
- save_yaml(data: Any, path: str) -> None
- flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict
- __all__ listing for explicit exports
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency safety
    yaml = None  # type: ignore


__all__ = [
    "seed_everything",
    "ensure_dir",
    "update_dict",
    "load_yaml",
    "save_yaml",
    "flatten_dict",
]


def seed_everything(seed: int) -> None:
    """Seed Python's, NumPy's and optionally PyTorch's RNGs for reproducibility.

    This helper aims to cover common libraries used in experimentation. If PyTorch
    is installed, it will attempt to seed its RNGs as well; failures are ignored
    gracefully to keep this function lightweight.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    # Optional PyTorch seeding for compatibility; best-effort and non-fatal if PyTorch isn't installed
    try:  # pragma: no cover
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist.

    path may be a full path or a directory path. If intermediate directories are
    missing, they will be created as well.
    """
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def update_dict(d: Dict[Any, Any], updates: Dict[Any, Any]) -> None:
    """Recursively update dictionary d with values from updates.

    Nested dictionaries are updated in-place. Non-dict values in updates overwrite
    existing values.
    """
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            update_dict(d[k], v)  # type: ignore[arg-type]
        else:
            d[k] = v


def load_yaml(path: str) -> Any:
    """Load YAML from path if possible. Returns None if loader unavailable or on error."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is not available. Install pyyaml to use load_yaml.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, path: str) -> None:
    """Safely save data to a YAML file at path. Creates directories as needed."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is not available. Install pyyaml to use save_yaml.")
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def flatten_dict(d: Dict[Any, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary into a single-level dict with dotted keys."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
