"""
Utilities for reproducibility, IO, and simple config helpers.
"""

import os
import json
import pickle
import random
from typing import Any, Dict

# Optional dependencies (best-effort)
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

__all__ = [
    "seed_everything",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "flatten_dict",
    "get_numpy_rng",
]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators for reproducibility.

    This function attempts to seed multiple RNG backends if they are available in the
    environment. It is a lightweight utility suitable for unit tests and small experiments.
    """
    random.seed(seed)
    if _HAS_NUMPY:
        np.random.seed(seed)
    if _HAS_TORCH:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            # If PyTorch is present but cannot seed for some reason, ignore gracefully
            pass


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists.

    Creates intermediate directories as needed and does nothing if the path already exists.
    """
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    """Serialize a Python object to a JSON file with indentation for readability."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str) -> None:
    """Serialize an object using Python's pickle to a binary file."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load a pickled object from a binary file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_dict(d: Dict[Any, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary into a single-level dictionary with dotted keys.

    Example:
      {'a': {'b': 1, 'c': 2}, 'd': 3} -> {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_numpy_rng(seed: int) -> "np.random.Generator":  # type: ignore[name-defined]
    """Return a NumPy Generator seeded with the given seed.

    This is a small helper to obtain a modern, stateless RNG where possible. Falls back to
    a legacy RandomState if necessary (for very old NumPy versions).
    """
    if not _HAS_NUMPY:
        raise RuntimeError("NumPy is not available in this environment.")
    try:
        return np.random.default_rng(seed)
    except AttributeError:
        # Fallback for older NumPy versions
        rng = np.random.RandomState(seed)  # type: ignore
        class _RNGWrapper:
            def normal(self, loc=0.0, scale=1.0, size=None):  # type: ignore
                return rng.normal(loc=loc, scale=scale, size=size)

            def integers(self, low, high=None, size=None, dtype=int):  # type: ignore
                return rng.randint(low, high, size=size, dtype=dtype)

        return _RNGWrapper()  # type: ignore

