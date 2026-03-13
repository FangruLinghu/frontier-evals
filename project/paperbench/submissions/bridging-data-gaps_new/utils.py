## utils.py
"""
Utilities for reproducibility, logging, and checkpoint management in DPMs-ANT experiments.

This module provides:
- set_seed(seed): Set random seeds across Python, NumPy, and PyTorch to ensure reproducible experiments.
- log_metrics(metrics, log_file=None): Flatten and log training/evaluation metrics with a timestamp; optionally persist to a file.
- save_checkpoint(path, state, metadata=None): Atomically save adaptor and related states to disk with optional metadata.
- load_checkpoint(path): Load a previously saved checkpoint (state and metadata).

Notes:
- The adaptor parameters (ψ) and other components are expected to be provided as PyTorch state_dicts
  or plain Python objects inside the 'state' dictionary. This module does not know model internals;
  it simply serializes/deserializes the given dictionaries.
- Tensors are moved to CPU before serialization to improve portability across devices.
"""

from __future__ import annotations

import os
import random
import time
import datetime
import json
from typing import Any, Dict, Optional, Callable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set the random seed across Python, NumPy, and PyTorch to ensure reproducible runs.

    This function also configures PyTorch/cuDNN behavior to be deterministic when possible.

    Args:
        seed (int): The seed to use for all RNGs.
    """
    # Python hash seed (for hash-based operations)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python random, NumPy, PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may impact performance)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch versions may not support this function
        pass

    # CuDNN deterministic mode
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _flatten_metrics(
    data: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary of metrics.

    Args:
        data (Dict[str, Any]): Nested dictionary of metrics.
        parent_key (str): Accumulated key path.
        sep (str): Separator used between hierarchical keys.

    Returns:
        Dict[str, Any]: Flattened dictionary with keys like "train/loss".
    """
    items: Dict[str, Any] = {}
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_metrics(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def log_metrics(metrics: Dict[str, Any], log_file: Optional[str] = None) -> None:
    """
    Log metrics with a timestamp. Supports nested dictionaries by flattening keys.

    The function prints a single-line summary for quick glance and writes to a log file
    if a path is provided.

    Args:
        metrics (Dict[str, Any]): Metrics to log. May be nested.
        log_file (Optional[str]): Path to a file to append logs. If None, logs only go to stdout.
    """
    if metrics is None:
        return

    flat = _flatten_metrics(metrics)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = []
    for k, v in flat.items():
        if isinstance(v, (int, float)):
            # Use a compact representation with 6 significant digits
            try:
                parts.append(f"{k}={float(v):.6g}")
            except Exception:
                parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={repr(v)}")

    line = f"[{ts}] " + ", ".join(parts)
    print(line)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            # Do not crash training due to logging failures
            print(f"[utils.log_metrics] Warning: could not write log to {log_file}: {e}")


def _to_cpu(obj: Any) -> Any:
    """
    Recursively move tensors to CPU for safe serialization.

    Args:
        obj (Any): Object to convert.

    Returns:
        Any: CPU-bound copy suitable for serialization.
    """
    if isinstance(obj, torch.Tensor):
        try:
            return obj.detach().cpu()
        except Exception:
            return obj
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_cpu(v) for v in obj]
        return type(obj)(converted)
    return obj


def save_checkpoint(path: str, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Atomically save a checkpoint dictionary to disk.

    The checkpoint contains:
      - "state": a dictionary of model/optimizer state_dicts or other serializable artifacts
      - "metadata": optional auxiliary information for reproducibility (seeds, config hash, etc.)

    The function moves all tensors to CPU before serialization for portability.

    Args:
        path (str): Destination path for the checkpoint file.
        state (Dict[str, Any]): State dictionary to save (e.g., adaptor ψ, optimizer states, etc.).
        metadata (Optional[Dict[str, Any]]): Optional metadata to accompany the state.
    """
    # Prepare payload
    payload: Dict[str, Any] = {
        "state": _to_cpu(state),
        "metadata": metadata or {},
    }

    # Ensure directory exists
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    tmp_path = path + ".tmp"

    # Serialize to a temporary file first for atomicity
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise RuntimeError(f"Failed to save checkpoint to {path}: {e}") from e


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a previously saved checkpoint.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        Dict[str, Any]: The loaded checkpoint dictionary with keys:
            - "state": dict of state_dicts (on CPU)
            - "metadata": associated metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    try:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload is not a dictionary as expected.")
        return payload
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}") from e


def get_worker_seed_fn(seed: int) -> Callable[[int], None]:
    """
    Helper to generate a worker initialization function for DataLoader to ensure
    each worker has a unique seed derived from a base seed.

    Args:
        seed (int): Base seed.

    Returns:
        Callable[[int], None]: A function suitable as worker_init_fn for DataLoader.
    """
    def _init(worker_id: int) -> None:
        # Each worker gets a unique seed
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        # If PyTorch is using multiprocessing, ensure CUDA seeds are covered as well
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

    return _init