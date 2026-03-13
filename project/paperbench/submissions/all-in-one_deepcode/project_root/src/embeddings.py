import numpy as np

"""
Embeddings module for joint θ (parameters) and x (observations) token construction.

This module provides a lightweight, self-contained Embeddings class used by the
tokenizer to assemble per-variable token embeddings. It is designed to be
deterministic given a seed and lightweight enough for quick experimentation
without requiring deep learning frameworks.

Public API:
- class Embeddings
  - __init__(d_model=64, max_vars=1024, max_funcs=256, seed=0)
  - get_id_embedding(idx)
  - get_mc_embedding(mc)
  - get_val_embedding(val)
  - get_func_embedding(func_idx)
  - get_time_embedding(t)
  - zero_embedding()

This module can be replaced with a more sophisticated learned embedding provider
without changing the surrounding code, keeping the FT (forward transform) pipelines
modular.
"""

from typing import Union

Number = Union[int, float, np.floating, np.integer]


class Embeddings:
    def __init__(self, d_model: int = 64, max_vars: int = 1024, max_funcs: int = 256, seed: int = 0):
        assert d_model > 0, "d_model must be positive"
        self.d_model = int(d_model)
        self.max_vars = int(max_vars)
        self.max_funcs = int(max_funcs)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)

        # Per-variable id embeddings (for θ_i or x_j identifiers)
        self.id_embeddings = rng.normal(loc=0.0, scale=0.1, size=(self.max_vars, self.d_model))
        # Per-variable function embeddings (for function-valued inputs)
        self.func_embeddings = rng.normal(loc=0.0, scale=0.1, size=(self.max_funcs, self.d_model))
        # Per-conditioning flag embeddings for MC (0/1)
        self.mc_embeddings = rng.normal(loc=0.0, scale=0.1, size=(2, self.d_model))  # index 0 -> MC=0, index 1 -> MC=1

        # A lightweight base embedding for values that are scalar-valued. We map
        # a scalar to a d_model-dimensional vector via sinusoidal Fourier-style features.
        # (No learnable weights needed for this lightweight fallback.)
        # Prepare a small frequency vector safely in case d_model is small.
        self._val_freqs = (np.arange(self.d_model, dtype=float) + 1.0) * 0.1

        # Function embedding projection for time conditioning of tokens.
        # Time embeddings are produced via Fourier features and projected to d_model.
        # We allocate a projection matrix for 2*L features, where L <= d_model//2.
        self.L = max(1, min(16, self.d_model // 2))
        self.time_proj = rng.normal(loc=0.0, scale=0.1, size=(2 * self.L, self.d_model))

    def get_id_embedding(self, idx: int) -> np.ndarray:
        i = int(idx) % self.max_vars
        return self.id_embeddings[i]

    def get_mc_embedding(self, mc: Union[int, float, bool]) -> np.ndarray:
        # MC expects 0 or 1
        mc_int = 1 if int(bool(mc)) else 0
        return self.mc_embeddings[mc_int]

    def get_val_embedding(self, val) -> np.ndarray:
        # Accept scalar or array-like; collapse to a scalar value
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                v = 0.0
            else:
                v = float(np.mean(np.asarray(val, dtype=float)))
        else:
            v = float(val)

        # Fourier-like features: sin and cos terms with exponentially spaced frequencies
        freqs = self._val_freqs
        # Ensure broadcasting works for scalar v
        vec = np.sin(v * freqs)
        # If vec shorter than d_model (shouldn't happen given construction), pad zeros
        if vec.shape[0] < self.d_model:
            pad = np.zeros(self.d_model - vec.shape[0], dtype=vec.dtype)
            vec = np.concatenate([vec, pad], axis=0)
        return vec[: self.d_model]

    def get_func_embedding(self, func_idx: int) -> np.ndarray:
        i = int(func_idx) % self.max_funcs
        return self.func_embeddings[i]

    def get_time_embedding(self, t) -> np.ndarray:
        t_val = float(t)
        # Fourier features: for k=0..L-1, include sin(2^k * pi * t) and cos(2^k * pi * t)
        feats = []
        for k in range(self.L):
            feats.append(np.sin((2 ** k) * np.pi * t_val))
            feats.append(np.cos((2 ** k) * np.pi * t_val))
        feats = np.array(feats, dtype=float)
        # Pad/truncate to length d_model
        if feats.size < self.d_model:
            pad = np.zeros(self.d_model - feats.size, dtype=float)
            feats = np.concatenate([feats, pad], axis=0)
        else:
            feats = feats[: self.d_model]
        # Project into d_model space (dense projection via learned-like weights)
        return feats @ self.time_proj  # shape (d_model,)

    def zero_embedding(self) -> np.ndarray:
        return np.zeros(self.d_model, dtype=float)


__all__ = ["Embeddings"]
