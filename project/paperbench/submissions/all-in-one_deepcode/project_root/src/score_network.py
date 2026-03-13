import numpy as np
from typing import Callable

"""
Minimal, deterministic score network head.

This module provides a lightweight, non-learned (randomly initialized but deterministic)
score network compatible with the rest of the repository. It is designed for unit tests
and small-scale demonstrations where a full training loop is not required.

Interface:
- ScoreNetwork: a simple two-layer network that outputs a score vector of length dim_x
  given an input x (length dim_x) and time t. Time is encoded with a small Fourier-like embedding
  (sin/cos) and concatenated to the input before the first linear layer.
- build_score_network(dim_x, hidden_dim=128, seed=0) -> Callable[[np.ndarray, float], np.ndarray]
  Returns a callable score_fn(x, t) that uses an internal ScoreNetwork instance.
"""

__all__ = ["ScoreNetwork", "build_score_network"]

class ScoreNetwork:
    def __init__(self, dim_x: int, hidden_dim: int = 128, seed: int = 0):
        """Initialize a tiny deterministic score network.
        - dim_x: dimensionality of the observation vector x
        - hidden_dim: hidden layer size
        - seed: random seed for deterministic initialization
        The network maps (x, t) -> s(x, t) with s having the same dimension as x.
        Time embedding adds 2 features, which are concatenated to x before the first layer.
        """
        self.dim_x = int(dim_x)
        self.hidden_dim = int(hidden_dim)
        self.total_input_dim = self.dim_x + 2  # x concatenated with [sin(pi t), cos(pi t)]
        rng = np.random.default_rng(seed)
        # Weights for a two-layer network:
        # z = [x, sin(pi t), cos(pi t)], shape = (total_input_dim,)
        self.W1 = rng.normal(loc=0.0, scale=0.02, size=(self.hidden_dim, self.total_input_dim))
        self.b1 = rng.normal(loc=0.0, scale=0.02, size=(self.hidden_dim,))
        # Map hidden -> output (dim_x)
        self.W2 = rng.normal(loc=0.0, scale=0.02, size=(self.dim_x, self.hidden_dim))
        self.b2 = rng.normal(loc=0.0, scale=0.02, size=(self.dim_x,))

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute s_phi(x, t) of shape (dim_x,)."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim_x:
            raise ValueError(f"Input x must have length {self.dim_x}, got {x.shape[0]}")
        # time embedding using simple sin/cos pair of the normalized time in [0,1]
        # Here t is assumed in [0,1]. If outside, a simple modulo is applied to keep values reasonable.
        tt = float(t)
        # Normalize t into [0,1] via clipping as a safeguard
        tt = max(0.0, min(1.0, tt))
        t_embed = np.array([np.sin(np.pi * tt), np.cos(np.pi * tt)], dtype=np.float32)
        z = np.concatenate([x, t_embed], axis=0)  # shape (total_input_dim,)
        h = self._relu(self.W1 @ z + self.b1)
        out = self.W2 @ h + self.b2  # shape (dim_x,)
        return out

def build_score_network(dim_x: int, hidden_dim: int = 128, seed: int = 0) -> Callable[[np.ndarray, float], np.ndarray]:
    """Factory that builds a tiny score function s_phi(x, t).

    Returns a callable score_fn(x, t) that can be passed to samplers.
    """
    sn = ScoreNetwork(dim_x, hidden_dim=hidden_dim, seed=seed)
    def score_fn(x: np.ndarray, t: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        return sn(x_arr.reshape(-1), float(t))
    return score_fn
