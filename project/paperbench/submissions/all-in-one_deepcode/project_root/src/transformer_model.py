import numpy as np
from typing import Optional, List, Dict, Any

"""
A lightweight Transformer backbone with optional ME-enabled attention support.
This is a minimal, self-contained implementation intended for unit tests and
simple demonstrations. It does not rely on heavy DL frameworks and uses NumPy
only.

Public API:
- TransformerModel(d_model=64, n_layers=6, n_heads=4, d_ff=150, seed=0, me_mask=None)
- encode(tokens: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray
- set_lengths(n_theta: Optional[int], n_x: Optional[int]) -> None
- __init__ exposes internal weights for inspection if needed (not required for tests).
"""

class TransformerModel:
    def __init__(self,
                 d_model: int = 64,
                 n_layers: int = 6,
                 n_heads: int = 4,
                 d_ff: int = 150,
                 seed: int = 0,
                 me_mask: Optional[Any] = None):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.seed = int(seed)
        self.me_mask = me_mask  # Optional mask provider (not strictly required)
        self._rng = np.random.default_rng(self.seed)
        self._head_dim = self.d_model // self.n_heads

        # Initialize per-layer weights (deterministic for tests)
        self.layers: List[Dict[str, np.ndarray]] = []
        for _ in range(self.n_layers):
            layer = {
                'W_q': self._rng.normal(scale=0.02, size=(self.d_model, self.d_model)),
                'W_k': self._rng.normal(scale=0.02, size=(self.d_model, self.d_model)),
                'W_v': self._rng.normal(scale=0.02, size=(self.d_model, self.d_model)),
                'W_o': self._rng.normal(scale=0.02, size=(self.d_model, self.d_model)),
                'W_ff1': self._rng.normal(scale=0.02, size=(self.d_model, self.d_ff)),
                'b_ff1': self._rng.normal(scale=0.0, size=(self.d_ff,)),
                'W_ff2': self._rng.normal(scale=0.02, size=(self.d_ff, self.d_model)),
                'b_ff2': self._rng.normal(scale=0.0, size=(self.d_model,)),
            }
            self.layers.append(layer)

        # Optional: a simple layer-norm statistic (not learned)
        self._eps = 1e-5

        # Lengths for potential ME masking integration (optional)
        self.n_theta: Optional[int] = None
        self.n_x: Optional[int] = None

    def set_lengths(self, n_theta: Optional[int], n_x: Optional[int]) -> None:
        """Optionally set the counts for θ and x to enable ME masking generation.
        This information can be used by a mask provider to build a dependency mask
        for attention, if such a provider is wired up.
        """
        self.n_theta = int(n_theta) if n_theta is not None else None
        self.n_x = int(n_x) if n_x is not None else None

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        s = np.sum(e, axis=axis, keepdims=True) + 1e-12
        return e / s

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + self._eps
        return (x - mean) / std

    def encode(self, tokens: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Run a forward pass through the lightweight Transformer encoder.
        tokens: shape (seq_len, d_model)
        mask: optional attention mask shape (seq_len, seq_len) with True=attend
        Returns: tensor of shape (seq_len, d_model)
        """
        x = tokens.copy()
        seq_len = x.shape[0]
        assert x.shape[1] == self.d_model, "Input tokens must have dimension d_model"

        # If no mask provided, attempt to generate from ME mask if available; otherwise full mask
        if mask is None:
            if self.me_mask is not None and hasattr(self.me_mask, 'build_mask'):
                try:
                    generated = self.me_mask.build_mask()
                    if isinstance(generated, np.ndarray) and generated.shape[0] == seq_len and generated.shape[1] == seq_len:
                        mask = generated
                    else:
                        mask = np.ones((seq_len, seq_len), dtype=bool)
                except Exception:
                    mask = np.ones((seq_len, seq_len), dtype=bool)
            else:
                mask = np.ones((seq_len, seq_len), dtype=bool)

        # Process through layers
        for layer in self.layers:
            W_q = layer['W_q']; W_k = layer['W_k']; W_v = layer['W_v']; W_o = layer['W_o']
            W_ff1 = layer['W_ff1']; b_ff1 = layer['b_ff1']
            W_ff2 = layer['W_ff2']; b_ff2 = layer['b_ff2']

            Q = x @ W_q  # (seq_len, d_model)
            K = x @ W_k
            V = x @ W_v

            head_dim = self._head_dim
            # reshape: (seq_len, n_heads, head_dim)
            Qh = Q.reshape(seq_len, self.n_heads, head_dim)
            Kh = K.reshape(seq_len, self.n_heads, head_dim)
            Vh = V.reshape(seq_len, self.n_heads, head_dim)

            # transpose to (n_heads, seq_len, head_dim)
            Qh = Qh.transpose(1, 0, 2)
            Kh = Kh.transpose(1, 0, 2)
            Vh = Vh.transpose(1, 0, 2)

            # attention scores: (n_heads, seq_len, seq_len)
            dk = float(head_dim)
            scores = np.matmul(Qh, Kh.transpose(0, 2, 1)) / np.sqrt(dk)

            # apply mask (same mask for all heads)
            if mask is not None:
                scores = np.where(mask[np.newaxis, :, :], scores, -1e9)

            weights = self._softmax(scores, axis=-1)

            attn = np.matmul(weights, Vh)  # (n_heads, seq_len, head_dim)
            attn = attn.transpose(1, 0, 2).reshape(seq_len, self.d_model)

            x_attn = attn @ W_o  # (seq_len, d_model)

            # residual and simple normalization
            x = x + x_attn
            x = self._normalize(x)

            # feed-forward
            ff = (x @ W_ff1) + b_ff1
            ff = np.maximum(ff, 0)  # ReLU
            ff = (ff @ W_ff2) + b_ff2
            x = x + ff
            x = self._normalize(x)

        return x

# Public export name for consumers
__all__ = ["TransformerModel"]
