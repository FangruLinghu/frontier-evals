import numpy as np
from typing import Optional

"""
Utility for ME (Masking/Dependency) based attention masks used in Transformer models.

This module provides a lightweight, deterministic way to construct a [sequence_length x sequence_length]
boolean mask that can be consumed by attention mechanisms. It supports two primary modes:
- directed: a simple hierarchical dependency where θ tokens attend only to θ, while x tokens may attend to all tokens.
- undirected: block cross-group attention; tokens attend only within their own group (θ-θ and x-x).

The mask is designed to be used in downstream attention implementations where True indicates attention is allowed
(from source token j to target token i, i.e., mask[i, j] == True means attention can be computed).
"""

__all__ = ["build_me_mask", "adapt_me_mask", "ME_Mask"]


def build_me_mask(n_theta: int, n_x: int, me_type: str = "directed") -> np.ndarray:
    """Build an ME (masking/attention dependency) mask for a given number of theta and x tokens.

    Args:
        n_theta: number of theta tokens (parameters)
        n_x: number of x tokens (observations)
        me_type: type of dependency structure: 'directed', 'undirected', or 'full'
                 - 'directed': theta attends only to theta; x attends to all tokens
                 - 'undirected': attention is allowed within groups only (theta<->theta, x<->x)
                 - 'full': full attention (no ME constraints)

    Returns:
        A boolean numpy array of shape (n_theta + n_x, n_theta + n_x) where True means attention is allowed.
    """
    total = int(n_theta + n_x)
    mask = np.zeros((total, total), dtype=bool)

    if me_type == "directed":
        # θ can attend only to θ
        if n_theta > 0:
            mask[:n_theta, :n_theta] = True
        # x can attend to all tokens (θ and x)
        if n_x > 0:
            mask[n_theta:, :] = True
    elif me_type == "undirected":
        # Within-group attention only
        if n_theta > 0:
            mask[:n_theta, :n_theta] = True
        if n_x > 0:
            mask[n_theta:, n_theta:] = True
        # Cross-group is disallowed
    else:
        # Full attention (no ME constraints)
        mask[:] = True

    return mask


def adapt_me_mask(existing_mask: Optional[np.ndarray], new_type: str = "directed", n_theta: Optional[int] = None, n_x: Optional[int] = None) -> np.ndarray:
    """Adapt an existing ME mask to a new type, optionally with updated group sizes.

    This is a lightweight helper to simulate dynamic adaptation when conditioning changes (e.g., switching
    from posterior to likelihood). If the existing_mask is provided, this function will attempt to rebuild
    a new mask with the updated type and return it. If sizes are not provided, it will fall back to the
    existing mask shape to infer n_theta and n_x.

    Args:
        existing_mask: optional existing mask to derive sizes from
        new_type: new masking type ('directed', 'undirected', 'full')
        n_theta: updated number of θ tokens; if None, inferred from existing_mask
        n_x: updated number of x tokens; if None, inferred from existing_mask

    Returns:
        New mask as a boolean numpy array.
    """
    if existing_mask is not None:
        total = existing_mask.shape[0]
        if n_theta is None:
            # Heuristic: split in the middle
            n_theta = total // 2
        if n_x is None:
            n_x = total - n_theta
        return build_me_mask(n_theta, n_x, me_type=new_type)

    # If no existing mask, just build from provided sizes
    if n_theta is None or n_x is None:
        raise ValueError("n_theta and n_x must be provided if existing_mask is not given.")
    return build_me_mask(n_theta, n_x, me_type=new_type)


class ME_Mask:
    """A small helper encapsulating an ME mask configuration for a transformer.

    Usage:
        me = ME_Mask(n_theta=K, n_x=M, me_type='directed')
        mask = me.build_mask()
        me.adapt('undirected')  # change type
        new_mask = me.build_mask()
    """

    def __init__(self, n_theta: int, n_x: int, me_type: str = "directed"):
        self.n_theta = int(n_theta)
        self.n_x = int(n_x)
        self.me_type = str(me_type)
        self._mask_cache: Optional[np.ndarray] = None
        self._update_cache()

    def _update_cache(self):
        self._mask_cache = build_me_mask(self.n_theta, self.n_x, self.me_type)

    def adapt(self, new_type: Optional[str] = None, n_theta: Optional[int] = None, n_x: Optional[int] = None):
        if new_type is not None:
            self.me_type = str(new_type)
        if n_theta is not None:
            self.n_theta = int(n_theta)
        if n_x is not None:
            self.n_x = int(n_x)
        self._update_cache()

    def build_mask(self) -> np.ndarray:
        return self._mask_cache
