"""Adaptor bottleneck configuration module.

This module centralizes per-backbone adaptor bottleneck specifications
for Houlsby-style adaptors used within diffusion backbones.

Backbones supported:
- ddpm: c=4, d=8
- ldm:  c=2, d=8

Public interface:
- bottleneck_configs: dict mapping backbone name -> {"c": int, "d": int}
- get_bottleneck_config(backbone_name: str) -> tuple[int, int]
"""
from typing import Dict, Tuple

# Per-backbone adaptor bottleneck configuration
# c: bottleneck multiplier for the channel compression path
# d: internal hidden dimension multiplier
bottleneck_configs: Dict[str, Dict[str, int]] = {
    "ddpm": {"c": 4, "d": 8},  # DDPM (pixel-space) adaptor settings
    "ldm": {"c": 2, "d": 8},   # Latent Diffusion Model adaptor settings
}


def get_bottleneck_config(backbone_name: str) -> Tuple[int, int]:
    """Return the (c, d) bottleneck configuration for a given backbone.

    Parameters:
        backbone_name: Name of the diffusion backbone (e.g., "ddpm", "ldm").

    Returns:
        A tuple (c, d) corresponding to the adaptor bottleneck configuration.

    Raises:
        ValueError if the backbone is unknown.
    """
    if backend := bottleneck_configs.get(backbone_name.lower()):
        return int(backend["c"]), int(backend["d"])
    raise ValueError(
        f"Unknown backbone '{backbone_name}'. Available backbones: {list(bottleneck_configs.keys())}"
    )


__all__ = ["bottleneck_configs", "get_bottleneck_config"]
