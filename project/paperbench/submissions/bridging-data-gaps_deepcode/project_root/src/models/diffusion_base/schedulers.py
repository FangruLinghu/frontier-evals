import numpy as np
import torch
from typing import Dict, Tuple, Optional

"""
Diffusion schedules utility module.

Provides simple beta schedules and precomputed auxiliary terms required by DDPM/LDM forward passes:
- linear_beta_schedule: simple linear progression of betas
- cosine_beta_schedule: cosine-based schedule (as popular in DDPM v2 literature)

Each schedule returns a numpy-backed dictionary with keys:
- betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

Helper get_beta_schedule(name, T, **kwargs) returns one of the above in numpy form.
Also provided: to_torch_schedule(schedule_np, device=None, dtype=torch.float32) to convert to torch tensors.
"""


def _as_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def linear_beta_schedule(T: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> Dict[str, np.ndarray]:
    """Create a linear beta schedule with T timesteps.

    Returns a dict with precomputed terms needed for denoising:
      - betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    """
    if T <= 0:
        raise ValueError("T must be a positive integer for beta schedule generation.")
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    betas = _as_float(betas)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


def cosine_beta_schedule(T: int, s: float = 0.008) -> Dict[str, np.ndarray]:
    """Cosine-based beta schedule (per DDPM v2 style).

    This computes alphas_cumprod via a cosine schedule and derives betas from consecutive ratios.
    Returns the same dictionary keys as linear schedule.
    """
    if T <= 0:
        raise ValueError("T must be a positive integer for beta schedule generation.")

    # t goes from 0 to 1 inclusive, with T steps
    t = np.linspace(0, 1, T + 1, dtype=np.float32)
    # See DDPM v2 cosine schedule formulation
    cos_term = np.cos((t + s) / (1.0 + s) * (np.pi / 2.0))
    alphas_cumprod = (cos_term ** 2).astype(np.float32)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # Derive betas from consecutive alphas_cumprod
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0.0, 0.999)

    # Align to length T
    betas = betas.astype(np.float32)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0, dtype=np.float32)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


def get_beta_schedule(name: str, T: int, **kwargs) -> Dict[str, np.ndarray]:
    """Factory to obtain a beta schedule by name.

    name: 'linear' (or 'ddpm') or 'cosine' (or 'cos').
    Returns a dict with numpy arrays as produced by the respective schedule.
    """
    key = (name or "").lower()
    if key in ("linear", "lin", "ddpm"):
        return linear_beta_schedule(T, kwargs.get("beta_start", 0.0001), kwargs.get("beta_end", 0.02))
    if key in ("cosine", "cos"):
        return cosine_beta_schedule(T, kwargs.get("s", 0.008))
    raise ValueError(f"Unknown beta schedule: {name}")


def _to_tensor_schedule(schedule_np: Dict[str, np.ndarray], device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in schedule_np.items():
        out[k] = torch.as_tensor(v, dtype=dtype, device=device)
    return out


def to_torch_schedule(schedule_np: Dict[str, np.ndarray], device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """Public helper to convert numpy schedule(s) to PyTorch tensors on a given device."""
    return _to_tensor_schedule(schedule_np, device, dtype)


# Public aliases for convenience
__all__ = [
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "get_beta_schedule",
    "to_torch_schedule",
]
