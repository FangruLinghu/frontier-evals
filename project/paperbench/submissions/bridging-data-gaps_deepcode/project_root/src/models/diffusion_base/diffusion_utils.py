import torch
from typing import Optional

"""
Diffusion utilities used by DDPM/LDM backbones.

This module provides small, backbone-agnostic helpers to perform common
diffusion-related tensor operations, such as sampling x_t from x_0 given a
timestep and optional noise, and reconstructing x_0 from x_t given a timestep
and the model's predicted epsilon.

Public helpers:
- q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
  Returns x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  where epsilon can be provided or sampled if None.

- predict_x0_from_xt(x_t, t, eps_theta, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
  Estimates x_0 given x_t, time t and predicted epsilon eps_theta(x_t, t).
"""


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor],
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Diffuse x_start by one step using timestep indices t.

    x_start: (B, C, ...) tensor representing x_0.
    t: (B,) tensor with integer timesteps in [0, T-1].
    noise: optional (B, C, ...) tensor; if None, a standard normal is sampled.
    sqrt_alphas_cumprod: 1D tensor of shape (T,), containing sqrt(alpha_bar_t).
    sqrt_one_minus_alphas_cumprod: 1D tensor of shape (T,), containing sqrt(1 - alpha_bar_t).

    Returns x_t with the same shape as x_start.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    B = x_start.size(0)
    # Build a (B, 1, 1, ..., 1) shaped view for broadcasting over x_start's shape
    view_shape = [B] + [1] * (x_start.dim() - 1)

    a = sqrt_alphas_cumprod.index_select(0, t).view(view_shape)
    b = sqrt_one_minus_alphas_cumprod.index_select(0, t).view(view_shape)

    return a * x_start + b * noise


def predict_x0_from_xt(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_theta: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Estimate x_0 from a diffusion step using predicted epsilon.

    x_t: (B, C, ...) noisy latent at time t.
    t: (B,) timestep indices.
    eps_theta: model's prediction of epsilon at (x_t, t).
    sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod: schedules as in q_sample.

    Returns x_0 estimate with same shape as x_t.
    """
    B = x_t.size(0)
    view_shape = [B] + [1] * (x_t.dim() - 1)

    a = sqrt_alphas_cumprod.index_select(0, t).view(view_shape)
    b = sqrt_one_minus_alphas_cumprod.index_select(0, t).view(view_shape)

    # x_t = a * x_0 + b * eps => x_0 = (x_t - b * eps) / a
    return (x_t - b * eps_theta) / (a + 1e-8)


__all__ = ["q_sample", "predict_x0_from_xt"]
