"""Adversarial Noise Inner-Maximisation (Eq. 6-7) for DPMs-ANT style training.

This module provides a lightweight, backbone-agnostic inner- maximisation routine to
identify the worst-case Gaussian perturbation epsilon* that degrades denoising during
adaptor training. The routine operates on a per-timestep x_t = sqrt(alpha_bar_t) x_0 +
sqrt(1 - alpha_bar_t) epsilon and optimises epsilon via gradient ascent on the negative
loss, followed by normalization to keep statistics stable.

Key equations (as referenced in the plan):
- Loss used inside the inner maximisation (Eq. 6–7):
  loss(epsilon) = || epsilon - epsilon_theta(x_t(epsilon), t) - sigma_hat^2 * gamma * grad_x_t log p_phi(y|x_t) ||^2
- Update step (inner loop):
  epsilon_{j+1} = Normalize( epsilon_j + omega * grad_epsilon ( - loss(epsilon_j) ) )
  where normalization enforces mean 0 and unit variance.

Public API
- inner_adversarial_noise(x0, t, eps_theta_fn, grad_logp_fn, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, gamma=5.0, J=10, omega=0.02, seed=None)
  Returns: epsilon_star (Tensor) of the same shape as x0.

Notes
- This module is intentionally lightweight and only implements the inner-max optimiser.
  It does not include dataset handling or outer-loop training logic.
- The function assumes a scalar timestep t for simplicity. If you require per-example timesteps,
  extend the indexing accordingly.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def _normalize_eps(eps: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to zero mean and unit variance (per-tensor)."""
    mean = eps.mean()
    var = eps.var(unbiased=False) + 1e-6
    std = torch.sqrt(var)
    return (eps - mean) / (std + 1e-12)


def inner_adversarial_noise(
    x0: torch.Tensor,
    t: int,
    eps_theta_fn: Callable[[torch.Tensor, int], torch.Tensor],
    grad_logp_fn: Callable[[torch.Tensor], torch.Tensor],
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    gamma: float = 5.0,
    J: int = 10,
    omega: float = 0.02,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Compute adversarial epsilon* for a given x0 and timestep t.

    Parameters:
    - x0: [B, C, ...] tensor representing the original target-domain sample (pre-noise).
    - t: scalar timestep index in [1, T]. Only scalar timesteps are supported in this minimal implementation.
    - eps_theta_fn: function (x_t, t) -> epsilon_theta(x_t, t)
    - grad_logp_fn: function (x_t) -> grad_x_t log p_phi(y|x_t) with same shape as x_t
    - sqrt_alphas_cumprod: 1D tensor of sqrt(alpha_bar_t) for all t in [0, T]
    - sqrt_one_minus_alphas_cumprod: 1D tensor of sqrt(1 - alpha_bar_t) for all t in [0, T]
    - gamma: similarity guidance weight (default 5.0)
    - J: number of inner-max steps
    - omega: inner-max step size
    - seed: optional random seed for reproducibility

    Returns:
    - epsilon_star: tensor with same shape as x0 representing the maximising perturbation.
    """
    if not isinstance(t, int):
        raise TypeError("Parameter t must be an integer timestep for this minimal implementation.")

    if seed is not None:
        torch.manual_seed(seed)

    device = x0.device
    dtype = x0.dtype
    B = x0.shape[0]

    # Initialize epsilon with standard normal noise matching x0's shape
    eps = torch.randn_like(x0, device=device, dtype=dtype)

    # Retrieve per-timestep scales
    if t < 0 or t >= sqrt_alphas_cumprod.numel():
        raise ValueError("t is out of bounds for the provided diffusion schedule.")
    alpha_bar_sqrt = sqrt_alphas_cumprod[t].to(device=device, dtype=dtype)
    one_minus_bar_sqrt = sqrt_one_minus_alphas_cumprod[t].to(device=device, dtype=dtype)
    # alpha_bar_t = (alpha_bar_sqrt)^2
    alpha_bar_t = (alpha_bar_sqrt * alpha_bar_sqrt).detach()
    sigma_hat_sq = max(0.0, 1.0 - float(alpha_bar_t))

    epsilon = eps.clone().detach()
    epsilon.requires_grad_(True)

    for _ in range(J):
        # Forward: compute x_t with current epsilon
        x_t = alpha_bar_sqrt * x0 + one_minus_bar_sqrt * epsilon
        # Model prediction and gradient signal
        eps_theta = eps_theta_fn(x_t, t)
        grad_logp = grad_logp_fn(x_t)

        # Loss term following Eq. (6)-(7)
        residual = epsilon - eps_theta - (sigma_hat_sq * gamma) * grad_logp
        loss = (residual * residual).mean()

        # Backprop to epsilon
        if epsilon.grad is not None:
            epsilon.grad.zero_()
        grad_eps = torch.autograd.grad(loss, epsilon, retain_graph=True, create_graph=False)[0]

        # Gradient ascent step on -loss (equivalently gradient descent on loss)
        with torch.no_grad():
            epsilon = (epsilon - omega * grad_eps).detach()
            epsilon = epsilon.requires_grad_(True)
            # Normalize to zero mean and unit variance (per-feature statistics of whole tensor)
            epsilon = _normalize_eps(epsilon)
            # Ensure detached copy for next iteration
            epsilon = epsilon.detach().clone().requires_grad_(True)

    epsilon_star = epsilon.detach()
    return epsilon_star


__all__ = ["inner_adversarial_noise"]
