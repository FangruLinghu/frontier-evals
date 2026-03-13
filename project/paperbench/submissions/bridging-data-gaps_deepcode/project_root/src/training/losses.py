# Core loss utilities for DPMs-ANT training (L_sim and L_AN terms)
# These utilities are designed to be lightweight and dependency-light while
# providing the essential equation-driven losses used by the training loop.

from typing import Callable, Optional
import torch

__all__ = ["L_sim", "L_AN"]


def L_sim(
    eps_t: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    eps_theta_fn: Callable[[torch.Tensor, int], torch.Tensor],
    grad_logp_fn: Callable[[torch.Tensor], torch.Tensor],
    gamma: float,
    sigma_hat_sq: torch.Tensor,
) -> torch.Tensor:
    """Similarity-guided training loss (Eq. 5 analogue).

    Computes the mean-squared error between the actual noise epsilon (eps_t)
    and the model-predicted noise eps_theta(x_t, t) with a gradient-based
    domain signal injected: eps_t - eps_theta - sigma_hat_sq * gamma * grad_logp
    where grad_logp is ∇_{x_t} log p_φ(y | x_t).

    Args:
        eps_t: Diffusion noise sample at time t, shape [B, ...].
        x_t: Diffusion state at time t, shape [B, ...]. (Required for eps_theta and grad)
        t: Timestep index (0-based).
        eps_theta_fn: Function that predicts ε_θ(x_t, t).
        grad_logp_fn: Function that returns ∇_{x_t} log p_φ(y | x_t).
        gamma: Similarity weight scalar.
        sigma_hat_sq: Scalar or tensor representing σ̂_t^2, broadcastable to eps_t.

    Returns:
        Scalar tensor representing the mean-squared similarity loss.
    """
    eps_theta = eps_theta_fn(x_t, t)
    grad_logp = grad_logp_fn(x_t)

    # Ensure shapes are broadcastable; rely on PyTorch broadcasting.
    residual = eps_t - eps_theta - (sigma_hat_sq * gamma * grad_logp)
    return residual.pow(2).mean()


def L_AN(
    eps_star: torch.Tensor,
    x0: torch.Tensor,
    t: int,
    eps_theta_fn: Callable[[torch.Tensor, int], torch.Tensor],
    grad_logp_fn: Callable[[torch.Tensor], torch.Tensor],
    gamma: float,
    sigma_hat_sq: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Adversarial-noise objective (Eq. 6/8 analogue).

    This computes the outer AN loss using the adversarial noise ε⋆ by forming
    x_t(ε⋆) = sqrt(ᾱ_t) x0 + sqrt(1 - ᾱ_t) ε⋆, then evaluating the same
    residual as L_sim but with that perturbed x_t.

    Args:
        eps_star: Adversarial noise candidate ε⋆ with shape compatible to x0.
        x0: Original clean sample in the target domain, shape [B, C, ...].
        t: Timestep index (0-based).
        eps_theta_fn: Function that predicts ε_θ(x_t, t).
        grad_logp_fn: Function that returns ∇_{x_t} log p_φ(y | x_t).
        gamma: Similarity weight scalar.
        sigma_hat_sq: σ̂_t^2 term (possibly scalar or tensor).
        alphas_cumprod: Precomputed ᾱ_t cumulative product tensor of length T.
        sqrt_alphas_cumprod: Precomputed sqrt(ᾱ_t) tensor of length T.
        sqrt_one_minus_alphas_cumprod: Precomputed sqrt(1 - ᾱ_t) tensor of length T.

    Returns:
        Scalar tensor representing the AN loss for the adversarial perturbation.
    """
    # Build x_t using ε⋆ based on provided diffusion schedule components.
    # ᾱ_t is the cumulative product up to t inclusive; handle indexing safety.
    alpha_bar = sqrt_alphas_cumprod[t]
    one_minus = sqrt_one_minus_alphas_cumprod[t]
    x_t_eps = alpha_bar * x0 + one_minus * eps_star

    eps_theta = eps_theta_fn(x_t_eps, t)
    grad_logp = grad_logp_fn(x_t_eps)

    residual = eps_star - eps_theta - (sigma_hat_sq * gamma * grad_logp)
    return residual.pow(2).mean()
