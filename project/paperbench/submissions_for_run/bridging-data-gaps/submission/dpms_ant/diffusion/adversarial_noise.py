"""
Adversarial Noise Selection for DPMs-ANT.

From Section 4.2 of the paper:
The adversarial noise selection finds the "worse-case" Gaussian noise
that the pre-trained model fails to denoise on the target dataset.

Equation (7):
    ε_{j+1} = Norm(ε_j + ω * ∇_{ε_j} ||ε_j - ε_θ(√ᾱ_t * x0 + √(1-ᾱ_t) * ε_j, t)||²)

where:
    - j ∈ {0, 1, ..., J-1}, J=10 steps
    - ω = 0.02 is the "learning rate" for noise ascent
    - Norm(·) normalizes to approximate mean=0, std=I
    - ε_0 ~ N(0, I)

The inner maximization finds the worst-case noise, then the outer
minimization trains the model on this noise.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from dpms_ant.diffusion.gaussian_diffusion import GaussianDiffusion


def normalize_noise(noise: torch.Tensor) -> torch.Tensor:
    """
    Normalize noise to have approximately mean=0 and std=I per sample.

    This is the Norm(·) function from Equation (7).

    Args:
        noise: Noise tensor of shape (B, C, H, W)

    Returns:
        Normalized noise with mean≈0, std≈1 per sample
    """
    # Normalize per sample
    b = noise.shape[0]
    noise_flat = noise.reshape(b, -1)

    # Subtract mean
    noise_flat = noise_flat - noise_flat.mean(dim=1, keepdim=True)

    # Scale to unit variance
    std = noise_flat.std(dim=1, keepdim=True)
    noise_flat = noise_flat / (std + 1e-8)

    return noise_flat.reshape(noise.shape)


def adversarial_noise_selection(
    model: nn.Module,
    x_start: torch.Tensor,
    t: torch.Tensor,
    diffusion: GaussianDiffusion,
    initial_noise: Optional[torch.Tensor] = None,
    J: int = 10,
    omega: float = 0.02,
) -> torch.Tensor:
    """
    Adversarial noise selection (Equation 7).

    Finds the "worse-case" Gaussian noise by gradient ascent on the
    denoising loss w.r.t. the noise.

    Args:
        model: Pre-trained denoising model (frozen)
        x_start: Clean target images x0, shape (B, C, H, W)
        t: Timesteps, shape (B,)
        diffusion: GaussianDiffusion instance
        initial_noise: Optional initial noise ε_0 (default: sample from N(0,I))
        J: Number of adversarial steps (default: 10)
        omega: Step size for noise update (default: 0.02)

    Returns:
        Adversarial noise ε* of shape (B, C, H, W)
    """
    if initial_noise is None:
        noise = torch.randn_like(x_start)
    else:
        noise = initial_noise.clone()

    # Extract diffusion coefficients
    sqrt_alphas_cumprod = diffusion._extract(
        diffusion.sqrt_alphas_cumprod, t, x_start.shape
    )
    sqrt_one_minus_alphas_cumprod = diffusion._extract(
        diffusion.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    model.eval()

    for j in range(J):
        noise = noise.detach().requires_grad_(True)

        # Compute noisy image: xt = √ᾱ_t * x0 + √(1-ᾱ_t) * ε
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

        # Model prediction
        model_output = model(x_t, t)

        # Handle learned variance models (split output)
        if model_output.shape[1] == x_start.shape[1] * 2:
            eps_pred, _ = torch.split(model_output, x_start.shape[1], dim=1)
        else:
            eps_pred = model_output

        # Compute denoising loss: ||ε - ε_θ(xt, t)||²
        loss = ((noise - eps_pred) ** 2).sum()

        # Gradient ascent on noise
        grad = torch.autograd.grad(loss, noise)[0]

        # Update noise: ε_{j+1} = Norm(ε_j + ω * ∇_{ε_j} L)
        noise = noise.detach() + omega * grad.detach()

        # Normalize to maintain Gaussian properties
        noise = normalize_noise(noise)

    return noise.detach()


def adversarial_noise_with_loss(
    model: nn.Module,
    x_start: torch.Tensor,
    t: torch.Tensor,
    diffusion: GaussianDiffusion,
    classifier: Optional[nn.Module] = None,
    gamma: float = 5.0,
    J: int = 10,
    omega: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute adversarial noise and the full ANT loss.

    This combines adversarial noise selection with similarity-guided training.

    Full loss (Equation 8):
        L(ψ) = E_{t,x0} [||ε* - ε_{θ,ψ}(x*_t, t) - σ̂²_t * γ * ∇_{x*_t} log pϕ(y=T|x*_t)||²]

    Args:
        model: Denoising model (UNet or UNetWithAdaptor)
        x_start: Clean target images x0
        t: Timesteps
        diffusion: GaussianDiffusion instance
        classifier: Optional binary classifier for similarity guidance
        gamma: Similarity guidance strength (default: 5)
        J: Adversarial noise steps
        omega: Adversarial noise step size

    Returns:
        Tuple of (adversarial_noise, noisy_image, loss)
    """
    # Step 1: Find adversarial noise using the frozen base model
    # Note: For adversarial noise, we use the base model without adaptor
    adv_noise = adversarial_noise_selection(
        model, x_start, t, diffusion,
        J=J, omega=omega,
    )

    # Step 2: Compute noisy image with adversarial noise
    x_t = diffusion.q_sample(x_start, t, noise=adv_noise)

    # Step 3: Model prediction (with adaptor if present)
    model_output = model(x_t, t)

    if model_output.shape[1] == x_start.shape[1] * 2:
        eps_pred, _ = torch.split(model_output, x_start.shape[1], dim=1)
    else:
        eps_pred = model_output

    # Step 4: Compute target (with or without similarity guidance)
    target = adv_noise.clone()

    if classifier is not None:
        # Compute similarity guidance: σ̂²_t * γ * ∇_xt log pϕ(y=T|xt)
        sigma_hat = diffusion._extract(diffusion.sigma_hat, t, x_start.shape)

        classifier_grad = classifier.get_target_gradient(x_t.detach(), t, target_label=1)
        guidance = sigma_hat ** 2 * gamma * classifier_grad

        target = target - guidance.detach()

    # Step 5: Compute loss
    loss = ((target - eps_pred) ** 2).mean()

    return adv_noise, x_t, loss
