"""
Stochastic Differential Equations for score-based diffusion models.

Implements VESDE (Variance Exploding) and VPSDE (Variance Preserving) as
described in Song et al. (2021) and used in the Simformer paper.

The forward SDE is:
    dx_t = f(x_t, t)dt + g(t)dw

The reverse SDE is:
    dx_t = [f(x_t, t) - g(t)^2 * s(x_t, t)]dt + g(t)dw̃

where s(x_t, t) = ∇_{x_t} log p_t(x_t) is the score function.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


class SDE(ABC):
    """
    Abstract base class for Stochastic Differential Equations.

    Defines the interface for SDEs used in score-based diffusion models.
    """

    def __init__(self, t_min: float = 1e-5, t_max: float = 1.0):
        """
        Args:
            t_min: Minimum time (avoid numerical issues at t=0)
            t_max: Maximum time
        """
        self.t_min = t_min
        self.t_max = t_max

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute drift coefficient f(x, t)."""
        pass

    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Compute diffusion coefficient g(t)."""
        pass

    @abstractmethod
    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and std of marginal distribution p(x_t | x_0).

        Returns:
            mean: μ_t(x_0)
            std: σ_t
        """
        pass

    @abstractmethod
    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from the prior distribution p_T(x_T)."""
        pass

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Uniformly sample time from [t_min, t_max]."""
        return torch.rand(batch_size, device=device) * (self.t_max - self.t_min) + self.t_min

    def perturb(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perturb clean data x_0 to noisy data x_t.

        Args:
            x_0: Clean data
            t: Time values

        Returns:
            x_t: Noisy data
            noise: The added noise (for score computation)
        """
        mean, std = self.marginal_prob(x_0, t)
        noise = torch.randn_like(x_0)
        x_t = mean + std * noise
        return x_t, noise

    def score_from_noise(
        self, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the true conditional score from noise.

        The conditional score is:
            ∇_{x_t} log p(x_t | x_0) = -noise / σ_t

        Args:
            noise: The noise that was added
            t: Time values

        Returns:
            The conditional score
        """
        _, std = self.marginal_prob(torch.zeros_like(noise), t)
        return -noise / (std + 1e-8)


class VESDE(SDE):
    """
    Variance Exploding SDE.

    Defined as:
        f(x, t) = 0
        g(t) = σ_min * (σ_max / σ_min)^t * sqrt(2 * log(σ_max / σ_min))

    Marginal distribution:
        p(x_t | x_0) = N(x_t; x_0, σ_t^2 I)
        where σ_t = σ_min * (σ_max / σ_min)^t

    Paper settings: σ_max = 15, σ_min = 0.0001

    Args:
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        t_min: Minimum time
        t_max: Maximum time
    """

    def __init__(
        self,
        sigma_min: float = 0.0001,
        sigma_max: float = 15.0,
        t_min: float = 1e-5,
        t_max: float = 1.0,
    ):
        super().__init__(t_min, t_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = math.log(sigma_max / sigma_min)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """VESDE has zero drift."""
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Compute diffusion coefficient g(t)."""
        sigma_t = self.sigma(t)
        return sigma_t * math.sqrt(2 * self.log_ratio)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute σ_t = σ_min * (σ_max / σ_min)^t."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Marginal distribution: N(x_0, σ_t^2 I)
        """
        # Expand t for broadcasting
        while t.dim() < x_0.dim():
            t = t.unsqueeze(-1)

        mean = x_0
        std = self.sigma(t)
        return mean, std

    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from p_T ≈ N(0, σ_max^2 I)."""
        return torch.randn(shape, device=device) * self.sigma_max

    def prior_mean_std(self) -> Tuple[float, float]:
        """Get mean and std of prior distribution."""
        return 0.0, self.sigma_max


class VPSDE(SDE):
    """
    Variance Preserving SDE (also known as DDPM).

    Defined as:
        f(x, t) = -0.5 * β(t) * x
        g(t) = sqrt(β(t))

    where β(t) = β_min + t * (β_max - β_min)

    Marginal distribution:
        p(x_t | x_0) = N(x_t; α_t * x_0, (1 - α_t^2) I)
        where α_t = exp(-0.5 * ∫_0^t β(s) ds)

    Paper settings: β_min = 0.01, β_max = 10

    Args:
        beta_min: Minimum beta value
        beta_max: Maximum beta value
        t_min: Minimum time
        t_max: Maximum time
    """

    def __init__(
        self,
        beta_min: float = 0.01,
        beta_max: float = 10.0,
        t_min: float = 1e-5,
        t_max: float = 1.0,
    ):
        super().__init__(t_min, t_max)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute β(t) = β_min + t * (β_max - β_min)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute drift coefficient f(x, t) = -0.5 * β(t) * x."""
        beta_t = self.beta(t)
        while beta_t.dim() < x.dim():
            beta_t = beta_t.unsqueeze(-1)
        return -0.5 * beta_t * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Compute diffusion coefficient g(t) = sqrt(β(t))."""
        return torch.sqrt(self.beta(t))

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute ∫_0^t β(s) ds = β_min * t + 0.5 * (β_max - β_min) * t^2."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute α_t = exp(-0.5 * ∫_0^t β(s) ds)."""
        return torch.exp(-0.5 * self.integral_beta(t))

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Marginal distribution: N(α_t * x_0, (1 - α_t^2) I)
        """
        # Expand t for broadcasting
        while t.dim() < x_0.dim():
            t = t.unsqueeze(-1)

        alpha_t = self.alpha(t)
        mean = alpha_t * x_0
        std = torch.sqrt(1 - alpha_t ** 2 + 1e-8)
        return mean, std

    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from p_T ≈ N(0, I) (approximately)."""
        return torch.randn(shape, device=device)

    def prior_mean_std(self) -> Tuple[float, float]:
        """Get mean and std of prior distribution."""
        return 0.0, 1.0


def get_sde(sde_type: str = "vesde", **kwargs) -> SDE:
    """
    Factory function to create SDE instances.

    Args:
        sde_type: "vesde" or "vpsde"
        **kwargs: Arguments to pass to the SDE constructor

    Returns:
        SDE instance
    """
    sde_type = sde_type.lower()

    if sde_type == "vesde":
        return VESDE(**kwargs)
    elif sde_type == "vpsde":
        return VPSDE(**kwargs)
    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")
