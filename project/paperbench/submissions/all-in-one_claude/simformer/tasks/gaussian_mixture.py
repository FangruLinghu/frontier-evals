"""
Gaussian Mixture task.

A task where the data is generated from a mixture of two Gaussians
with different covariances but the same mean θ.

Prior: θ ~ U(-10, 10)
Likelihood: x | θ ~ 0.5 * N(θ, I) + 0.5 * N(θ, 0.01 * I)

Both θ and x are 2-dimensional.
"""

import torch
from simformer.tasks.base import BenchmarkTask


class GaussianMixtureTask(BenchmarkTask):
    """
    Gaussian Mixture benchmark task.

    The likelihood is a mixture of two Gaussians with different scales,
    making the posterior multimodal in terms of uncertainty.
    """

    def __init__(
        self,
        dim: int = 2,
        prior_low: float = -10.0,
        prior_high: float = 10.0,
        cov_large: float = 1.0,
        cov_small: float = 0.01,
        mixture_weight: float = 0.5,
        n_observations: int = 10,
    ):
        """
        Args:
            dim: Dimension of both θ and x
            prior_low: Lower bound of uniform prior
            prior_high: Upper bound of uniform prior
            cov_large: Variance of the first component
            cov_small: Variance of the second component
            mixture_weight: Weight of the first component
            n_observations: Number of test observations
        """
        super().__init__(
            n_params=dim,
            n_data=dim,
            name="gaussian_mixture",
            n_observations=n_observations,
        )

        self.dim = dim
        self.prior_low = prior_low
        self.prior_high = prior_high
        self.cov_large = cov_large
        self.cov_small = cov_small
        self.mixture_weight = mixture_weight

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """Sample from the uniform prior p(θ) = U(prior_low, prior_high)."""
        return (
            torch.rand(n_samples, self.dim) * (self.prior_high - self.prior_low)
            + self.prior_low
        )

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate x | θ ~ 0.5 * N(θ, cov_large * I) + 0.5 * N(θ, cov_small * I).
        """
        n_samples = theta.shape[0]

        # Decide which component for each sample
        component = torch.rand(n_samples) < self.mixture_weight

        # Generate noise
        noise_large = torch.randn_like(theta) * (self.cov_large ** 0.5)
        noise_small = torch.randn_like(theta) * (self.cov_small ** 0.5)

        # Combine based on component
        noise = torch.where(
            component.unsqueeze(-1).expand_as(theta),
            noise_large,
            noise_small,
        )

        return theta + noise

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x | θ) for the Gaussian mixture likelihood.
        """
        diff = x - theta

        # Log probability for each component
        log_prob_large = -0.5 * (diff ** 2).sum(dim=-1) / self.cov_large
        log_prob_large -= 0.5 * self.dim * (
            torch.log(torch.tensor(2 * 3.14159)) + torch.log(torch.tensor(self.cov_large))
        )

        log_prob_small = -0.5 * (diff ** 2).sum(dim=-1) / self.cov_small
        log_prob_small -= 0.5 * self.dim * (
            torch.log(torch.tensor(2 * 3.14159)) + torch.log(torch.tensor(self.cov_small))
        )

        # Log-sum-exp for mixture
        log_mixture_weight = torch.log(torch.tensor(self.mixture_weight))
        log_one_minus_weight = torch.log(torch.tensor(1 - self.mixture_weight))

        log_prob = torch.logsumexp(
            torch.stack([
                log_mixture_weight + log_prob_large,
                log_one_minus_weight + log_prob_small,
            ], dim=0),
            dim=0,
        )

        return log_prob

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log p(θ) for uniform prior."""
        # Check if theta is in bounds
        in_bounds = (
            (theta >= self.prior_low) & (theta <= self.prior_high)
        ).all(dim=-1)

        log_prob = torch.where(
            in_bounds,
            torch.tensor(-self.dim * torch.log(torch.tensor(self.prior_high - self.prior_low))),
            torch.tensor(float("-inf")),
        )

        return log_prob

    def get_dependency_structure(self):
        """
        Parameters are independent, data points are independent given theta.
        """
        # Parameters are independent
        param_structure = torch.eye(self.dim)

        # Data points are independent given theta
        data_structure = torch.eye(self.dim)

        # All parameters affect all data (dense)
        param_to_data = torch.ones(self.dim, self.dim)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }
