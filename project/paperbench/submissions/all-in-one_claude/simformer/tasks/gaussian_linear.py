"""
Gaussian Linear task.

A simple linear Gaussian model where both prior and likelihood are Gaussian.

Prior: θ ~ N(0, 0.1 * I)
Likelihood: x | θ ~ N(θ, 0.1 * I)

Both θ and x are 10-dimensional.
"""

import torch
from simformer.tasks.base import BenchmarkTask


class GaussianLinearTask(BenchmarkTask):
    """
    Gaussian Linear benchmark task.

    This is a fully factorized task where each parameter independently
    generates the corresponding data dimension.
    """

    def __init__(
        self,
        dim: int = 10,
        prior_std: float = 0.1,
        likelihood_std: float = 0.1,
        n_observations: int = 10,
    ):
        """
        Args:
            dim: Dimension of both θ and x
            prior_std: Standard deviation of the prior (actually variance in paper)
            likelihood_std: Standard deviation of the likelihood
            n_observations: Number of test observations
        """
        super().__init__(
            n_params=dim,
            n_data=dim,
            name="gaussian_linear",
            n_observations=n_observations,
        )

        self.dim = dim
        self.prior_std = prior_std
        self.likelihood_std = likelihood_std

        # Prior: N(0, prior_std * I)
        self.prior_mean = torch.zeros(dim)
        self.prior_cov = prior_std * torch.eye(dim)

        # Likelihood: N(θ, likelihood_std * I)
        self.likelihood_cov = likelihood_std * torch.eye(dim)

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """Sample from the prior p(θ) = N(0, prior_std * I)."""
        return torch.randn(n_samples, self.dim) * (self.prior_std ** 0.5)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulate x | θ ~ N(θ, likelihood_std * I)."""
        noise = torch.randn_like(theta) * (self.likelihood_std ** 0.5)
        return theta + noise

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x | θ)."""
        diff = x - theta
        log_prob = -0.5 * (diff ** 2).sum(dim=-1) / self.likelihood_std
        log_prob -= 0.5 * self.dim * (torch.log(torch.tensor(2 * 3.14159)) + torch.log(torch.tensor(self.likelihood_std)))
        return log_prob

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log p(θ)."""
        log_prob = -0.5 * (theta ** 2).sum(dim=-1) / self.prior_std
        log_prob -= 0.5 * self.dim * (torch.log(torch.tensor(2 * 3.14159)) + torch.log(torch.tensor(self.prior_std)))
        return log_prob

    def get_dependency_structure(self):
        """
        Fully factorized structure: each θ_i only affects x_i.
        """
        # Parameters are independent
        param_structure = torch.eye(self.dim)

        # Each data point is independent
        data_structure = torch.eye(self.dim)

        # Each parameter affects only the corresponding data
        param_to_data = torch.eye(self.dim)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }

    def get_reference_posterior(
        self,
        x_obs: torch.Tensor,
        n_samples: int = 10000,
    ) -> torch.Tensor:
        """
        Analytical posterior for Gaussian linear model.

        The posterior is:
        p(θ | x) = N(μ_post, Σ_post)

        where:
        Σ_post^{-1} = Σ_prior^{-1} + Σ_likelihood^{-1}
        μ_post = Σ_post * (Σ_prior^{-1} * μ_prior + Σ_likelihood^{-1} * x)

        For our case with μ_prior = 0:
        μ_post = Σ_post * Σ_likelihood^{-1} * x
        """
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)

        # Posterior precision (scalar since diagonal)
        precision_prior = 1.0 / self.prior_std
        precision_likelihood = 1.0 / self.likelihood_std
        precision_post = precision_prior + precision_likelihood

        # Posterior variance
        var_post = 1.0 / precision_post

        # Posterior mean
        mean_post = var_post * precision_likelihood * x_obs

        # Sample from posterior
        samples = mean_post + torch.randn(n_samples, self.dim) * (var_post ** 0.5)

        return samples
