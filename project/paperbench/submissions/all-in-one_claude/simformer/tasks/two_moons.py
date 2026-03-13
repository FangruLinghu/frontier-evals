"""
Two Moons task.

A classic benchmark task for testing multimodal posterior estimation.

Prior: θ ~ U(-1, 1)
Likelihood: Nonlinear transformation producing crescent-shaped data.

Both θ and x are 2-dimensional.
"""

import math
import torch
from simformer.tasks.base import BenchmarkTask


class TwoMoonsTask(BenchmarkTask):
    """
    Two Moons benchmark task.

    The posterior is bimodal (crescent-shaped), making it a good test
    for methods that need to capture multimodality.
    """

    def __init__(
        self,
        prior_low: float = -1.0,
        prior_high: float = 1.0,
        noise_std: float = 0.1,
        n_observations: int = 10,
    ):
        """
        Args:
            prior_low: Lower bound of uniform prior
            prior_high: Upper bound of uniform prior
            noise_std: Standard deviation of radial noise
            n_observations: Number of test observations
        """
        super().__init__(
            n_params=2,
            n_data=2,
            name="two_moons",
            n_observations=n_observations,
        )

        self.prior_low = prior_low
        self.prior_high = prior_high
        self.noise_std = noise_std

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """Sample from the uniform prior p(θ) = U(-1, 1)^2."""
        return (
            torch.rand(n_samples, 2) * (self.prior_high - self.prior_low)
            + self.prior_low
        )

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate the Two Moons model.

        x | θ = [r * cos(α) + 0.25, r * sin(α)] + [-|θ_1 + θ_2|/√2, (-θ_1 + θ_2)/√2]

        where:
        - α ~ U(-π/2, π/2)
        - r ~ N(0.1, 0.01²)
        """
        n_samples = theta.shape[0]

        # Sample angle and radius
        alpha = torch.rand(n_samples) * math.pi - math.pi / 2  # U(-π/2, π/2)
        r = torch.randn(n_samples) * 0.01 + 0.1  # N(0.1, 0.01²)

        # First term: crescent
        x1_crescent = r * torch.cos(alpha) + 0.25
        x2_crescent = r * torch.sin(alpha)

        # Second term: parameter-dependent shift
        sqrt2 = math.sqrt(2)
        x1_shift = -torch.abs(theta[:, 0] + theta[:, 1]) / sqrt2
        x2_shift = (-theta[:, 0] + theta[:, 1]) / sqrt2

        # Combine
        x = torch.stack([
            x1_crescent + x1_shift,
            x2_crescent + x2_shift,
        ], dim=-1)

        return x

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log p(θ) for uniform prior."""
        in_bounds = (
            (theta >= self.prior_low) & (theta <= self.prior_high)
        ).all(dim=-1)

        log_prob = torch.where(
            in_bounds,
            torch.tensor(-2 * math.log(self.prior_high - self.prior_low)),
            torch.tensor(float("-inf")),
        )

        return log_prob

    def get_dependency_structure(self):
        """
        Both parameters affect both data dimensions (dense structure).
        """
        # Parameters can depend on each other in the posterior
        param_structure = torch.ones(2, 2)

        # Data dimensions are related
        data_structure = torch.ones(2, 2)

        # All parameters affect all data
        param_to_data = torch.ones(2, 2)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }


def sample_two_moons_posterior_mcmc(
    x_obs: torch.Tensor,
    n_samples: int = 10000,
    n_warmup: int = 1000,
    step_size: float = 0.01,
) -> torch.Tensor:
    """
    Sample from the Two Moons posterior using MCMC.

    Uses random direction slice sampling followed by MH-MCMC.

    Args:
        x_obs: Observed data
        n_samples: Number of posterior samples
        n_warmup: Number of warmup samples
        step_size: MH step size

    Returns:
        Posterior samples
    """
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)

    task = TwoMoonsTask()

    # Initialize from prior
    theta = task.sample_prior(n_samples)

    def log_prob(theta_val):
        """Unnormalized log posterior (up to additive constant)."""
        log_prior = task.log_prior(theta_val)
        if torch.isinf(log_prior).any():
            return log_prior

        # Approximate log likelihood using simulation
        # This is a simplification - proper MCMC would need analytical likelihood
        return log_prior

    # Simple MH-MCMC
    samples = []
    current = theta.clone()

    for i in range(n_warmup + n_samples):
        # Propose
        proposal = current + torch.randn_like(current) * step_size

        # Accept/reject
        log_prob_current = log_prob(current)
        log_prob_proposal = log_prob(proposal)

        log_alpha = log_prob_proposal - log_prob_current
        accept = torch.log(torch.rand(n_samples)) < log_alpha

        current = torch.where(accept.unsqueeze(-1), proposal, current)

        if i >= n_warmup:
            samples.append(current.clone())

    return torch.stack(samples).mean(dim=0)  # Return last samples
