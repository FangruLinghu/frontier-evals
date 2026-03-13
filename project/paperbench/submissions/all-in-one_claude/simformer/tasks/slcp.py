"""
SLCP (Simple Likelihood Complex Posterior) task.

A challenging task designed to produce complex posteriors from simple likelihoods.

Prior: θ ~ U(-3, 3)^5
Likelihood: x_i ~ N(μ_θ, Σ_θ) for i = 1, ..., 4

where:
- μ_θ = [θ_1, θ_2]
- Σ_θ = [[θ_3², tanh(θ_5) * θ_3² * θ_4²], [tanh(θ_5) * θ_3² * θ_4², θ_4²]]

θ is 5-dimensional, x is 8-dimensional (4 i.i.d. 2D observations).
"""

import torch
from simformer.tasks.base import BenchmarkTask


class SLCPTask(BenchmarkTask):
    """
    SLCP (Simple Likelihood Complex Posterior) benchmark task.

    Features 4 i.i.d. observations, which is useful for testing
    the Simformer's ability to handle repeated observations.
    """

    def __init__(
        self,
        prior_low: float = -3.0,
        prior_high: float = 3.0,
        n_iid: int = 4,
        n_observations: int = 10,
    ):
        """
        Args:
            prior_low: Lower bound of uniform prior
            prior_high: Upper bound of uniform prior
            n_iid: Number of i.i.d. observations
            n_observations: Number of test observations
        """
        super().__init__(
            n_params=5,
            n_data=n_iid * 2,  # Each observation is 2D
            name="slcp",
            n_observations=n_observations,
        )

        self.prior_low = prior_low
        self.prior_high = prior_high
        self.n_iid = n_iid

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """Sample from the uniform prior p(θ) = U(-3, 3)^5."""
        return (
            torch.rand(n_samples, 5) * (self.prior_high - self.prior_low)
            + self.prior_low
        )

    def _get_mean_cov(self, theta: torch.Tensor):
        """
        Compute mean and covariance from parameters.

        μ_θ = [θ_1, θ_2]
        Σ_θ = [[θ_3², tanh(θ_5) * θ_3² * θ_4²],
               [tanh(θ_5) * θ_3² * θ_4², θ_4²]]
        """
        n_samples = theta.shape[0]

        # Mean
        mu = theta[:, :2]  # (n_samples, 2)

        # Covariance components
        theta_3_sq = theta[:, 2] ** 2
        theta_4_sq = theta[:, 3] ** 2
        tanh_theta_5 = torch.tanh(theta[:, 4])
        off_diag = tanh_theta_5 * theta_3_sq * theta_4_sq

        # Build covariance matrix
        cov = torch.zeros(n_samples, 2, 2, device=theta.device)
        cov[:, 0, 0] = theta_3_sq
        cov[:, 0, 1] = off_diag
        cov[:, 1, 0] = off_diag
        cov[:, 1, 1] = theta_4_sq

        return mu, cov

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate x | θ where x consists of n_iid i.i.d. samples from N(μ_θ, Σ_θ).
        """
        n_samples = theta.shape[0]

        # Get mean and covariance
        mu, cov = self._get_mean_cov(theta)

        # Add small diagonal for numerical stability
        cov = cov + 1e-6 * torch.eye(2, device=theta.device).unsqueeze(0)

        # Cholesky decomposition for sampling
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError:
            # Fallback if Cholesky fails
            L = torch.eye(2, device=theta.device).unsqueeze(0).expand(n_samples, -1, -1)

        # Generate all i.i.d. samples
        x_list = []
        for _ in range(self.n_iid):
            z = torch.randn(n_samples, 2, device=theta.device)
            x_i = mu + torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
            x_list.append(x_i)

        # Concatenate all observations
        x = torch.cat(x_list, dim=-1)  # (n_samples, 2 * n_iid)

        return x

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x | θ) for the SLCP likelihood.

        Since observations are i.i.d., the total log likelihood is the sum.
        """
        n_samples = theta.shape[0]

        # Get mean and covariance
        mu, cov = self._get_mean_cov(theta)

        # Add small diagonal for numerical stability
        cov = cov + 1e-6 * torch.eye(2, device=theta.device).unsqueeze(0)

        # Compute log likelihood for each i.i.d. observation
        total_log_prob = torch.zeros(n_samples, device=theta.device)

        for i in range(self.n_iid):
            x_i = x[:, 2*i:2*(i+1)]  # (n_samples, 2)
            diff = x_i - mu  # (n_samples, 2)

            # Log determinant
            try:
                log_det = torch.logdet(cov)
            except RuntimeError:
                log_det = torch.zeros(n_samples, device=theta.device)

            # Mahalanobis distance
            cov_inv = torch.linalg.inv(cov)
            mahal = torch.bmm(diff.unsqueeze(1), torch.bmm(cov_inv, diff.unsqueeze(-1)))
            mahal = mahal.squeeze(-1).squeeze(-1)

            # Log probability
            log_prob = -0.5 * (2 * torch.log(torch.tensor(2 * 3.14159)) + log_det + mahal)
            total_log_prob = total_log_prob + log_prob

        return total_log_prob

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log p(θ) for uniform prior."""
        in_bounds = (
            (theta >= self.prior_low) & (theta <= self.prior_high)
        ).all(dim=-1)

        log_prob = torch.where(
            in_bounds,
            torch.tensor(-5 * torch.log(torch.tensor(self.prior_high - self.prior_low))),
            torch.tensor(float("-inf")),
        )

        return log_prob

    def get_dependency_structure(self):
        """
        SLCP has i.i.d. observations, so the data structure is block-diagonal.
        All parameters affect all data.
        """
        # Parameters can be correlated in the posterior
        param_structure = torch.ones(5, 5)

        # Data observations are i.i.d. (2D blocks)
        data_structure = torch.zeros(self.n_data, self.n_data)
        for i in range(self.n_iid):
            start = i * 2
            end = (i + 1) * 2
            data_structure[start:end, start:end] = 1.0

        # All parameters affect all data
        param_to_data = torch.ones(self.n_data, 5)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }
