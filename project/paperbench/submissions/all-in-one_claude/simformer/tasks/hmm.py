"""
HMM (Hidden Markov Model) task.

A task where the parameters form a Markov chain.

Parameters (latent states):
- θ_0 ~ N(0, 0.5²)
- θ_{i+1} ~ N(θ_i, 0.5²) for i = 0, ..., 9

Observations:
- x_i ~ N(θ_i², 0.5²) for i = 0, ..., 9

Both θ and x are 10-dimensional.
"""

import torch
import math
from simformer.tasks.base import BenchmarkTask


class HMMTask(BenchmarkTask):
    """
    HMM benchmark task with Markovian parameter structure.

    The nonlinear observation model (x_i ~ N(θ_i², 0.5²)) creates
    bimodal posteriors, making this a challenging inference task.
    """

    def __init__(
        self,
        dim: int = 10,
        init_std: float = 0.5,
        transition_std: float = 0.5,
        observation_std: float = 0.5,
        n_observations: int = 10,
    ):
        """
        Args:
            dim: Number of time steps (dimension of θ and x)
            init_std: Standard deviation of initial state
            transition_std: Standard deviation of transitions
            observation_std: Standard deviation of observations
            n_observations: Number of test observations
        """
        super().__init__(
            n_params=dim,
            n_data=dim,
            name="hmm",
            n_observations=n_observations,
        )

        self.dim = dim
        self.init_std = init_std
        self.transition_std = transition_std
        self.observation_std = observation_std

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the Markovian prior:
        θ_0 ~ N(0, init_std²)
        θ_{i+1} ~ N(θ_i, transition_std²)
        """
        theta = torch.zeros(n_samples, self.dim)

        # θ_0 ~ N(0, init_std²)
        theta[:, 0] = torch.randn(n_samples) * self.init_std

        # θ_{i+1} ~ N(θ_i, transition_std²)
        for i in range(self.dim - 1):
            theta[:, i + 1] = theta[:, i] + torch.randn(n_samples) * self.transition_std

        return theta

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate observations:
        x_i ~ N(θ_i², observation_std²)
        """
        noise = torch.randn_like(theta) * self.observation_std
        return theta ** 2 + noise

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(θ) for the Markovian prior.

        log p(θ) = log p(θ_0) + Σ log p(θ_{i+1}|θ_i)
        """
        log_prob = torch.zeros(theta.shape[0], device=theta.device)

        # log p(θ_0)
        log_prob += -0.5 * (theta[:, 0] / self.init_std) ** 2
        log_prob += -math.log(self.init_std) - 0.5 * math.log(2 * math.pi)

        # log p(θ_{i+1}|θ_i)
        for i in range(self.dim - 1):
            diff = theta[:, i + 1] - theta[:, i]
            log_prob += -0.5 * (diff / self.transition_std) ** 2
            log_prob += -math.log(self.transition_std) - 0.5 * math.log(2 * math.pi)

        return log_prob

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x|θ).
        """
        mean = theta ** 2
        diff = x - mean
        log_prob = -0.5 * (diff / self.observation_std) ** 2
        log_prob += -math.log(self.observation_std) - 0.5 * math.log(2 * math.pi)
        return log_prob.sum(dim=-1)

    def get_dependency_structure(self):
        """
        HMM structure:
        - Parameters form a chain: θ_i -> θ_{i+1}
        - Each observation depends only on its corresponding parameter
        """
        # Parameter structure (Markovian: each θ_i can see θ_{i-1} and itself)
        param_structure = torch.zeros(self.dim, self.dim)
        for i in range(self.dim):
            param_structure[i, i] = 1.0
            if i > 0:
                param_structure[i, i - 1] = 1.0

        # Data structure (observations are independent given theta)
        data_structure = torch.eye(self.dim)

        # Each observation depends only on its corresponding parameter
        param_to_data = torch.eye(self.dim)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }
