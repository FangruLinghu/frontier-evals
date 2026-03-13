"""
Tree task.

A nonlinear tree-shaped task with hierarchical dependencies.

Parameters:
- θ_0 ~ N(0, 1)
- θ_1 ~ N(θ_0, 1)  (Note: paper says N(1, 1) for θ_1 but that seems like a typo)
- θ_2 ~ N(θ_0, 1)

Data:
- x_0 ~ N(sin(θ_1)², 0.2²)
- x_1 ~ N(0.1 * θ_1², 0.2²)
- x_2 ~ N(0.1 * θ_2², 0.6²)
- x_3 ~ N(cos(θ_2)², 0.1²)

Both θ and x are 3-dimensional and 4-dimensional respectively.
"""

import torch
import math
from simformer.tasks.base import BenchmarkTask


class TreeTask(BenchmarkTask):
    """
    Tree benchmark task with hierarchical dependencies.

    The tree structure creates multimodal conditionals, making it
    a good test for arbitrary conditional sampling.
    """

    def __init__(
        self,
        n_observations: int = 10,
    ):
        """
        Args:
            n_observations: Number of test observations
        """
        super().__init__(
            n_params=3,
            n_data=4,
            name="tree",
            n_observations=n_observations,
        )

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the hierarchical prior:
        θ_0 ~ N(0, 1)
        θ_1 ~ N(θ_0, 1)
        θ_2 ~ N(θ_0, 1)
        """
        theta = torch.zeros(n_samples, 3)

        # θ_0 ~ N(0, 1)
        theta[:, 0] = torch.randn(n_samples)

        # θ_1 ~ N(θ_0, 1)
        theta[:, 1] = theta[:, 0] + torch.randn(n_samples)

        # θ_2 ~ N(θ_0, 1)
        theta[:, 2] = theta[:, 0] + torch.randn(n_samples)

        return theta

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate data from the tree model:
        x_0 ~ N(sin(θ_1)², 0.2²)
        x_1 ~ N(0.1 * θ_1², 0.2²)
        x_2 ~ N(0.1 * θ_2², 0.6²)
        x_3 ~ N(cos(θ_2)², 0.1²)
        """
        n_samples = theta.shape[0]
        x = torch.zeros(n_samples, 4)

        theta_0, theta_1, theta_2 = theta[:, 0], theta[:, 1], theta[:, 2]

        # x_0 ~ N(sin(θ_1)², 0.2²)
        x[:, 0] = torch.sin(theta_1) ** 2 + torch.randn(n_samples) * 0.2

        # x_1 ~ N(0.1 * θ_1², 0.2²)
        x[:, 1] = 0.1 * theta_1 ** 2 + torch.randn(n_samples) * 0.2

        # x_2 ~ N(0.1 * θ_2², 0.6²)
        x[:, 2] = 0.1 * theta_2 ** 2 + torch.randn(n_samples) * 0.6

        # x_3 ~ N(cos(θ_2)², 0.1²)
        x[:, 3] = torch.cos(theta_2) ** 2 + torch.randn(n_samples) * 0.1

        return x

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(θ) for the hierarchical prior.

        log p(θ) = log p(θ_0) + log p(θ_1|θ_0) + log p(θ_2|θ_0)
        """
        theta_0, theta_1, theta_2 = theta[:, 0], theta[:, 1], theta[:, 2]

        # log p(θ_0) = log N(0, 1)
        log_p_0 = -0.5 * theta_0 ** 2 - 0.5 * math.log(2 * math.pi)

        # log p(θ_1|θ_0) = log N(θ_0, 1)
        log_p_1 = -0.5 * (theta_1 - theta_0) ** 2 - 0.5 * math.log(2 * math.pi)

        # log p(θ_2|θ_0) = log N(θ_0, 1)
        log_p_2 = -0.5 * (theta_2 - theta_0) ** 2 - 0.5 * math.log(2 * math.pi)

        return log_p_0 + log_p_1 + log_p_2

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x|θ).
        """
        theta_1, theta_2 = theta[:, 1], theta[:, 2]
        x_0, x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # log p(x_0|θ_1) with std=0.2
        mean_0 = torch.sin(theta_1) ** 2
        log_p_0 = -0.5 * ((x_0 - mean_0) / 0.2) ** 2 - math.log(0.2) - 0.5 * math.log(2 * math.pi)

        # log p(x_1|θ_1) with std=0.2
        mean_1 = 0.1 * theta_1 ** 2
        log_p_1 = -0.5 * ((x_1 - mean_1) / 0.2) ** 2 - math.log(0.2) - 0.5 * math.log(2 * math.pi)

        # log p(x_2|θ_2) with std=0.6
        mean_2 = 0.1 * theta_2 ** 2
        log_p_2 = -0.5 * ((x_2 - mean_2) / 0.6) ** 2 - math.log(0.6) - 0.5 * math.log(2 * math.pi)

        # log p(x_3|θ_2) with std=0.1
        mean_3 = torch.cos(theta_2) ** 2
        log_p_3 = -0.5 * ((x_3 - mean_3) / 0.1) ** 2 - math.log(0.1) - 0.5 * math.log(2 * math.pi)

        return log_p_0 + log_p_1 + log_p_2 + log_p_3

    def get_dependency_structure(self):
        """
        Tree structure:
        - θ_0 is the root
        - θ_1 and θ_2 depend on θ_0
        - x_0, x_1 depend on θ_1
        - x_2, x_3 depend on θ_2
        """
        # Parameter structure (directed: θ_1, θ_2 <- θ_0)
        param_structure = torch.tensor([
            [1, 1, 1],  # θ_0 can see all
            [0, 1, 0],  # θ_1 can only see itself
            [0, 0, 1],  # θ_2 can only see itself
        ], dtype=torch.float32)

        # Data structure (each x_i is independent given theta)
        data_structure = torch.eye(4)

        # Which parameters affect which data
        param_to_data = torch.tensor([
            [0, 0, 0],  # x_0 depends on θ_1
            [0, 1, 0],  # x_1 depends on θ_1
            [0, 0, 1],  # x_2 depends on θ_2
            [0, 0, 1],  # x_3 depends on θ_2
        ], dtype=torch.float32)

        # Actually x_0 and x_1 depend on θ_1, x_2 and x_3 depend on θ_2
        param_to_data = torch.tensor([
            [0, 1, 0],  # x_0 depends on θ_1
            [0, 1, 0],  # x_1 depends on θ_1
            [0, 0, 1],  # x_2 depends on θ_2
            [0, 0, 1],  # x_3 depends on θ_2
        ], dtype=torch.float32)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }
