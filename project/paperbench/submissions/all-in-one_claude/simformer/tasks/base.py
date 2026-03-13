"""
Base classes for simulation-based inference tasks.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
import torch
import torch.nn as nn


class Task(ABC):
    """
    Abstract base class for SBI tasks.

    A task defines:
    - A prior distribution p(θ)
    - A simulator that generates data x given parameters θ
    - Optional dependency structure for the Simformer
    """

    def __init__(
        self,
        n_params: int,
        n_data: int,
        name: str = "task",
    ):
        """
        Args:
            n_params: Number of parameter dimensions
            n_data: Number of data dimensions
            name: Name of the task
        """
        self.n_params = n_params
        self.n_data = n_data
        self.name = name

    @abstractmethod
    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the prior distribution p(θ).

        Args:
            n_samples: Number of samples

        Returns:
            Samples of shape (n_samples, n_params)
        """
        pass

    @abstractmethod
    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate data given parameters.

        Args:
            theta: Parameters of shape (n_samples, n_params)

        Returns:
            Data of shape (n_samples, n_data)
        """
        pass

    def sample_joint(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the joint distribution p(θ, x).

        Args:
            n_samples: Number of samples

        Returns:
            Tuple of (theta, x)
        """
        theta = self.sample_prior(n_samples)
        x = self.simulate(theta)
        return theta, x

    def get_dependency_structure(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get the dependency structure for the task.

        Returns:
            Dictionary with keys:
            - "param_structure": (n_params, n_params) adjacency matrix
            - "data_structure": (n_data, n_data) adjacency matrix
            - "param_to_data": (n_data, n_params) which params affect which data

            Returns None for dense (no special structure) tasks.
        """
        return None

    def get_reference_posterior(
        self,
        x_obs: torch.Tensor,
        n_samples: int = 10000,
    ) -> torch.Tensor:
        """
        Get reference posterior samples using MCMC or rejection sampling.

        This is used for evaluation. Subclasses should override if they
        have a tractable posterior or can run MCMC efficiently.

        Args:
            x_obs: Observed data
            n_samples: Number of posterior samples

        Returns:
            Posterior samples of shape (n_samples, n_params)
        """
        raise NotImplementedError(
            "Reference posterior not implemented for this task. "
            "Use MCMC or other methods externally."
        )

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log likelihood log p(x|θ).

        Only implemented for tasks with tractable likelihoods.

        Args:
            theta: Parameters
            x: Data

        Returns:
            Log likelihood values
        """
        raise NotImplementedError("Log likelihood not implemented for this task.")

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute log prior log p(θ).

        Args:
            theta: Parameters

        Returns:
            Log prior values
        """
        raise NotImplementedError("Log prior not implemented for this task.")


class SimulatorTask(Task):
    """
    Task with a callable simulator function.

    Useful for wrapping existing simulators.
    """

    def __init__(
        self,
        n_params: int,
        n_data: int,
        prior_fn: callable,
        simulator_fn: callable,
        name: str = "simulator_task",
    ):
        """
        Args:
            n_params: Number of parameter dimensions
            n_data: Number of data dimensions
            prior_fn: Function that takes n_samples and returns theta
            simulator_fn: Function that takes theta and returns x
            name: Name of the task
        """
        super().__init__(n_params, n_data, name)
        self.prior_fn = prior_fn
        self.simulator_fn = simulator_fn

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        return self.prior_fn(n_samples)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        return self.simulator_fn(theta)


class BenchmarkTask(Task):
    """
    Extended task class for benchmark tasks with evaluation utilities.
    """

    def __init__(
        self,
        n_params: int,
        n_data: int,
        name: str,
        n_observations: int = 10,
    ):
        """
        Args:
            n_params: Number of parameter dimensions
            n_data: Number of data dimensions
            name: Name of the task
            n_observations: Number of test observations to generate
        """
        super().__init__(n_params, n_data, name)
        self.n_observations = n_observations
        self._test_observations = None
        self._test_parameters = None

    def get_test_observations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get test observations and their true parameters.

        Returns:
            Tuple of (observations, true_parameters)
        """
        if self._test_observations is None:
            theta, x = self.sample_joint(self.n_observations)
            self._test_parameters = theta
            self._test_observations = x

        return self._test_observations, self._test_parameters

    def generate_training_data(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data.

        Args:
            n_samples: Number of training samples

        Returns:
            Tuple of (theta, x)
        """
        return self.sample_joint(n_samples)

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the task."""
        return {
            "name": self.name,
            "n_params": self.n_params,
            "n_data": self.n_data,
            "n_observations": self.n_observations,
        }


def create_task_with_structure(
    base_task: Task,
    param_structure: Optional[torch.Tensor] = None,
    data_structure: Optional[torch.Tensor] = None,
    param_to_data: Optional[torch.Tensor] = None,
) -> Task:
    """
    Create a task with custom dependency structure.

    Args:
        base_task: The base task
        param_structure: Parameter dependency matrix
        data_structure: Data dependency matrix
        param_to_data: Parameter to data dependency matrix

    Returns:
        Task with custom structure
    """

    class StructuredTask(type(base_task)):
        def __init__(self):
            super().__init__()

        def get_dependency_structure(self):
            return {
                "param_structure": param_structure,
                "data_structure": data_structure,
                "param_to_data": param_to_data,
            }

    structured = StructuredTask.__new__(StructuredTask)
    structured.__dict__.update(base_task.__dict__)
    return structured
