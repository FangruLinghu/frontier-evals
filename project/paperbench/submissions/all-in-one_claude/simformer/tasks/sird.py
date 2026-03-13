"""
SIRD (Susceptible-Infected-Recovered-Deceased) task.

An epidemiological model with time-dependent contact rate.

The SIRD model is defined by:
dS/dt = -β(t) * S * I
dI/dt = β(t) * S * I - γ * I - μ * I
dR/dt = γ * I
dD/dt = μ * I

Where:
- β(t) is a time-dependent contact rate (function-valued parameter)
- γ is the recovery rate (global parameter)
- μ is the death rate (global parameter)

This task demonstrates inference with ∞-dimensional (function-valued) parameters.
"""

import torch
import math
from typing import Optional, Tuple, List
from simformer.tasks.base import BenchmarkTask


def rbf_kernel(t1: torch.Tensor, t2: torch.Tensor, variance: float = 2.5**2, length_scale: float = 7.0) -> torch.Tensor:
    """
    RBF (Radial Basis Function) kernel for Gaussian Process.

    k(t1, t2) = variance * exp(-0.5 * ||t1 - t2||^2 / length_scale^2)
    """
    diff = t1.unsqueeze(-1) - t2.unsqueeze(-2)  # (n1, n2)
    return variance * torch.exp(-0.5 * diff ** 2 / length_scale ** 2)


def sample_gp_prior(times: torch.Tensor, n_samples: int, kernel_fn=rbf_kernel) -> torch.Tensor:
    """
    Sample from a Gaussian Process prior.

    Args:
        times: Time points of shape (n_times,)
        n_samples: Number of samples
        kernel_fn: Kernel function

    Returns:
        Samples of shape (n_samples, n_times)
    """
    n_times = len(times)

    # Compute kernel matrix
    K = kernel_fn(times, times)

    # Add small diagonal for numerical stability
    K = K + 1e-6 * torch.eye(n_times)

    # Cholesky decomposition
    L = torch.linalg.cholesky(K)

    # Sample
    z = torch.randn(n_samples, n_times)
    return torch.mm(z, L.T)


def solve_sird(
    beta_values: torch.Tensor,
    beta_times: torch.Tensor,
    gamma: torch.Tensor,
    mu: torch.Tensor,
    t_eval: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    dt: float = 0.1,
) -> torch.Tensor:
    """
    Solve SIRD ODE.

    Args:
        beta_values: Contact rate values at beta_times, shape (batch_size, n_beta_times)
        beta_times: Times for beta values
        gamma: Recovery rate, shape (batch_size,)
        mu: Death rate, shape (batch_size,)
        t_eval: Times at which to evaluate
        initial_state: Initial [S, I, R, D] state
        dt: Time step

    Returns:
        Solution of shape (batch_size, len(t_eval), 4)
    """
    batch_size = beta_values.shape[0]
    device = beta_values.device

    if initial_state is None:
        # Default: mostly susceptible, small infected
        initial_state = torch.tensor([[0.99, 0.01, 0.0, 0.0]], device=device).expand(batch_size, -1)

    def get_beta(t):
        """Interpolate beta at time t."""
        # Simple linear interpolation
        idx = torch.searchsorted(beta_times, t)
        idx = torch.clamp(idx, 1, len(beta_times) - 1)

        t0 = beta_times[idx - 1]
        t1 = beta_times[idx]
        w = (t - t0) / (t1 - t0 + 1e-8)
        w = torch.clamp(w, 0, 1)

        beta_0 = beta_values[:, idx - 1]
        beta_1 = beta_values[:, idx]

        return (1 - w) * beta_0 + w * beta_1

    def sird_derivatives(state, t):
        """Compute SIRD derivatives."""
        S, I, R, D = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        beta_t = get_beta(t)

        dS = -beta_t * S * I
        dI = beta_t * S * I - gamma * I - mu * I
        dR = gamma * I
        dD = mu * I

        return torch.stack([dS, dI, dR, dD], dim=-1)

    # Integrate using RK4
    t_start = t_eval[0].item()
    t_end = t_eval[-1].item()
    n_steps = int((t_end - t_start) / dt) + 1
    t_grid = torch.linspace(t_start, t_end, n_steps, device=device)

    state = initial_state.clone()
    solutions = []
    t_eval_idx = 0

    for i in range(n_steps - 1):
        t = t_grid[i].item()

        while t_eval_idx < len(t_eval) and t_eval[t_eval_idx].item() <= t + dt / 2:
            solutions.append(state.clone())
            t_eval_idx += 1

        # RK4 step
        k1 = sird_derivatives(state, t)
        k2 = sird_derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = sird_derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = sird_derivatives(state + dt * k3, t + dt)
        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ensure valid proportions
        state = torch.clamp(state, min=0, max=1)

    while t_eval_idx < len(t_eval):
        solutions.append(state.clone())
        t_eval_idx += 1

    return torch.stack(solutions, dim=1)


class SIRDTask(BenchmarkTask):
    """
    SIRD benchmark task with time-dependent contact rate.

    This task demonstrates inference with:
    - Function-valued (∞-dimensional) parameters
    - Time series observations
    - Mixture of global and local parameters
    """

    def __init__(
        self,
        beta_times: Optional[torch.Tensor] = None,
        observation_times: Optional[torch.Tensor] = None,
        noise_std: float = 0.05,
        t_span: Tuple[float, float] = (0.0, 40.0),
        n_beta_points: int = 20,
        n_observations: int = 10,
    ):
        """
        Args:
            beta_times: Times for beta function evaluation
            observation_times: Times for observations
            noise_std: Observation noise (log-normal scale)
            t_span: Time span for simulation
            n_beta_points: Number of points to discretize beta
            n_observations: Number of test observations
        """
        if beta_times is None:
            beta_times = torch.linspace(t_span[0], t_span[1], n_beta_points)

        if observation_times is None:
            observation_times = torch.linspace(t_span[0], t_span[1], 10)

        self.beta_times = beta_times
        self.observation_times = observation_times
        self.noise_std = noise_std
        self.t_span = t_span
        self.n_beta_points = len(beta_times)
        self.n_obs_times = len(observation_times)

        # Parameters: gamma, mu (global) + beta values (function-valued)
        n_params = 2 + self.n_beta_points

        # Data: I, R, D at each observation time (not S, as it's determined)
        n_data = self.n_obs_times * 3

        super().__init__(
            n_params=n_params,
            n_data=n_data,
            name="sird",
            n_observations=n_observations,
        )

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the prior:
        - γ, μ ~ U(0, 0.5)
        - β(t) ~ sigmoid(GP(0, k)) where k is RBF kernel
        """
        device = self.beta_times.device

        # Sample global parameters
        gamma = torch.rand(n_samples) * 0.5  # U(0, 0.5)
        mu = torch.rand(n_samples) * 0.5  # U(0, 0.5)

        # Sample beta function from GP prior (pre-sigmoid)
        beta_raw = sample_gp_prior(self.beta_times, n_samples)

        # Combine
        theta = torch.cat([
            gamma.unsqueeze(-1),
            mu.unsqueeze(-1),
            beta_raw,
        ], dim=-1)

        return theta

    def _extract_params(self, theta: torch.Tensor):
        """Extract gamma, mu, and beta from theta."""
        gamma = theta[:, 0]
        mu = theta[:, 1]
        beta_raw = theta[:, 2:]

        # Transform beta to (0, 1) using sigmoid
        beta = torch.sigmoid(beta_raw)

        return gamma, mu, beta

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate SIRD model with log-normal observation noise.
        """
        n_samples = theta.shape[0]
        device = theta.device

        gamma, mu, beta = self._extract_params(theta)

        # Solve ODE
        solution = solve_sird(
            beta,
            self.beta_times.to(device),
            gamma,
            mu,
            self.observation_times.to(device),
        )  # (batch_size, n_obs_times, 4)

        # Extract I, R, D (not S)
        observations = solution[:, :, 1:]  # (batch_size, n_obs_times, 3)

        # Add log-normal noise
        log_obs = torch.log(observations + 1e-8)
        noisy_log_obs = log_obs + torch.randn_like(log_obs) * self.noise_std
        noisy_obs = torch.exp(noisy_log_obs)

        # Flatten
        return noisy_obs.reshape(n_samples, -1)

    def get_dependency_structure(self):
        """
        SIRD structure:
        - Global parameters (gamma, mu) affect all
        - Local parameters (beta at time t) primarily affect observations near t
        - All observations are dependent through dynamics
        """
        n_global = 2
        n_local = self.n_beta_points

        # Parameter structure: global params are connected, local params form a chain
        param_structure = torch.zeros(self.n_params, self.n_params)

        # Global params fully connected
        param_structure[:n_global, :n_global] = 1.0

        # Local params form a Markov chain (adjacent times connected)
        for i in range(n_local):
            param_structure[n_global + i, n_global + i] = 1.0
            if i > 0:
                param_structure[n_global + i, n_global + i - 1] = 1.0
                param_structure[n_global + i - 1, n_global + i] = 1.0

        # Global affects local
        param_structure[:n_global, n_global:] = 1.0
        param_structure[n_global:, :n_global] = 1.0

        # Data structure: all observations connected
        data_structure = torch.ones(self.n_data, self.n_data)

        # All parameters affect all data (through dynamics)
        param_to_data = torch.ones(self.n_data, self.n_params)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }

    def get_beta_times(self) -> torch.Tensor:
        """Get the times for beta function evaluation."""
        return self.beta_times

    def get_observation_times(self) -> torch.Tensor:
        """Get the observation times."""
        return self.observation_times
