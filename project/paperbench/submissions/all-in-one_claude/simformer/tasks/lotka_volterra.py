"""
Lotka-Volterra task.

Classic predator-prey dynamics model from ecology.

The model describes the interaction between prey (x) and predator (y) populations:
dx/dt = αx - βxy
dy/dt = δxy - γy

Parameters: θ = [α, β, γ, δ] (growth, hunting, death, reproduction rates)

This task supports unstructured observations at arbitrary time points.
"""

import torch
import math
from typing import Optional, Tuple, List
from simformer.tasks.base import BenchmarkTask


def lotka_volterra_ode(state: torch.Tensor, t: float, params: torch.Tensor) -> torch.Tensor:
    """
    Compute derivatives for Lotka-Volterra ODE.

    Args:
        state: Current state [prey, predator] of shape (batch_size, 2)
        t: Current time
        params: Parameters [α, β, γ, δ] of shape (batch_size, 4)

    Returns:
        Derivatives [dx/dt, dy/dt] of shape (batch_size, 2)
    """
    x, y = state[:, 0], state[:, 1]
    alpha, beta, gamma, delta = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y

    return torch.stack([dx, dy], dim=-1)


def solve_lotka_volterra(
    params: torch.Tensor,
    t_span: Tuple[float, float],
    t_eval: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    method: str = "rk4",
    dt: float = 0.01,
) -> torch.Tensor:
    """
    Solve Lotka-Volterra ODE using simple numerical integration.

    Args:
        params: Parameters [α, β, γ, δ] of shape (batch_size, 4)
        t_span: (t_start, t_end)
        t_eval: Times at which to evaluate the solution
        initial_state: Initial [prey, predator] state
        method: Integration method ("euler" or "rk4")
        dt: Time step for integration

    Returns:
        Solution of shape (batch_size, len(t_eval), 2)
    """
    batch_size = params.shape[0]
    device = params.device

    if initial_state is None:
        # Default initial state from paper
        initial_state = torch.tensor([[1.0, 0.5]], device=device).expand(batch_size, -1)

    # Create time grid for integration
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_grid = torch.linspace(t_start, t_end, n_steps, device=device)

    # Initialize solution
    state = initial_state.clone()
    solutions = []
    t_eval_idx = 0

    for i in range(n_steps - 1):
        t = t_grid[i]

        # Store solution at evaluation times
        while t_eval_idx < len(t_eval) and t_eval[t_eval_idx] <= t + dt / 2:
            solutions.append(state.clone())
            t_eval_idx += 1

        # Integration step
        if method == "euler":
            deriv = lotka_volterra_ode(state, t.item(), params)
            state = state + dt * deriv
        else:  # RK4
            k1 = lotka_volterra_ode(state, t.item(), params)
            k2 = lotka_volterra_ode(state + 0.5 * dt * k1, t.item() + 0.5 * dt, params)
            k3 = lotka_volterra_ode(state + 0.5 * dt * k2, t.item() + 0.5 * dt, params)
            k4 = lotka_volterra_ode(state + dt * k3, t.item() + dt, params)
            state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ensure non-negative populations
        state = torch.clamp(state, min=1e-6)

    # Store remaining evaluation points
    while t_eval_idx < len(t_eval):
        solutions.append(state.clone())
        t_eval_idx += 1

    return torch.stack(solutions, dim=1)


class LotkaVolterraTask(BenchmarkTask):
    """
    Lotka-Volterra benchmark task.

    This task demonstrates inference with:
    - Unstructured observations at arbitrary time points
    - Time series data
    - ODE-based simulators
    """

    def __init__(
        self,
        observation_times: Optional[torch.Tensor] = None,
        noise_std: float = 0.1,
        t_span: Tuple[float, float] = (0.0, 15.0),
        n_observations: int = 10,
    ):
        """
        Args:
            observation_times: Times at which to observe (default: regular grid)
            noise_std: Observation noise standard deviation
            t_span: Time span for simulation
            n_observations: Number of test observations
        """
        if observation_times is None:
            # Default: 15 observations evenly spaced
            observation_times = torch.linspace(t_span[0], t_span[1], 15)

        self.observation_times = observation_times
        self.noise_std = noise_std
        self.t_span = t_span
        self.n_obs_times = len(observation_times)

        # Each observation time gives 2 values (prey, predator)
        n_data = self.n_obs_times * 2

        super().__init__(
            n_params=4,
            n_data=n_data,
            name="lotka_volterra",
            n_observations=n_observations,
        )

    def _transform_params(self, theta_raw: torch.Tensor) -> torch.Tensor:
        """
        Transform raw parameters to valid ranges.

        Uses sigmoid transformation to ensure positive parameters.
        """
        # Sigmoid transformation scaled to reasonable ranges
        # α, β, γ, δ ∈ [0.5, 2.5] approximately
        theta = torch.sigmoid(theta_raw) * 2 + 0.5
        return theta

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the prior.

        Prior is a transformed Normal distribution.
        """
        # Sample from standard normal and will transform later
        theta_raw = torch.randn(n_samples, 4)
        return theta_raw

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate Lotka-Volterra dynamics with observation noise.
        """
        n_samples = theta.shape[0]
        device = theta.device

        # Transform parameters to valid ranges
        params = self._transform_params(theta)

        # Solve ODE
        solution = solve_lotka_volterra(
            params,
            self.t_span,
            self.observation_times.to(device),
            method="rk4",
        )  # (batch_size, n_obs_times, 2)

        # Add observation noise
        noise = torch.randn_like(solution) * self.noise_std
        noisy_solution = solution + noise

        # Flatten to (batch_size, n_data)
        return noisy_solution.reshape(n_samples, -1)

    def get_dependency_structure(self):
        """
        Lotka-Volterra structure:
        - All parameters affect all observations
        - Observations at different times are dependent through the dynamics
        """
        # All parameters can affect each other in the posterior
        param_structure = torch.ones(4, 4)

        # All data points are dependent (time series)
        data_structure = torch.ones(self.n_data, self.n_data)

        # All parameters affect all data
        param_to_data = torch.ones(self.n_data, 4)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }

    def get_observation_times(self) -> torch.Tensor:
        """Get the observation times."""
        return self.observation_times

    def set_observation_times(self, times: torch.Tensor):
        """Set new observation times (for unstructured observations)."""
        self.observation_times = times
        self.n_obs_times = len(times)
        self.n_data = self.n_obs_times * 2
