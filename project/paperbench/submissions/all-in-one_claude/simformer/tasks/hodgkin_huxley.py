"""
Hodgkin-Huxley task.

A biophysical model of neuronal membrane dynamics.

The model describes the voltage across the membrane of neurons using
coupled ODEs for voltage and gating variables.

Parameters:
- C_m: Membrane capacitance
- g_Na: Maximum sodium conductance
- g_K: Maximum potassium conductance
- g_L: Leak conductance
- E_Na: Sodium reversal potential
- E_K: Potassium reversal potential
- E_L: Leak reversal potential

This task uses summary statistics as in Gonçalves et al. (2020).
"""

import torch
import math
from typing import Optional, Tuple, List
from simformer.tasks.base import BenchmarkTask


def alpha_m(V, V0=-65.0):
    """Rate function alpha_m."""
    x = -0.25 * (V - V0 - 13.0)
    return 0.32 * efun(x) / 0.25


def beta_m(V, V0=-65.0):
    """Rate function beta_m."""
    x = 0.2 * (V - V0 - 40.0)
    return 0.28 * efun(x) / 0.2


def alpha_h(V, V0=-65.0):
    """Rate function alpha_h."""
    return 0.128 * torch.exp(-(V - V0 - 17.0) / 18.0)


def beta_h(V, V0=-65.0):
    """Rate function beta_h."""
    return 4.0 / (1.0 + torch.exp(-(V - V0 - 40.0) / 5.0))


def alpha_n(V, V0=-65.0):
    """Rate function alpha_n."""
    x = -0.2 * (V - V0 - 15.0)
    return 0.032 * efun(x) / 0.2


def beta_n(V, V0=-65.0):
    """Rate function beta_n."""
    return 0.5 * torch.exp(-(V - V0 - 10.0) / 40.0)


def efun(x):
    """Helper function for rate functions."""
    # efun(x) = x / (exp(x) - 1) for x != 0, else 1
    return torch.where(
        torch.abs(x) < 1e-4,
        1 - x / 2,
        x / (torch.exp(x) - 1.0 + 1e-8)
    )


def hh_derivatives(state, t, params, I_inj):
    """
    Compute Hodgkin-Huxley derivatives.

    State: [V, m, h, n, H]
    - V: Membrane voltage
    - m, h, n: Gating variables
    - H: Integrated sodium current (for energy)
    """
    V, m, h, n, H = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
    C_m, g_Na, g_K, g_L, E_Na, E_K, E_L = (
        params[:, 0], params[:, 1], params[:, 2], params[:, 3],
        params[:, 4], params[:, 5], params[:, 6]
    )

    # Currents
    I_Na = g_Na * m ** 3 * h * (V - E_Na)
    I_K = g_K * n ** 4 * (V - E_K)
    I_L = g_L * (V - E_L)

    # Voltage derivative
    dV = (I_inj - I_Na - I_K - I_L) / C_m

    # Gating variable derivatives
    dm = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1 - h) - beta_h(V) * h
    dn = alpha_n(V) * (1 - n) - beta_n(V) * n

    # Energy (integrated sodium current)
    dH = g_Na * m ** 3 * h * (V - E_Na)

    return torch.stack([dV, dm, dh, dn, dH], dim=-1)


def solve_hh(
    params: torch.Tensor,
    t_eval: torch.Tensor,
    I_inj_fn,
    dt: float = 0.025,
    noise_std: float = 0.05,
) -> torch.Tensor:
    """
    Solve Hodgkin-Huxley equations.

    Args:
        params: Parameters [C_m, g_Na, g_K, g_L, E_Na, E_K, E_L]
        t_eval: Times to evaluate
        I_inj_fn: Function returning injected current at time t
        dt: Time step
        noise_std: Noise to add to voltage

    Returns:
        Solution of shape (batch_size, len(t_eval), 5)
    """
    batch_size = params.shape[0]
    device = params.device

    # Initial state at resting potential
    V0 = -65.0
    m0 = alpha_m(torch.tensor([V0])) / (alpha_m(torch.tensor([V0])) + beta_m(torch.tensor([V0])))
    h0 = alpha_h(torch.tensor([V0])) / (alpha_h(torch.tensor([V0])) + beta_h(torch.tensor([V0])))
    n0 = alpha_n(torch.tensor([V0])) / (alpha_n(torch.tensor([V0])) + beta_n(torch.tensor([V0])))

    initial_state = torch.tensor([[V0, m0.item(), h0.item(), n0.item(), 0.0]], device=device)
    initial_state = initial_state.expand(batch_size, -1).clone()

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

        I_inj = I_inj_fn(t)

        # Euler-Maruyama step (with noise)
        deriv = hh_derivatives(state, t, params, I_inj)
        noise = torch.zeros_like(state)
        noise[:, 0] = torch.randn(batch_size, device=device) * noise_std * math.sqrt(dt)
        state = state + dt * deriv + noise

        # Clamp gating variables to [0, 1]
        state[:, 1:4] = torch.clamp(state[:, 1:4], 0, 1)

    while t_eval_idx < len(t_eval):
        solutions.append(state.clone())
        t_eval_idx += 1

    return torch.stack(solutions, dim=1)


def compute_summary_statistics(voltage_trace: torch.Tensor, dt: float = 0.025) -> torch.Tensor:
    """
    Compute summary statistics from voltage trace.

    Based on Gonçalves et al. (2020).

    Statistics include:
    - Number of spikes
    - Mean and std of ISI
    - Mean spike amplitude
    - Resting potential estimate
    - etc.
    """
    batch_size = voltage_trace.shape[0]
    n_time = voltage_trace.shape[1]

    stats = []

    for b in range(batch_size):
        V = voltage_trace[b]

        # Spike detection (threshold crossing at 0 mV)
        threshold = 0.0
        above = V > threshold
        crossings = torch.where(above[1:] & ~above[:-1])[0]
        n_spikes = len(crossings)

        # Basic statistics
        mean_V = V.mean()
        std_V = V.std()
        min_V = V.min()
        max_V = V.max()

        # ISI statistics
        if n_spikes > 1:
            isis = (crossings[1:] - crossings[:-1]).float() * dt
            mean_isi = isis.mean()
            std_isi = isis.std() if len(isis) > 1 else torch.tensor(0.0)
        else:
            mean_isi = torch.tensor(0.0)
            std_isi = torch.tensor(0.0)

        # Combine statistics
        stat = torch.stack([
            torch.tensor(float(n_spikes)),
            mean_V,
            std_V,
            min_V,
            max_V,
            mean_isi,
            std_isi,
        ])
        stats.append(stat)

    return torch.stack(stats)


class HodgkinHuxleyTask(BenchmarkTask):
    """
    Hodgkin-Huxley benchmark task.

    This task uses summary statistics of voltage traces for inference,
    as the raw time series would be too high-dimensional.

    Demonstrates:
    - Complex ODE-based simulator
    - Summary statistics
    - Interval constraints via diffusion guidance
    """

    def __init__(
        self,
        t_max: float = 200.0,
        dt: float = 0.025,
        I_inj_start: float = 50.0,
        I_inj_end: float = 150.0,
        I_inj_amplitude: float = 4.0,
        n_summary_stats: int = 7,
        include_energy: bool = True,
        n_observations: int = 10,
    ):
        """
        Args:
            t_max: Maximum simulation time (ms)
            dt: Time step (ms)
            I_inj_start: Start of current injection (ms)
            I_inj_end: End of current injection (ms)
            I_inj_amplitude: Amplitude of injected current (mA)
            n_summary_stats: Number of summary statistics
            include_energy: Whether to include energy as additional statistic
            n_observations: Number of test observations
        """
        self.t_max = t_max
        self.dt = dt
        self.I_inj_start = I_inj_start
        self.I_inj_end = I_inj_end
        self.I_inj_amplitude = I_inj_amplitude
        self.include_energy = include_energy

        n_data = n_summary_stats + (1 if include_energy else 0)

        super().__init__(
            n_params=7,  # C_m, g_Na, g_K, g_L, E_Na, E_K, E_L
            n_data=n_data,
            name="hodgkin_huxley",
            n_observations=n_observations,
        )

        # Default parameter ranges (for prior)
        self.param_ranges = {
            'C_m': (0.5, 2.0),
            'g_Na': (60.0, 150.0),
            'g_K': (10.0, 40.0),
            'g_L': (0.05, 0.5),
            'E_Na': (40.0, 60.0),
            'E_K': (-90.0, -70.0),
            'E_L': (-80.0, -50.0),
        }

    def _I_inj(self, t: float) -> float:
        """Injected current function."""
        if self.I_inj_start <= t <= self.I_inj_end:
            return self.I_inj_amplitude
        return 0.0

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """
        Sample from uniform prior over parameter ranges.
        """
        theta = torch.zeros(n_samples, 7)

        ranges = list(self.param_ranges.values())
        for i, (low, high) in enumerate(ranges):
            theta[:, i] = torch.rand(n_samples) * (high - low) + low

        return theta

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate Hodgkin-Huxley and compute summary statistics.
        """
        n_samples = theta.shape[0]
        device = theta.device

        # Time points
        t_eval = torch.arange(0, self.t_max, self.dt, device=device)

        # Solve HH
        solution = solve_hh(
            theta,
            t_eval,
            self._I_inj,
            dt=self.dt,
        )  # (batch_size, n_time, 5)

        # Extract voltage and energy
        voltage = solution[:, :, 0]  # (batch_size, n_time)
        energy = solution[:, -1, 4]  # Final integrated sodium current

        # Compute summary statistics
        stats = compute_summary_statistics(voltage, self.dt)  # (batch_size, n_stats)

        # Add energy if requested
        if self.include_energy:
            stats = torch.cat([stats, energy.unsqueeze(-1)], dim=-1)

        return stats

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log p(θ) for uniform prior."""
        in_bounds = torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device)

        ranges = list(self.param_ranges.values())
        for i, (low, high) in enumerate(ranges):
            in_bounds = in_bounds & (theta[:, i] >= low) & (theta[:, i] <= high)

        log_vol = sum(math.log(high - low) for low, high in ranges)

        return torch.where(
            in_bounds,
            torch.tensor(-log_vol, device=theta.device),
            torch.tensor(float("-inf"), device=theta.device),
        )

    def get_dependency_structure(self):
        """
        HH structure: all parameters interact in complex ways.
        """
        # All parameters affect all others through the dynamics
        param_structure = torch.ones(7, 7)

        # All summary statistics are related
        data_structure = torch.ones(self.n_data, self.n_data)

        # All parameters affect all statistics
        param_to_data = torch.ones(self.n_data, 7)

        return {
            "param_structure": param_structure,
            "data_structure": data_structure,
            "param_to_data": param_to_data,
        }

    def get_param_names(self) -> List[str]:
        """Get parameter names."""
        return list(self.param_ranges.keys())

    def get_stat_names(self) -> List[str]:
        """Get summary statistic names."""
        names = ['n_spikes', 'mean_V', 'std_V', 'min_V', 'max_V', 'mean_ISI', 'std_ISI']
        if self.include_energy:
            names.append('energy')
        return names
