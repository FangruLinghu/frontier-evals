## data/simulator.py
"""Simulator implementations for all benchmark tasks in the Simformer paper.

This module provides classes to generate paired (theta, x) samples from various
simulators, including analytical models and ODE-based systems. It also supports
generating ground-truth posterior samples via MCMC for evaluation.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.experimental import ode
from typing import Dict, Tuple, List, Optional, Callable
import numpy as np
from functools import partial
import math

# Type aliases
PRNGKey = jnp.ndarray
ThetaDict = Dict[str, float]
XDict = Dict[str, float]
SampleFn = Callable[[PRNGKey], Tuple[ThetaDict, XDict]]


class BaseSimulator:
    """Abstract base class for all simulators."""

    def __init__(self, key: PRNGKey):
        self.key = key

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        """Generate a single (theta, x) pair from the joint distribution."""
        raise NotImplementedError

    def simulate(self, theta: ThetaDict) -> XDict:
        """Run forward simulation given parameters theta."""
        raise NotImplementedError

    def log_likelihood(self, theta: ThetaDict, x_observed: XDict) -> float:
        """Compute log p(x_observed | theta)."""
        raise NotImplementedError

    def log_prior(self, theta: ThetaDict) -> float:
        """Compute log p(theta)."""
        raise NotImplementedError

    def sample_posterior_mcmc(
        self,
        x_observed: XDict,
        num_chains: int = 10,
        total_samples: int = 4000
    ) -> jnp.ndarray:
        """Generate posterior samples using slice + MH-MCMC as in Appendix A2.2."""
        # This is a simplified implementation; a full MCMC would require more infrastructure
        # In practice, this would use libraries like blackjax or custom kernels
        raise NotImplementedError("MCMC implementation requires additional dependencies")


class GaussianLinearSimulator(BaseSimulator):
    """Gaussian Linear task from Lueckmann et al. (2021)."""

    def __init__(self, key: PRNGKey, dim: int = 10):
        super().__init__(key)
        self.dim = dim
        self.prior_std = jnp.sqrt(0.1)

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 2)
        # Sample theta ~ N(0, 0.1 * I)
        theta_vals = random.normal(keys[0], (self.dim,)) * self.prior_std
        theta_dict = {f"theta_{i}": float(theta_vals[i]) for i in range(self.dim)}
        
        # Sample x | theta ~ N(theta, 0.1 * I)
        x_vals = theta_vals + random.normal(keys[1], (self.dim,)) * self.prior_std
        x_dict = {f"x_{i}": float(x_vals[i]) for i in range(self.dim)}
        
        return theta_dict, x_dict

    def simulate(self, theta: ThetaDict) -> XDict:
        theta_vals = jnp.array([theta[f"theta_{i}"] for i in range(self.dim)])
        noise = random.normal(self.key, (self.dim,)) * self.prior_std
        x_vals = theta_vals + noise
        return {f"x_{i}": float(x_vals[i]) for i in range(self.dim)}

    def log_prior(self, theta: ThetaDict) -> float:
        theta_vals = jnp.array([theta[f"theta_{i}"] for i in range(self.dim)])
        return jnp.sum(jax.scipy.stats.norm.logpdf(theta_vals, 0.0, self.prior_std))

    def log_likelihood(self, theta: ThetaDict, x_observed: XDict) -> float:
        theta_vals = jnp.array([theta[f"theta_{i}"] for i in range(self.dim)])
        x_vals = jnp.array([x_observed[f"x_{i}"] for i in range(self.dim)])
        return jnp.sum(jax.scipy.stats.norm.logpdf(x_vals, theta_vals, self.prior_std))


class GaussianMixtureSimulator(BaseSimulator):
    """Gaussian Mixture task."""

    def __init__(self, key: PRNGKey):
        super().__init__(key)

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 3)
        # Sample theta ~ Uniform(-10, 10)
        theta_val = random.uniform(keys[0], minval=-10.0, maxval=10.0)
        theta_dict = {"theta": float(theta_val)}
        
        # Sample x | theta ~ 0.5*N(theta,I) + 0.5*N(theta,0.01*I)
        component = random.bernoulli(keys[1])
        cov_scale = 1.0 if component else 0.1
        x_vals = jnp.array([
            theta_val + random.normal(keys[2], ()) * cov_scale,
            theta_val + random.normal(keys[2], ()) * cov_scale
        ])
        x_dict = {f"x_{i}": float(x_vals[i]) for i in range(2)}
        
        return theta_dict, x_dict

    def log_prior(self, theta: ThetaDict) -> float:
        theta_val = theta["theta"]
        if -10.0 <= theta_val <= 10.0:
            return -jnp.log(20.0)  # Uniform prior density
        else:
            return -jnp.inf

    def log_likelihood(self, theta: ThetaDict, x_observed: XDict) -> float:
        theta_val = theta["theta"]
        x_vals = jnp.array([x_observed["x_0"], x_observed["x_1"]])
        
        # Mixture likelihood
        log_prob1 = jnp.sum(jax.scipy.stats.norm.logpdf(x_vals, theta_val, 1.0))
        log_prob2 = jnp.sum(jax.scipy.stats.norm.logpdf(x_vals, theta_val, 0.1))
        return jax.scipy.special.logsumexp(jnp.array([log_prob1, log_prob2]) + jnp.log(0.5))


class TwoMoonsSimulator(BaseSimulator):
    """Two Moons task."""

    def __init__(self, key: PRNGKey):
        super().__init__(key)

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 4)
        # Sample theta ~ Uniform([-1,1]^2)
        theta_vals = random.uniform(keys[0], (2,), minval=-1.0, maxval=1.0)
        theta_dict = {f"theta_{i}": float(theta_vals[i]) for i in range(2)}
        
        # Generate data according to paper's equation
        alpha = random.uniform(keys[1], minval=-math.pi/2, maxval=math.pi/2)
        r = random.normal(keys[2]) * 0.012 + 0.1  # N(0.1, 0.012^2)
        
        base_point = jnp.array([
            r * jnp.cos(alpha) + 0.25,
            r * jnp.sin(alpha)
        ])
        
        rotation_angle = (-theta_vals[0] + theta_vals[1]) / jnp.sqrt(2.0)
        translation = -jnp.abs(theta_vals[0] + theta_vals[1]) / jnp.sqrt(2.0)
        
        rotation_matrix = jnp.array([
            [jnp.cos(rotation_angle), -jnp.sin(rotation_angle)],
            [jnp.sin(rotation_angle), jnp.cos(rotation_angle)]
        ])
        
        x_vals = jnp.dot(rotation_matrix, base_point) + jnp.array([translation, 0.0])
        x_dict = {f"x_{i}": float(x_vals[i]) for i in range(2)}
        
        return theta_dict, x_dict

    def log_prior(self, theta: ThetaDict) -> float:
        theta_vals = jnp.array([theta["theta_0"], theta["theta_1"]])
        in_bounds = jnp.all((theta_vals >= -1.0) & (theta_vals <= 1.0))
        return -jnp.log(4.0) if in_bounds else -jnp.inf


class SLCPSimulator(BaseSimulator):
    """SLCP (Simple Likelihood Complex Posterior) task."""

    def __init__(self, key: PRNGKey):
        super().__init__(key)

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 6)
        # Sample theta ~ Uniform([-3,3]^5)
        theta_vals = random.uniform(keys[0], (5,), minval=-3.0, maxval=3.0)
        theta_dict = {f"theta_{i}": float(theta_vals[i]) for i in range(5)}
        
        # Compute mean and covariance from theta
        mu_theta = jnp.array([theta_vals[0]])
        tanh_term = jnp.tanh(theta_vals[4]) * theta_vals[2]**2 * theta_vals[3]**2
        sigma_theta = jnp.array([
            [theta_vals[1], tanh_term],
            [tanh_term, theta_vals[3]**2]
        ])
        
        # Generate 4 i.i.d. observations
        x_vals = []
        for i in range(4):
            x_i = random.multivariate_normal(keys[i+1], mu_theta, sigma_theta)
            x_vals.extend([float(x_i[0]), float(x_i[1])])
        
        x_dict = {f"x_{i}": x_vals[i] for i in range(8)}
        return theta_dict, x_dict

    def log_prior(self, theta: ThetaDict) -> float:
        theta_vals = jnp.array([theta[f"theta_{i}"] for i in range(5)])
        in_bounds = jnp.all((theta_vals >= -3.0) & (theta_vals <= 3.0))
        return -jnp.log(6.0**5) if in_bounds else -jnp.inf


class TreeSimulator(BaseSimulator):
    """Tree-shaped hierarchical task."""

    def __init__(self, key: PRNGKey):
        super().__init__(key)

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 7)
        # Hierarchical parameter generation
        theta_0 = random.normal(keys[0])  # N(0,1)
        theta_1 = theta_0 + random.normal(keys[1])  # N(theta_0,1)
        theta_2 = theta_0 + random.normal(keys[2])  # N(theta_0,1)
        
        theta_dict = {
            "theta_0": float(theta_0),
            "theta_1": float(theta_1),
            "theta_2": float(theta_2)
        }
        
        # Generate observations
        x_1 = jnp.sin(theta_1)**2 + random.normal(keys[3]) * 0.2
        x_2 = 0.1 * theta_1**2 + random.normal(keys[4]) * 0.2
        x_3 = 0.1 * theta_2**2 + random.normal(keys[5]) * 0.6
        x_4 = jnp.cos(theta_2)**2 + random.normal(keys[6]) * 0.6
        
        x_dict = {
            "x_1": float(x_1),
            "x_2": float(x_2),
            "x_3": float(x_3),
            "x_4": float(x_4)
        }
        
        return theta_dict, x_dict

    def log_prior(self, theta: ThetaDict) -> float:
        t0, t1, t2 = theta["theta_0"], theta["theta_1"], theta["theta_2"]
        log_p_t0 = jax.scipy.stats.norm.logpdf(t0, 0.0, 1.0)
        log_p_t1 = jax.scipy.stats.norm.logpdf(t1, t0, 1.0)
        log_p_t2 = jax.scipy.stats.norm.logpdf(t2, t0, 1.0)
        return log_p_t0 + log_p_t1 + log_p_t2


class HMMSimulator(BaseSimulator):
    """Hidden Markov Model task."""

    def __init__(self, key: PRNGKey, n_steps: int = 10):
        super().__init__(key)
        self.n_steps = n_steps
        self.transition_std = 0.5
        self.observation_std = 0.5

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 2*self.n_steps + 1)
        # Generate latent states (theta_i)
        theta_vals = []
        theta_prev = random.normal(keys[0]) * self.transition_std
        
        for i in range(self.n_steps):
            if i == 0:
                theta_i = theta_prev
            else:
                theta_i = theta_prev + random.normal(keys[i]) * self.transition_std
            theta_vals.append(theta_i)
            theta_prev = theta_i
        
        theta_dict = {f"theta_{i}": float(theta_vals[i]) for i in range(self.n_steps)}
        
        # Generate observations x_i ~ N(theta_i^2, 0.5^2)
        x_dict = {}
        for i in range(self.n_steps):
            x_i = theta_vals[i]**2 + random.normal(keys[self.n_steps + i]) * self.observation_std
            x_dict[f"x_{i}"] = float(x_i)
            
        return theta_dict, x_dict

    def log_prior(self, theta: ThetaDict) -> float:
        theta_vals = jnp.array([theta[f"theta_{i}"] for i in range(self.n_steps)])
        log_prob = jax.scipy.stats.norm.logpdf(theta_vals[0], 0.0, self.transition_std)
        
        for i in range(1, self.n_steps):
            log_prob += jax.scipy.stats.norm.logpdf(
                theta_vals[i], theta_vals[i-1], self.transition_std
            )
        return log_prob


class LotkaVolterraSimulator(BaseSimulator):
    """Lotka-Volterra predator-prey model with irregular observations."""

    def __init__(
        self, 
        key: PRNGKey,
        duration: float = 50.0,
        noise_std: float = 0.1,
        initial_conditions: Tuple[float, float] = (1.0, 1.0)
    ):
        super().__init__(key)
        self.duration = duration
        self.noise_std = noise_std
        self.initial_conditions = initial_conditions

    def _ode_fn(self, state, t, params):
        """Right-hand side of the Lotka-Volterra equations."""
        x, y = state
        alpha, beta, gamma, delta = params
        dx_dt = alpha * x - beta * x * y
        dy_dt = delta * x * y - gamma * y
        return jnp.array([dx_dt, dy_dt])

    def _simulate_trajectory(self, params: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
        """Solve ODE and return trajectory at specified times."""
        solution = ode.odeint(
            lambda y, t: self._ode_fn(y, t, params),
            jnp.array(self.initial_conditions),
            times,
            rtol=1e-6,
            atol=1e-6
        )
        return solution

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 3)
        
        # Sample parameters from sigmoid-transformed normal [1,3]
        raw_params = random.normal(keys[0], (4,))
        params = 1.0 + 2.0 / (1.0 + jnp.exp(-raw_params))  # Sigmoid transformation
        
        theta_dict = {
            "alpha": float(params[0]),
            "beta": float(params[1]),
            "gamma": float(params[2]),
            "delta": float(params[3])
        }
        
        # Generate irregular observation times
        n_obs = random.randint(keys[1], (), 2, 10)  # 2-10 observations
        obs_times = random.uniform(keys[2], (n_obs,)) * self.duration
        obs_times = jnp.sort(obs_times)
        
        # Solve ODE
        trajectory = self._simulate_trajectory(params, obs_times)
        
        # Add Gaussian noise
        noise = random.normal(keys[2], trajectory.shape) * self.noise_std
        noisy_trajectory = trajectory + noise
        
        # Create observation dictionary
        x_dict = {}
        for i, t in enumerate(obs_times):
            x_dict[f"prey_t{t:.3f}"] = float(noisy_trajectory[i, 0])
            x_dict[f"predator_t{t:.3f}"] = float(noisy_trajectory[i, 1])
        
        return theta_dict, x_dict


class SIRDSimulator(BaseSimulator):
    """SIRD model with time-dependent contact rate."""

    def __init__(
        self,
        key: PRNGKey,
        duration: float = 50.0,
        gp_kernel_scale: float = 7.0,
        gp_amplitude: float = 2.5,
        noise_std: float = 0.05,
        population: float = 1.0
    ):
        super().__init__(key)
        self.duration = duration
        self.gp_kernel_scale = gp_kernel_scale
        self.gp_amplitude = gp_amplitude
        self.noise_std = noise_std
        self.population = population

    def _rbf_kernel(self, t1: jnp.ndarray, t2: jnp.ndarray) -> jnp.ndarray:
        """RBF kernel for GP prior on beta(t)."""
        diff = t1[:, None] - t2[None, :]
        return self.gp_amplitude**2 * jnp.exp(-0.5 * (diff / self.gp_kernel_scale)**2)

    def _sample_beta_function(self, key: PRNGKey, times: jnp.ndarray) -> jnp.ndarray:
        """Sample a realization of beta(t) from GP prior."""
        n_times = len(times)
        K = self._rbf_kernel(times, times) + 1e-6 * jnp.eye(n_times)
        chol_K = jnp.linalg.cholesky(K)
        gp_sample = jnp.dot(chol_K, random.normal(key, (n_times,)))
        beta_raw = gp_sample
        # Apply sigmoid to constrain to [0,1]
        beta_t = 1.0 / (1.0 + jnp.exp(-beta_raw))
        return beta_t

    def _ode_fn(self, state, t, params_and_beta_interp):
        """Right-hand side of SIRD equations with interpolated beta(t)."""
        S, I, R, D = state
        gamma, mu = params_and_beta_interp[:2]
        beta_t = params_and_beta_interp[2]  # Interpolated value
        
        dS_dt = -beta_t * S * I
        dI_dt = beta_t * S * I - gamma * I - mu * I
        dR_dt = gamma * I
        dD_dt = mu * I
        
        return jnp.array([dS_dt, dI_dt, dR_dt, dD_dt])

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 4)
        
        # Sample global parameters
        gamma = random.uniform(keys[0], minval=0.0, maxval=0.5)
        mu = random.uniform(keys[1], minval=0.0, maxval=0.5)
        
        # Sample beta(t) function
        n_time_points = 50
        times = jnp.linspace(0, self.duration, n_time_points)
        beta_t = self._sample_beta_function(keys[2], times)
        
        # Initial conditions (all susceptible except small infected)
        init_state = jnp.array([0.99, 0.01, 0.0, 0.0]) * self.population
        
        # Solve ODE with time-varying beta
        def ode_wrapper(state, t):
            # Interpolate beta at current time
            beta_interp = jnp.interp(t, times, beta_t)
            params = jnp.array([gamma, mu, beta_interp])
            return self._ode_fn(state, t, params)
        
        solution = ode.odeint(
            ode_wrapper,
            init_state,
            times,
            rtol=1e-6,
            atol=1e-6
        )
        
        # Generate irregular observations with log-normal noise
        n_obs = random.randint(keys[3], (), 3, 8)
        obs_times_idx = random.choice(keys[3], len(times), (n_obs,), replace=False)
        obs_times_idx = jnp.sort(obs_times_idx)
        
        x_dict = {}
        for idx in obs_times_idx:
            t = float(times[idx])
            state = solution[idx]
            # Add log-normal noise (multiplicative)
            noise_factor = jnp.exp(random.normal(keys[3]) * self.noise_std)
            x_dict[f"S_t{t:.3f}"] = float(state[0] * noise_factor)
            x_dict[f"I_t{t:.3f}"] = float(state[1] * noise_factor)
            x_dict[f"R_t{t:.3f}"] = float(state[2] * noise_factor)
            x_dict[f"D_t{t:.3f}"] = float(state[3] * noise_factor)
        
        theta_dict = {
            "gamma": float(gamma),
            "mu": float(mu)
        }
        
        # Also store some beta values for potential conditioning
        for i in range(0, len(beta_t), 10):
            theta_dict[f"beta_t{times[i]:.3f}"] = float(beta_t[i])
        
        return theta_dict, x_dict


class HodgkinHuxleySimulator(BaseSimulator):
    """Hodgkin-Huxley neuron model with summary statistics."""

    def __init__(
        self,
        key: PRNGKey,
        duration: float = 200.0,
        stim_start: float = 50.0,
        stim_end: float = 150.0,
        stim_amp: float = 4.0,
        dt: float = 0.1
    ):
        super().__init__(key)
        self.duration = duration
        self.stim_start = stim_start
        self.stim_end = stim_end
        self.stim_amp = stim_amp
        self.dt = dt
        self.times = jnp.arange(0, duration, dt)

    def _efun(self, x):
        """Helper function for rate equations."""
        return jnp.where(x < 1e-4, 1 - x/2, x / (jnp.exp(x) - 1.0))

    def _alpha_m(self, V):
        return 0.32 * self._efun(-0.25*(V + 52.0)) / 0.25

    def _beta_m(self, V):
        return 0.28 * self._efun(0.2*(V + 25.0)) / 0.2

    def _alpha_h(self, V):
        return 0.128 * jnp.exp(-(V + 47.0)/18.0)

    def _beta_h(self, V):
        return 4.0 / (1.0 + jnp.exp(-(V + 17.0)/5.0))

    def _alpha_n(self, V):
        return 0.032 * self._efun(-0.2*(V + 30.0)) / 0.2

    def _beta_n(self, V):
        return 0.5 * jnp.exp(-(V + 10.0)/40.0)

    def _ode_fn(self, state, t, params):
        """Full Hodgkin-Huxley equations."""
        V, m, h, n = state
        g_Na, g_K, g_L, E_Na, E_K, E_L, C_m = params
        
        # Input current
        I_inj = self.stim_amp if (t >= self.stim_start and t <= self.stim_end) else 0.0
        
        # Currents
        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_K = g_K * n**4 * (V - E_K)
        I_L = g_L * (V - E_L)
        
        # Membrane equation
        dV_dt = (I_inj - I_Na - I_K - I_L) / C_m + 0.05 * random.normal(self.key)
        
        # Gating variables
        dm_dt = self._alpha_m(V) * (1 - m) - self._beta_m(V) * m
        dh_dt = self._alpha_h(V) * (1 - h) - self._beta_h(V) * h
        dn_dt = self._alpha_n(V) * (1 - n) - self._beta_n(V) * n
        
        return jnp.array([dV_dt, dm_dt, dh_dt, dn_dt])

    def _extract_summary_stats(self, voltage_trace: jnp.ndarray) -> Dict[str, float]:
        """Extract summary statistics as in Goncalves et al. (2020)."""
        # Placeholder implementation - actual stats should match cited work
        stats = {}
        stats["voltage_mean"] = float(jnp.mean(voltage_trace))
        stats["voltage_std"] = float(jnp.std(voltage_trace))
        stats["spike_count"] = float(jnp.sum((voltage_trace[1:] > 0) & (voltage_trace[:-1] <= 0)))
        
        # Find spike times for interval calculations
        spike_times = jnp.where((voltage_trace > 0) & (jnp.concatenate([jnp.array([False]), voltage_trace[:-1] <= 0])))[0] * self.dt
        if len(spike_times) > 1:
            isi = jnp.diff(spike_times)
            stats["isi_mean"] = float(jnp.mean(isi))
            stats["isi_cv"] = float(jnp.std(isi) / jnp.mean(isi))
        else:
            stats["isi_mean"] = 0.0
            stats["isi_cv"] = 0.0
            
        return stats

    def _calculate_energy(self, m_trace: jnp.ndarray, h_trace: jnp.ndarray, V_trace: jnp.ndarray, E_Na: float) -> float:
        """Calculate energy consumption based on sodium charge flux."""
        # Simplified energy calculation from Deistler et al. (2022b)
        Na_current = m_trace**3 * h_trace * (V_trace - E_Na)
        energy = jnp.sum(jnp.abs(Na_current)) * self.dt
        return float(energy)

    def sample(self, key: PRNGKey) -> Tuple[ThetaDict, XDict]:
        keys = random.split(key, 2)
        
        # Sample 7 parameters
        param_keys = random.split(keys[0], 7)
        g_Na = float(random.uniform(param_keys[0], minval=10.0, maxval=120.0))
        g_K = float(random.uniform(param_keys[1], minval=5.0, maxval=36.0))
        g_L = float(random.uniform(param_keys[2], minval=0.1, maxval=0.3))
        E_Na = float(random.uniform(param_keys[3], minval=40.0, maxval=60.0))
        E_K = float(random.uniform(param_keys[4], minval=-80.0, maxval=-60.0))
        E_L = float(random.uniform(param_keys[5], minval=-70.0, maxval=-50.0))
        C_m = float(random.uniform(param_keys[6], minval=0.8, maxval=1.2))
        
        theta_dict = {
            "g_Na": g_Na,
            "g_K": g_K,
            "g_L": g_L,
            "E_Na": E_Na,
            "E_K": E_K,
            "E_L": E_L,
            "C_m": C_m
        }
        
        # Initial state
        init_state = jnp.array([-65.0, 0.05, 0.6, 0.31])
        params = jnp.array([g_Na, g_K, g_L, E_Na, E_K, E_L, C_m])
        
        # Solve ODE
        solution = ode.odeint(
            lambda state, t: self._ode_fn(state, t, params),
            init_state,
            self.times,
            rtol=1e-6,
            atol=1e-6
        )
        
        voltage_trace = solution[:, 0]
        m_trace = solution[:, 1]
        h_trace = solution[:, 2]
        n_trace = solution[:, 3]
        
        # Extract summary statistics
        x_dict = self._extract_summary_stats(voltage_trace)
        
        # Add energy consumption as extra statistic
        energy = self._calculate_energy(m_trace, h_trace, voltage_trace, E_Na)
        x_dict["energy"] = energy
        
        return theta_dict, x_dict


def get_simulator(task_name: str, key: PRNGKey) -> BaseSimulator:
    """Factory function to create simulator instances."""
    simulator_map = {
        "Gaussian_Linear": lambda: GaussianLinearSimulator(key),
        "Gaussian_Mixture": lambda: GaussianMixtureSimulator(key),
        "Two_Moons": lambda: TwoMoonsSimulator(key),
        "SLCP": lambda: SLCPSimulator(key),
        "Tree": lambda: TreeSimulator(key),
        "HMM": lambda: HMMSimulator(key),
        "Lotka_Volterra": lambda: LotkaVolterraSimulator(key),
        "SIRD": lambda: SIRDSimulator(key),
        "Hodgkin_Huxley": lambda: HodgkinHuxleySimulator(key)
    }
    
    if task_name not in simulator_map:
        raise ValueError(f"Unknown task: {task_name}")
    
    return simulator_map[task_name]()
