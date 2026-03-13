# Hodgkin-Huxley-inspired data sampler (deterministic, test-friendly)

"""
A lightweight Hodgkin-Huxley (HH) inspired data sampler compatible with the project\'s
data_samplers API: sample_prior, simulate, log_likelihood, get_mixture_weights.

Notes:
- This is a simplified, deterministic, single-neuron HH-like model used for unit tests
  and demonstrations. It uses a 4-dimensional state: V (membrane potential), m, h, n
  (gating variables).
- theta maps to external current and per-component conductance scales.
- The sampler supports mixture components for flexibility, but keeps the dynamics
  lightweight for quick experimentation.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

__all__ = ["HodgkinHuxleySampler"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class HodgkinHuxleySampler:
    def __init__(
        self,
        dim_theta: int = 3,
        dim_x: int = 4,  # [V, m, h, n]
        seed: int = 0,
        n_components: int = 1,
        A: Optional[np.ndarray] = None,
        Sigma_x: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        # Basic configuration and RNG
        self.dim_theta = int(dim_theta)
        self.dim_x = int(dim_x)
        self.rng = np.random.default_rng(seed)
        self.n_components = max(1, int(n_components))
        self.A = A  # Optional, not used directly in this lightweight HH sampler
        self.Sigma_x = Sigma_x  # Optional covariance for log-likelihood
        if weights is None:
            weights = np.ones(self.n_components) / self.n_components
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.size != self.n_components:
            # normalize to correct shape if needed
            self.weights = np.ones(self.n_components) / max(1, self.n_components)
        self.weights /= np.sum(self.weights)

        # default emission covariance for log-likelihood
        if self.Sigma_x is None:
            self.Sigma_x = np.eye(self.dim_x) * 0.5  # moderate noise
        else:
            self.Sigma_x = np.array(self.Sigma_x, dtype=float)
            if self.Sigma_x.ndim == 0:
                self.Sigma_x = np.eye(self.dim_x) * float(self.Sigma_x)

        # simple default x0 (initial state) if needed elsewhere
        self._x0 = np.zeros(self.dim_x, dtype=float)
        # standard HH constants (mV, mS/cm^2, uF/cm^2 -> normalized units)
        self.C_m = 1.0
        self.E_Na = 115.0  # mV
        self.E_K = -12.0  # mV
        self.E_L = -65.0  # mV
        self.g_Na_base = 120.0  # mS/cm^2
        self.g_K_base = 36.0
        self.g_L = 0.3
        # integration settings
        self._dt = 0.02  # ms per step
        self._t_end = 20.0  # ms
        self._steps = max(1, int(round(self._t_end / self._dt)))

    # ODE helpers: HH gating variable dynamics (standard formulations)
    def _alpha_m(self, V: float) -> float:
        num = 0.1 * (25.0 - V)
        den = np.expm1((25.0 - V) / 10.0)  # exp(x) - 1 with safety
        if abs(den) < 1e-12:
            den = 1e-12
        return num / den

    def _beta_m(self, V: float) -> float:
        return 4.0 * np.exp(-V / 18.0)

    def _alpha_h(self, V: float) -> float:
        return 0.07 * np.exp(-V / 20.0)

    def _beta_h(self, V: float) -> float:
        return 1.0 / (np.exp((30.0 - V) / 10.0) + 1.0)

    def _alpha_n(self, V: float) -> float:
        num = 0.01 * (10.0 - V)
        den = np.expm1((10.0 - V) / 10.0)
        if abs(den) < 1e-12:
            den = 1e-12
        return num / den

    def _beta_n(self, V: float) -> float:
        return 0.125 * np.exp(-V / 80.0)

    def _m_inf(self, V: float) -> float:
        a = self._alpha_m(V)
        b = self._beta_m(V)
        if a + b <= 0:
            return 0.0
        return a / (a + b)

    def _h_inf(self, V: float) -> float:
        a = self._alpha_h(V)
        b = self._beta_h(V)
        if a + b <= 0:
            return 0.0
        return a / (a + b)

    def _n_inf(self, V: float) -> float:
        a = self._alpha_n(V)
        b = self._beta_n(V)
        if a + b <= 0:
            return 0.0
        return a / (a + b)

    # RK4 integrator for a single trajectory
    def _rk4_step(self, V, m, h, n, dt, I_ext, g_Na, g_K):
        def dVdv(V, m, h, n):
            return (
                I_ext
                - g_Na * (m ** 3) * h * (V - self.E_Na)
                - g_K * (n ** 4) * (V - self.E_K)
                - self.g_L * (V - self.E_L)
            ) / self.C_m

        def dm_dt(V, m):
            return self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m

        def dh_dt(V, h):
            return self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h

        def dn_dt(V, n):
            return self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n

        k1_V = dVdv(V, m, h, n)
        k1_m = dm_dt(V, m)
        k1_h = dh_dt(V, h)
        k1_n = dn_dt(V, n)

        V2 = V + 0.5 * dt * k1_V
        m2 = m + 0.5 * dt * k1_m
        h2 = h + 0.5 * dt * k1_h
        n2 = n + 0.5 * dt * k1_n
        k2_V = dVdv(V2, m2, h2, n2)
        k2_m = dm_dt(V2, m2)
        k2_h = dh_dt(V2, h2)
        k2_n = dn_dt(V2, n2)

        V3 = V + 0.5 * dt * k2_V
        m3 = m + 0.5 * dt * k2_m
        h3 = h + 0.5 * dt * k2_h
        n3 = n + 0.5 * dt * k2_n
        k3_V = dVdv(V3, m3, h3, n3)
        k3_m = dm_dt(V3, m3)
        k3_h = dh_dt(V3, h3)
        k3_n = dn_dt(V3, n3)

        V4 = V + dt * k3_V
        m4 = m + dt * k3_m
        h4 = h + dt * k3_h
        n4 = n + dt * k3_n
        k4_V = dVdv(V4, m4, h4, n4)
        k4_m = dm_dt(V4, m4)
        k4_h = dh_dt(V4, h4)
        k4_n = dn_dt(V4, n4)

        V_next = V + (dt / 6.0) * (k1_V + 2.0 * k2_V + 2.0 * k3_V + k4_V)
        m_next = m + (dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m)
        h_next = h + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
        n_next = n + (dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)

        # clamp gating variables to [0, 1]
        m_next = max(0.0, min(1.0, m_next))
        h_next = max(0.0, min(1.0, h_next))
        n_next = max(0.0, min(1.0, n_next))
        return V_next, m_next, h_next, n_next

    def _hh_trajectory(self, theta: np.ndarray, comp: int) -> np.ndarray:
        # Map theta to external current and component-specific scale factors
        # Theta shape: (dim_theta,)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        # Basic injections: map first element to external drive, others to modest perturbations
        I_ext = float(max(0.0, theta[0] if theta.size > 0 else 0.0)) * 5.0  # scaled current
        # Component-dependent conductance scales
        comp_scale_Na = 0.8 + 0.4 * ((comp + 1) / max(1, self.n_components))
        comp_scale_K = 0.8 + 0.4 * ((self.n_components - comp) / max(1, self.n_components))
        g_Na = self.g_Na_base * comp_scale_Na
        g_K = self.g_K_base * comp_scale_K
        # initialize V and gating variables at rest using V0=-65 mV
        V = -65.0
        m = self._m_inf(V)
        h = self._h_inf(V)
        n = self._n_inf(V)

        # Run RK4 integration
        dt = self._dt
        steps = self._steps
        for _ in range(steps):
            V, m, h, n = self._rk4_step(V, m, h, n, dt, I_ext, g_Na, g_K)
        # Return final state vector with length dim_x
        vec = np.array([V, m, h, n], dtype=float)
        # If dim_x mismatches, pad or trim accordingly (best-effort)
        if vec.size != self.dim_x:
            if self.dim_x > vec.size:
                pad = np.zeros(self.dim_x - vec.size, dtype=float)
                vec = np.concatenate([vec, pad])
            else:
                vec = vec[: self.dim_x]
        return vec

    def sample_prior(self, n: int = 1) -> np.ndarray:
        if n <= 0:
            return np.empty((0, self.dim_theta))
        return self.rng.normal(size=(n, self.dim_theta)) if n > 1 else self.rng.normal(size=(self.dim_theta,))

    def simulate(self, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> np.ndarray:
        # Sample theta if not provided
        if theta is None:
            theta = self.sample_prior(1)
            theta = theta.reshape(-1)
        else:
            theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size < 1:
            theta = np.zeros(self.dim_theta, dtype=float)
        comp = component if component is not None else int(self.rng.integers(0, max(1, self.n_components)))
        # simulate a single trajectory for the given theta and component
        x = self._hh_trajectory(theta, comp)
        return x if self.dim_x > 1 else x.flatten()

    def log_likelihood(self, x: np.ndarray, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> float:
        # Simple Gaussian likelihood around HH trajectory mean
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim_x:
            # attempt to reshape or truncate/pad
            if x.size < self.dim_x:
                x = np.pad(x, (0, self.dim_x - x.size), mode="constant")
            else:
                x = x[: self.dim_x]
        if theta is None:
            theta = np.zeros(self.dim_theta, dtype=float)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size < self.dim_theta:
            theta = np.pad(theta, (0, self.dim_theta - theta.size), mode="constant")
        # mean:
        if component is not None:
            mean = self.simulate(theta, component=component)
        else:
            # mixture: average log-likelihoods with weights
            means = []
            for k in range(self.n_components):
                means.append(self.simulate(theta, component=k))
            mean = np.mean(np.stack(means, axis=0), axis=0)
        # covariance
        cov = self.Sigma_x
        # compute log pdf
        return _logpdf_gaussian(x, mean, cov)

    def get_mixture_weights(self) -> np.ndarray:
        return self.weights


def _logpdf_gaussian(x: np.ndarray, mean: np.ndarray, cov: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    d = mean.size
    if cov is None:
        cov = np.eye(d)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        var = float(cov)
        if var <= 0:
            var = 1e-8
        diff = x - mean
        return -0.5 * (np.sum((diff ** 2)) / var + d * np.log(2.0 * np.pi * var))
    # ensure proper shape
    if cov.shape != (d, d):
        # try to broadcast identity if wrong shape
        cov = np.eye(d) * float(cov.flatten()[0]) if cov.size == 1 else np.eye(d)
    try:
        L = np.linalg.cholesky(cov)
        solve = np.linalg.solve(L, x - mean)
        quad = np.sum(solve ** 2)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (quad + d * np.log(2.0 * np.pi) + logdet)
    except np.linalg.LinAlgError:
        inv = np.linalg.inv(cov)
        quad = (x - mean).T @ inv @ (x - mean)
        logdet = np.log(np.linalg.det(cov) + 1e-12)
        return -0.5 * (quad + d * np.log(2.0 * np.pi) + logdet)
