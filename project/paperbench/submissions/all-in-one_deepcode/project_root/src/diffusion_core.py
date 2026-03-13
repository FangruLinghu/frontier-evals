import numpy as np
from typing import Tuple, Union, Optional

"""
Forward diffusion core implementations for VPSDE and VESDE.

This module provides lightweight, numerically stable utilities to work with
forward diffusion processes used in the all-in-one diffusion-based inference plan.
The implementations are intentionally minimal and deterministic, suitable for
unit tests and small-scale experiments.

Public API:
- class VPSDE(beta_min=0.01, beta_max=10.0):
    - beta(t): scalar or array
    - f(x, t): drift term for x in the forward process: f(x,t) = -0.5 * beta(t) * x
    - g(t): diffusion coefficient: g(t) = sqrt(beta(t))
    - marginal_mean_and_var(x0, t): returns (mean_xt, var_xt) for x_t | x_0
    - sample_forward(x0, t, steps=50, rng=None): Euler–Maruyama simulation from 0 to t

- class VESDE(sigma_min=1e-4, sigma_max=15.0):
    - g(t): diffusion coefficient in the forward process
    - f(x,t) = 0 (Ornstein–Uhlenbeck like drift is zero here)
    - marginal_mean_and_var(x0, t): returns (mean, var) with mean = x0, var = ∫_0^t g(s)^2 ds
    - sample_forward(x0, t, steps=50, rng=None): Euler–Maruyama simulation from 0 to t

Notes:
- Time t is interpreted as a scalar in [0, 1] (or smaller) and is broadcastable
  with input x0 vectors of arbitrary shape.
- All computations are NumPy-based and deterministic given a seed.
"""

__all__ = ["VPSDE", "VESDE"]


def _as_array(x: Union[np.ndarray, float, int]) -> np.ndarray:
    return np.asarray(x)


class VPSDE:
    def __init__(self, beta_min: float = 0.01, beta_max: float = 10.0, steps: int = 1000):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.steps = max(1, int(steps))

    def beta(self, t: Union[float, np.ndarray]) -> np.ndarray:
        t_arr = _as_array(t)
        return self.beta_min + t_arr * (self.beta_max - self.beta_min)

    def f(self, x: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
        b = self.beta(t)
        return -0.5 * b * x

    def g(self, t: Union[float, np.ndarray]) -> np.ndarray:
        b = self.beta(t)
        return np.sqrt(b)

    def _I(self, t: Union[float, np.ndarray]) -> np.ndarray:
        # I(t) = ∫_0^t beta(s) ds = beta_min * t + 0.5*(beta_max - beta_min) * t^2
        t_arr = _as_array(t)
        return self.beta_min * t_arr + 0.5 * (self.beta_max - self.beta_min) * (t_arr ** 2)

    def marginal_mean_and_var(self, x0: np.ndarray, t: Union[float, np.ndarray], steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, var) of x_t given x0 under VPSDE forward process.
        Uses analytical mean and numerically robust variance via integral formula.
        """
        t_arr = _as_array(t)
        I_t = self._I(t_arr)
        mean = x0 * np.exp(-0.5 * I_t)

        # Variance: var = ∫_0^t exp(-∫_s^t beta(u) du) * beta(s) ds
        # Compute via numerical trapezoidal integration over s in [0, t]
        steps = self.steps if steps is None else max(1, int(steps))
        # Build a vectorized grid of s values for integration
        if np.isscalar(t_arr):
            s_vals = np.linspace(0.0, float(t_arr), steps + 1)
        else:
            # Broadcast: create a 1D grid along the last axis and rely on numpy broadcasting later
            s_vals = np.linspace(0.0, float(np.max(t_arr)), max(steps, 1) + 1)

        # We compute inner integral ∫_s^t beta(u) du using closed form and then exponentiate
        # inner(s) = beta_min*(t - s) + 0.5*(beta_max - beta_min)*(t^2 - s^2)
        beta_min = self.beta_min
        beta_max = self.beta_max
        t_scalar = float(np.max(t_arr)) if not np.isscalar(t_arr) else float(t_arr)
        s = s_vals
        inner = beta_min * (t_scalar - s) + 0.5 * (beta_max - beta_min) * (t_scalar**2 - s**2)
        beta_s = beta_min + s * (beta_max - beta_min)
        integrand = np.exp(-inner) * beta_s
        # Integrate along s with trapezoidal rule
        var = np.trapz(integrand, s)
        # If t was not scalar, var may be broadcastable to x0 shape; ensure compatibility
        var = np.asarray(var)
        return mean, var

    def sample_forward(self, x0: np.ndarray, t: Union[float, np.ndarray], steps: int = 50, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Euler–Maruyama forward sample from x0 to time t using VPSDE dynamics.
        This uses a fixed number of steps and a simple Euler discretization.
        """
        if rng is None:
            rng = np.random.default_rng()
        t_total = float(t) if np.isscalar(t) else float(np.max(t))
        x = x0.copy()
        dt = t_total / max(1, int(steps))
        for _ in range(max(1, int(steps))):
            current_t = dt * _  # approximate current time step index
            f_val = self.f(x, current_t)  # shape matches x
            g_val = self.g(current_t)
            dw = rng.normal(size=x.shape) * np.sqrt(dt)
            x = x + f_val * dt + g_val * dw
        return x


class VESDE:
    def __init__(self, sigma_min: float = 1e-4, sigma_max: float = 15.0):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        # precompute ratio and log for closed-forms
        self._ratio = self.sigma_max / self.sigma_min
        self._log_ratio = np.log(self._ratio) if self._ratio > 0 else 0.0

    def f(self, x: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
        # In VESDE, forward drift is 0
        return np.zeros_like(x)

    def g(self, t: Union[float, np.ndarray]) -> np.ndarray:
        t_arr = _as_array(t)
        # g(t) = sigma_min * (sigma_max/sigma_min)^t * sqrt(2 * log(sigma_max/sigma_min))
        factor = (self._ratio) ** t_arr
        return self.sigma_min * factor * np.sqrt(2.0 * self._log_ratio)

    def marginal_mean_and_var(self, x0: np.ndarray, t: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # Mean stays at x0 for zero drift; variance given by integral of g(s)^2 ds
        t_arr = _as_array(t)
        # var = ∫_0^t g(s)^2 ds with g(s)^2 = sigma_min^2 * (sigma_max/sigma_min)^{2s} * (2*log_ratio)
        log_ratio = self._log_ratio if self._log_ratio != 0 else 0.0
        ratio = self._ratio
        # When ratio <= 0, fallback to 0 variance
        if ratio <= 0:
            var = np.zeros_like(x0, dtype=float)
        else:
            # ∫_0^t sigma_min^2 * (ratio)^{2s} * (2 log_ratio) ds = sigma_min^2 * ( (ratio)^{2t} - 1 )
            var = (self.sigma_min ** 2) * (ratio ** (2.0 * t_arr) - 1.0)
        mean = x0.copy()
        return mean, var

    def sample_forward(self, x0: np.ndarray, t: Union[float, np.ndarray], steps: int = 50, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        t_total = float(t) if np.isscalar(t) else float(np.max(t))
        x = x0.copy()
        dt = t_total / max(1, int(steps))
        for _ in range(max(1, int(steps))):
            current_t = dt * _
            g_val = self.g(current_t)
            dw = rng.normal(size=x.shape) * np.sqrt(dt)
            x = x + g_val * dw  # f = 0
        return x


__all__ = ["VPSDE", "VESDE"]
