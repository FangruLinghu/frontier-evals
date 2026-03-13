# Code Implementation Summary
# - Lotka-Volterra data sampler extending the toy LV dynamics with a simple, test-friendly interface.
# - Provides sample_prior, simulate, and log_likelihood methods compatible with other samplers in the project.

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class LotkaVolterraSampler:
    """Simple Lotka-Volterra data sampler.

    Dynamics (2 species):
        dx1/dt = a * x1 - b * x1 * x2
        dx2/dt = -c * x2 + d * x1 * x2

    Theta parameters (dim_theta) map to (a, b, c, d) in order if provided.
    - If dim_theta < 4, missing parameters default to (a=1.0, b=0.1, c=1.0, d=0.1).
    - If dim_theta >= 4, the first four entries are used; extra entries are ignored.

    Observations x has dimension dim_x (default 2 for the two-species system).
    The forward sim is integrated with RK4 and a non-negative constraint on x.
    """

    def __init__(self, dim_theta: int, dim_x: int = 2, seed: int = 0,
                 sigma_x: Optional[np.ndarray] = None,
                 x0: Optional[np.ndarray] = None,
                 n_components: int = 1,
                 weights: Optional[np.ndarray] = None,
                 Sigma_x: Optional[np.ndarray] = None):
        if dim_x != 2:
            raise ValueError("Lotka-Volterra sampler currently supports dim_x == 2 (two species).")
        self.dim_theta = int(dim_theta)
        self.dim_x = int(dim_x)
        self.rng = np.random.default_rng(seed)

        # initial state for simulations
        self.x0 = np.asarray(x0, dtype=float) if x0 is not None else np.array([1.0, 0.5], dtype=float)
        self.x0 = self.x0.reshape(self.dim_x)

        # noise level for log-likelihood (diagonal covariance handling)
        if sigma_x is None:
            self.sigma_x = np.array([0.1 for _ in range(self.dim_x)], dtype=float)
        else:
            self.sigma_x = np.asarray(sigma_x, dtype=float).reshape(self.dim_x)

        # Options for mixture components (not essential for a single-component LV, but kept for API parity)
        self.n_components = max(1, int(n_components))
        if self.n_components > 1:
            if weights is None:
                self.weights = np.ones(self.n_components, dtype=float) / self.n_components
            else:
                w = np.asarray(weights, dtype=float).reshape(-1)
                if w.size != self.n_components:
                    raise ValueError("weights length must equal n_components")
                self.weights = w / np.sum(w)
            # per-component covariances (default to diag(0.1))
            if Sigma_x is None:
                self.Sigmas = [np.diag(self.sigma_x) * 1.0 for _ in range(self.n_components)]
            else:
                sig = np.asarray(Sigma_x, dtype=float)
                if sig.ndim == 3:
                    if sig.shape[0] != self.n_components or sig.shape[1] != self.dim_x or sig.shape[2] != self.dim_x:
                        raise ValueError("Sigma_x with three dims must be (n_components, dim_x, dim_x)")
                    self.Sigmas = [sig[k] for k in range(self.n_components)]
                elif sig.ndim == 2 and sig.shape == (self.dim_x, self.dim_x):
                    self.Sigmas = [sig for _ in range(self.n_components)]
                else:
                    raise ValueError("Sigma_x must be shape (dim_x, dim_x) or (n_components, dim_x, dim_x)")
        else:
            self.weights = np.array([1.0], dtype=float)
            if Sigma_x is None:
                self.Sigmas = [np.diag(self.sigma_x)]
            else:
                cov = np.asarray(Sigma_x, dtype=float)
                if cov.shape != (self.dim_x, self.dim_x):
                    raise ValueError("Sigma_x must be shape (dim_x, dim_x) for single-component LV sampler")
                self.Sigmas = [cov]

        # Clip or scale theta dimension if shorter than 4
        self._min_params = [1.0, 0.1, 1.0, 0.1]  # defaults for (a,b,c,d)

    def _rhs(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # Map theta to LV parameters a,b,c,d with reasonable defaults
        a = float(theta[0]) if theta.size > 0 else self._min_params[0]
        b = float(theta[1]) if theta.size > 1 else self._min_params[1]
        c = float(theta[2]) if theta.size > 2 else self._min_params[2]
        d = float(theta[3]) if theta.size > 3 else self._min_params[3]

        x1, x2 = x[0], x[1]
        dx1 = a * x1 - b * x1 * x2
        dx2 = -c * x2 + d * x1 * x2
        return np.array([dx1, dx2], dtype=float)

    def _rk4_step(self, x: np.ndarray, theta: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._rhs(x, theta)
        k2 = self._rhs(x + 0.5 * dt * k1, theta)
        k3 = self._rhs(x + 0.5 * dt * k2, theta)
        k4 = self._rhs(x + dt * k3, theta)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # populations cannot be negative
        x_next = np.maximum(x_next, 0.0)
        return x_next

    def simulate_theta(self, theta: np.ndarray, t_end: float = 1.0, dt: float = 0.01) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        x = self.x0.copy()
        steps = max(1, int(round(t_end / dt)))
        for _ in range(steps):
            x = self._rk4_step(x, theta, dt)
        return x

    def simulate(self, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> np.ndarray:
        """Return x at final time given theta.
        If theta is a batch, returns an array of shape (n, dim_x).
        If theta is None, draws a random theta from the prior for a single sample.
        """
        if theta is None:
            theta = self.sample_prior(1)[0]
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)

        if theta_arr.ndim != 1:
            theta_arr = theta_arr.ravel()
        x = self.x0.copy()
        t_end = 1.0
        dt = 0.01
        steps = max(1, int(round(t_end / dt)))
        for _ in range(steps):
            x = self._rk4_step(x, theta_arr, dt)
        if component is not None:
            # for compatibility with mixture API, simply return same x; component affects nothing here
            pass
        return x

    def sample_prior(self, n: int = 1) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.empty((0, self.dim_theta))
        return self.rng.normal(loc=0.0, scale=1.0, size=(n, self.dim_theta))

    def log_likelihood(self, x: np.ndarray, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> float:
        # Compute log p(x | theta) under a simple Gaussian noise model around the LV-predicted mean.
        x = np.asarray(x, dtype=float).reshape(-1)
        if theta is None:
            theta = self.sample_prior(1)[0]
        theta = np.asarray(theta, dtype=float).reshape(-1)
        mean = self.simulate(theta)
        # Choose covariance per component if provided; otherwise use the global diag covariance
        if self.n_components > 1 and component is not None:
            comp = int(component)
            cov = self.Sigmas[comp] if comp < len(self.Sigmas) else self.Sigmas[0]
        else:
            cov = self.Sigmas[0]
        # Use diagonal Cov for numerical stability
        if cov.ndim == 2 and cov.shape == (self.dim_x, self.dim_x):
            diag = np.diag(cov)
        else:
            diag = np.ones(self.dim_x) * 0.1
        diff = x - mean
        # handle shapes
        if diff.size != self.dim_x:
            # try broadcasting mean to match x size
            diff = (x - mean[: x.size])[: self.dim_x]
        var = diag
        # Avoid zero-variance
        var = np.where(var <= 0.0, 1e-6, var)
        logpdf = -0.5 * np.sum((diff ** 2) / var) - 0.5 * np.sum(np.log(2.0 * np.pi * var))
        return float(logpdf)

    def get_mixture_weights(self) -> np.ndarray:
        return self.weights.copy()


__all__ = ["LotkaVolterraSampler"]
