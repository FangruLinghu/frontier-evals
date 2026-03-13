# Code Implementation Summary
# - SIR data sampler with RK4 integration for unit tests and demonstrations.
# - Maps a theta vector to epidemiological parameters (beta, gamma, and optional N).
# - Supports a simple mixture-component extension to align with the project API.

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class SirSampler:
    """SIR data sampler with RK4 integration; supports sampling priors, simulating trajectories, and computing log-likelihoods.

    The state vector x is [S, I, R] representing the population fractions.
    The dynamics (normalized by total population N) are:
      dS/dt = -beta * S * I / N
      dI/dt = beta * S * I / N - gamma * I
      dR/dt = gamma * I
    Theta maps to (beta, gamma) via a softplus transform to ensure positivity, and an
    optional third theta element controls the total population N (default N=1.0).

    This implementation aims to be lightweight and deterministic for unit tests.
    """

    def __init__(
        self,
        dim_theta: int,
        dim_x: int = 3,
        seed: int = 0,
        sigma_x: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
        n_components: int = 1,
        weights: Optional[np.ndarray] = None,
        Sigma_x: Optional[np.ndarray] = None,
    ) -> None:
        assert dim_x == 3, "Current SIR implementation expects dim_x to be 3 (S, I, R)"
        self.dim_theta = int(dim_theta)
        self.dim_x = int(dim_x)
        self.rng = np.random.default_rng(seed)

        self.n_components = max(1, int(n_components))
        # Initialize per-component linear maps A_k: x = A_k @ theta (shape: dim_x x dim_theta)
        self.As = self.rng.normal(scale=0.5, size=(self.n_components, self.dim_x, self.dim_theta))

        # Initialize per-component covariances Sigma_x_k for x|theta
        if Sigma_x is not None:
            Sigma_arr = np.asarray(Sigma_x)
            if Sigma_arr.ndim == 2:
                Sigma_arr = np.stack([Sigma_arr] * self.n_components, axis=0)
            assert Sigma_arr.shape[-2:] == (self.dim_x, self.dim_x)
            self.Sigmas = Sigma_arr
        else:
            self.Sigmas = np.stack([np.eye(self.dim_x) * 0.01 for _ in range(self.n_components)], axis=0)

        # Mixture weights
        if weights is not None:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != self.n_components:
                raise ValueError("weights length must match n_components")
            self.weights = w / np.sum(w)
        else:
            self.weights = np.ones(self.n_components, dtype=float) / float(self.n_components)

        # Optional initial state (not strictly required for the sampler, but useful for defaults)
        if x0 is not None:
            self.x0_default = np.asarray(x0, dtype=float)
        else:
            # Default to near-complete susceptible population
            self.x0_default = np.array([0.99, 0.01, 0.0], dtype=float)

        self._ensure_valid_inputs()

    # ---------------------------------------------------------------------
    # Public API
    def sample_prior(self, n: int = 1) -> np.ndarray:
        """Samples theta from a standard Normal prior.
        Returns shape (n, dim_theta) or (dim_theta,) when n == 1.
        """
        theta = self.rng.normal(loc=0.0, scale=1.0, size=(n, self.dim_theta))
        if n == 1:
            return theta[0]
        return theta

    def simulate(self, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> np.ndarray:
        """Simulate x given theta and (optional) component.
        If theta is None, a sample is drawn from the prior.
        If component is provided, uses that component's mapping; otherwise samples from the mixture.
        Returns x with shape (dim_x,).
        """
        if theta is None:
            theta = self.sample_prior(n=1)[0]
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.dim_theta:
            raise ValueError("theta has incompatible dimension")

        # Select component
        if component is None:
            comp = self.rng.choice(self.n_components, p=self.weights)
        else:
            comp = int(component)
            if comp < 0 or comp >= self.n_components:
                raise ValueError("component index out of range")

        # Map theta to parameters
        beta, gamma, N = self._theta_to_params(theta)
        # Build mean x = A_k @ theta
        mean = self.As[comp] @ theta  # shape (dim_x,)
        cov = self.Sigmas[comp]
        x = self.rng.multivariate_normal(mean=mean, cov=cov)
        # Basic physical constraints: non-negative components, conservation S+I+R = N
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 0.0)
        s = np.sum(x)
        if s <= 0 or np.isclose(s, 0.0):
            x = self.x0_default.copy()
        else:
            if s != N:
                x = x * (N / s)
        return x

    def log_likelihood(self, x: np.ndarray, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> float:
        """Compute log p(x | theta) under the mixture model.
        If component is specified, uses that component's distribution only.
        If component is None, returns log-sum-exp over components.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if theta is None:
            # If no theta provided, use a random theta from prior for likelihood evaluation
            theta = self.sample_prior(n=1)[0]
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.dim_theta:
            raise ValueError("theta has incompatible dimension for log_likelihood")

        def log_p_for_component(k: int) -> float:
            mean = self.As[k] @ theta
            cov = self.Sigmas[k]
            return _logpdf_gaussian(x, mean, cov) + np.log(self.weights[k] + 1e-12)

        if component is not None:
            k = int(component)
            if k < 0 or k >= self.n_components:
                raise ValueError("component index out of range")
            mean = self.As[k] @ theta
            cov = self.Sigmas[k]
            return _logpdf_gaussian(x, mean, cov) + np.log(self.weights[k] + 1e-12)
        else:
            logs = [log_p_for_component(k) for k in range(self.n_components)]
            # log-sum-exp for numerical stability
            a = max(logs)
            ssum = sum(np.exp(l - a) for l in logs)
            return a + np.log(ssum + 1e-12)

    def get_mixture_weights(self) -> np.ndarray:
        return self.weights.copy()

    # ---------------------------------------------------------------------
    # Internal helpers
    def _theta_to_params(self, theta: np.ndarray) -> Tuple[float, float, float]:
        # Map theta to positive beta, gamma using softplus transforms; optional N from extra theta
        t = theta.reshape(-1)
        beta = self._softplus(t[0]) if t.size > 0 else 0.5
        gamma = self._softplus(t[1]) if t.size > 1 else 0.2
        N = 1.0
        if t.size > 2:
            N = max(0.1, 1.0 + float(abs(t[2])))
        return float(beta), float(gamma), float(N)

    @staticmethod
    def _softplus(x: float) -> float:
        return float(np.log1p(np.exp(float(x))))

    def _ensure_valid_inputs(self) -> None:
        if self.dim_theta <= 0:
            raise ValueError("dim_theta must be positive")
        if self.n_components <= 0:
            raise ValueError("n_components must be at least 1")


# Helper: a small Gaussian logpdf implementation supporting 1D or 2D x
def _logpdf_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    d = mean.size

    diff = x - mean
    try:
        L = np.linalg.cholesky(cov)
        # solve L * y = diff
        y = np.linalg.solve(L, diff)
        quad = float(np.dot(y, y))
        logdet = 2.0 * float(np.sum(np.log(np.diagonal(L))))
        logpdf = -0.5 * (quad + logdet + d * np.log(2.0 * np.pi))
        return logpdf
    except np.linalg.LinAlgError:
        # fallback to full inversion
        inv = np.linalg.inv(cov)
        quad = float(diff.T @ inv @ diff)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = np.log(np.abs(np.linalg.det(cov)) + 1e-12)
        logpdf = -0.5 * (quad + logdet + d * np.log(2.0 * np.pi))
        return float(logpdf)


__all__ = ["SirSampler"]
