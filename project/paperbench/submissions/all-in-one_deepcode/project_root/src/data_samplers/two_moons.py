# Code Implementation Summary
# Implements a lightweight Two Moons data sampler for joint theta (parameters) and x (observations).

import numpy as np
from typing import Optional, Tuple

__all__ = ["TwoMoonsSampler"]


class TwoMoonsSampler:
    """Two Moons data sampler.

    This is a minimal, deterministic toy sampler intended for unit tests and
    small-scale demonstrations. It provides a simple API inspired by the
    GaussianLinear and GaussianMixture samplers used in this repository:
    - sample_prior(n=1) -> theta samples from N(0, I_dim_theta)
    - simulate(theta=None, component=None) -> x samples given theta
    - log_likelihood(x, theta, component=None) -> log p(x | theta)

    Notes:
    - The mapping from theta to x is deterministic (no external randomness in
      simulate when theta is provided). A small Gaussian observation noise is
      added to x for realism when computing simulate results.
    - dim_theta is 2 by default; dim_x is 2 to match the classic two-moons
      geometry. A simple linear transformation A is learned (initialized) to
      inject theta-dependence into x.
    """

    def __init__(self, dim_theta: int = 2, dim_x: int = 2, seed: int = 0,
                 A: Optional[np.ndarray] = None, noise_std: float = 0.05):
        self.dim_theta = int(dim_theta)
        self.dim_x = int(dim_x)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        # Transformation from theta to x: A @ theta
        if A is None:
            # seed-dependent small random projection
            self.A = self._rng.normal(scale=0.5, size=(self.dim_x, self.dim_theta))
        else:
            self.A = np.asarray(A).reshape(self.dim_x, self.dim_theta)
        # Observation noise standard deviation
        self.noise_std = float(noise_std)
        # Pre-warm a fixed base offset to ensure theta can influence x differently
        self._base_offset = self._rng.uniform(-0.5, 0.5, size=(self.dim_x,))

    # ------------------- Public API -------------------
    def sample_prior(self, n: int = 1) -> np.ndarray:
        """Sample theta ~ N(0, I) with shape (n, dim_theta) or (dim_theta,) if n==1."""
        n = int(n)
        theta = self._rng.normal(size=(n, self.dim_theta))
        if n == 1:
            return theta[0]
        return theta

    def simulate(self, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> np.ndarray:
        """Sample x given theta.

        - If theta is None, sample from the prior.
        - If a single theta is provided (shape (dim_theta,)), return a vector (dim_x,).
        - If a batch of thetas is provided (shape (n, dim_theta)), return (n, dim_x).
        - component is accepted for API compatibility but ignored in this simple sampler.
        """
        if theta is None:
            theta_arr = self.sample_prior(n=1)
            theta_arr = np.atleast_2d(theta_arr)
        else:
            theta_arr = np.asarray(theta)
            if theta_arr.ndim == 1:
                theta_arr = theta_arr.reshape(1, -1)
        # Ensure shapes
        if theta_arr.shape[1] != self.dim_theta:
            raise ValueError(f"theta must have size {self.dim_theta} (got {theta_arr.shape[1]}).")

        xs = []
        for th in theta_arr:
            # Deterministic base mapping from theta to a 2D point
            base = np.array([np.cos(th[0]), np.sin(th[0])])
            # Inject theta-dependence with a simple affine map
            x_from_theta = self.A @ th
            x = base + x_from_theta + self._base_offset
            # Add a small fixed observation noise to emulate x
            x = x + self._rng.normal(scale=self.noise_std, size=self.dim_x)
            xs.append(x)
        xs = np.stack(xs, axis=0)
        if xs.shape[0] == 1:
            return xs[0]
        return xs

    def log_likelihood(self, x: np.ndarray, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> float:
        """Compute log p(x | theta) under a simple Gaussian observation model.

        We model x as x ~ N(mean(theta), cov = sigma^2 I) with sigma = noise_std.
        The mean is computed via the same deterministic mapping used in simulate.
        If theta is None, uses the prior mean by sampling a single theta.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.dim_x:
            raise ValueError(f"x must have size {self.dim_x} (got {x.shape[1]}).")
        if theta is None:
            theta = self.sample_prior(n=1)
        th = np.asarray(theta).reshape(-1)
        if th.size != self.dim_theta:
            raise ValueError(f"theta must have size {self.dim_theta}.")
        # Deterministic mean for this theta
        mean = self.simulate(theta=th)
        # Ensure shapes
        mean = np.asarray(mean).reshape(1, -1)
        diff = x - mean
        sigma2 = max(self.noise_std**2, 1e-12)
        # Compute logpdf for each row
        d = self.dim_x
        logps = -0.5 * (d * np.log(2 * np.pi) + d * np.log(sigma2) + (diff ** 2).sum(axis=1) / sigma2)
        # Return average log-likelihood if multiple samples
        return float(logps.mean())

    def get_mixture_weights(self) -> np.ndarray:
        """Return a placeholder weights array for API compatibility.
        This sampler is not a mixture model; return a single component weight."""
        return np.array([1.0])


# End of module
