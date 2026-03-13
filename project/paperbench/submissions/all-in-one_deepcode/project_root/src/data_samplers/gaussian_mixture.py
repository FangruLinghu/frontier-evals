# Code Implementation: Gaussian Mixture Data Sampler
"""project_root.src.data_samplers.gausssian_mixture

A lightweight Gaussian Mixture data sampler compatible with the existing
GaussianLinearSampler interface pattern used in the repository. This class
offers a simple mixture of multivariate Gaussian conditionals for x given theta.

Interface:
- GaussianMixtureSampler(dim_theta, dim_x, n_components, seed=0, A=None, Sigma_x=None, weights=None)
- sample_prior(n=1) -> np.ndarray: draws n samples of theta from N(0, I)
- simulate(theta=None, component=None) -> np.ndarray: samples x given theta and an
  (optional) mixture component. If theta is not provided, samples from the prior.
  If component is provided, uses that component; otherwise samples according to weights.
- log_likelihood(x, theta, component=None) -> float or np.ndarray:
  log p(x | theta) under the mixture (or per-component if provided).
- get_mixture_weights() -> np.ndarray: returns the component weights.

Notes:
- This implementation focuses on determinism and compactness suitable for unit tests
  and small-scale demonstrations. It does not attempt to cover all numerical edge cases
  of real SBI training pipelines.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

__all__ = ["GaussianMixtureSampler"]


def _logpdf_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute log PDF of x ~ N(mean, cov).

    Supports x as a 1-D vector with shape (d,) or a 2-D matrix with shape (n, d).
    Returns a scalar for 1-D input or a 1-D array of length n for 2-D input.
    Uses Cholesky decomposition for numerical stability.
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    d = mean.shape[0]
    if x.ndim == 1:
        diff = x - mean
        L = np.linalg.cholesky(cov)
        solve = np.linalg.solve(L, diff)
        quad = float(np.dot(solve, solve))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_pdf = -0.5 * (d * np.log(2 * np.pi) + log_det + quad)
        return log_pdf
    elif x.ndim == 2:
        diff = x - mean  # broadcast if necessary
        L = np.linalg.cholesky(cov)
        solve = np.linalg.solve(L, diff.T)  # shape (d, n)
        quad = np.sum(solve * solve, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_pdf = -0.5 * (d * np.log(2 * np.pi) + log_det + quad)
        return log_pdf
    else:
        raise ValueError("x must be 1D or 2D array.")


class GaussianMixtureSampler:
    """Mixture of Gaussian conditionals for x given theta.

    x | theta ~ sum_k w_k * N(A_k * theta, Sigma_x_k)
    theta ~ N(0, I_dim_theta) by default.
    """

    def __init__(
        self,
        dim_theta: int,
        dim_x: int,
        n_components: int,
        seed: int = 0,
        A: Optional[np.ndarray] = None,
        Sigma_x: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.dim_theta = int(dim_theta)
        self.dim_x = int(dim_x)
        self.n_components = int(n_components)
        self._rng = np.random.default_rng(seed)

        # Per-component linear mappings A_k: (dim_x, dim_theta)
        if A is None:
            self.As = [self._rng.normal(scale=1.0, size=(self.dim_x, self.dim_theta))
                       for _ in range(self.n_components)]
        else:
            A = np.asarray(A)
            if A.shape != (self.n_components, self.dim_x, self.dim_theta):
                raise ValueError("A must have shape (n_components, dim_x, dim_theta)")
            self.As = [A[k] for k in range(self.n_components)]

        # Per-component observation covariances Sigma_x_k: (dim_x, dim_x)
        if Sigma_x is None:
            self.Sigmas = [np.eye(self.dim_x) for _ in range(self.n_components)]
        else:
            Sigma_x = np.asarray(Sigma_x)
            if Sigma_x.shape != (self.n_components, self.dim_x, self.dim_x):
                raise ValueError("Sigma_x must have shape (n_components, dim_x, dim_x)")
            self.Sigmas = [Sigma_x[k] for k in range(self.n_components)]

        # Mixture weights
        if weights is None:
            w = np.ones(self.n_components) / float(self.n_components)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (self.n_components,):
                raise ValueError("weights must have shape (n_components,)")
            w = w / np.sum(w)
        self.weights = w

    # ---- Public API ----
    def sample_prior(self, n: int = 1) -> np.ndarray:
        """Sample theta from the prior N(0, I_dim_theta).

        Returns array of shape (n, dim_theta) or (dim_theta,) for n==1.
        """
        n = int(n)
        if n <= 0:
            return np.zeros((0, self.dim_theta))
        samples = self._rng.normal(size=(n, self.dim_theta))
        if n == 1:
            return samples[0]
        return samples

    def simulate(self, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> np.ndarray:
        """Sample x given theta and an optional component.

        - If theta is None, samples a single theta from prior.
        - If component is None, samples a component index according to weights.
        - Returns x of shape (dim_x,) for a single sample or (n, dim_x) if theta is (n, dim_theta).
        """
        # If theta is None, generate a single theta
        if theta is None:
            theta = self.sample_prior(1)
        theta_arr = np.asarray(theta)

        # Helper to sample a single x given theta and component idx
        def sample_one(theta_vec: np.ndarray, k: int) -> np.ndarray:
            A_k = self.As[k]
            Sigma_k = self.Sigmas[k]
            mean = A_k @ theta_vec  # shape (dim_x,)
            cov = Sigma_k
            # Sample x ~ N(mean, cov)
            x = self._rng.multivariate_normal(mean, cov)
            return x

        if theta_arr.ndim == 1:
            # Single sample
            k = component
            if k is None:
                k = int(self._rng.choice(self.n_components, p=self.weights))
            x = sample_one(theta_arr, k)
            return x
        elif theta_arr.ndim == 2:
            # Batch sampling: theta shape (n, dim_theta)
            n = theta_arr.shape[0]
            if component is not None:
                comps = np.full(n, int(component), dtype=int)
            else:
                comps = self._rng.choice(self.n_components, size=n, p=self.weights)
            xs = []
            for i in range(n):
                xs.append(sample_one(theta_arr[i], int(comps[i])))
            return np.stack(xs, axis=0)
        else:
            raise ValueError("theta must be 1D or 2D array with shape (dim_theta,) or (n, dim_theta)")

    def log_likelihood(self, x: np.ndarray, theta: np.ndarray, component: Optional[int] = None) -> float:
        """Compute log p(x | theta) under the mixture model.

        If component is provided, uses that component's distribution only.
        If component is None, computes log-sum-exp over components.
        """
        x = np.asarray(x)
        theta = np.asarray(theta)
        if x.shape[0] != self.dim_x:
            raise ValueError("x has incompatible dimension with dim_x")
        if theta.shape[-1] != self.dim_theta:
            raise ValueError("theta has incompatible dimension with dim_theta")

        if component is not None:
            k = int(component)
            mean = self.As[k] @ theta
            cov = self.Sigmas[k]
            return float(_logpdf_gaussian(x, mean, cov))
        else:
            # log-sum-exp over components
            log_weights = np.log(self.weights + 1e-20)
            log_p = []  # list of log pdfs per component for the given theta
            for k in range(self.n_components):
                mean = self.As[k] @ theta
                cov = self.Sigmas[k]
                logp = _logpdf_gaussian(x, mean, cov)
                log_p.append(logp + log_weights[k])
            log_p = np.asarray(log_p)
            # log-sum-exp across components
            max_logp = np.max(log_p)
            if np.isfinite(max_logp):
                return float(max_logp + np.log(np.sum(np.exp(log_p - max_logp))))
            else:
                return float(-np.inf)

    def get_mixture_weights(self) -> np.ndarray:
        return self.weights
