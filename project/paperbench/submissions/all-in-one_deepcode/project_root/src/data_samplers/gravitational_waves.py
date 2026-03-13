# Code Implementation Summary
# gravitational_waves.py

"""
Gravitational Waves data sampler (high-dimensional observations).

This module provides a lightweight, test-friendly data sampler that maps a
latent parameter theta to high-dimensional observations x via a linear-Gaussian
mapping for each mixture component. It follows the standard data_samplers API
used elsewhere in the repository:

- sample_prior(n=1) -> theta samples from N(0, I_dim_theta)
- simulate(theta=None, component=None) -> x samples from p(x | theta)
- log_likelihood(x, theta=None, component=None) -> float or np.ndarray
- get_mixture_weights() -> np.ndarray of shape (n_components,)

The implementation is deterministic (seeded RNG) and designed for unit tests and
quick demonstrations rather than full scientific fidelity.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List

__all__ = ["GravitationalWavesSampler"]


class GravitationalWavesSampler:
    """Gravitational Waves data sampler with mixture components.

    Each component k defines a linear mapping A_k (dim_x x dim_theta) and a
    Gaussian observation noise with covariance Sigma_x_k (dim_x x dim_x).
    Observation dimension dim_x is typically high (e.g., 128).
    """

    def __init__(
        self,
        dim_theta: int,
        dim_x: int = 128,
        seed: int = 0,
        n_components: int = 1,
        A: Optional[List[np.ndarray]] = None,
        Sigma_x: Optional[List[np.ndarray]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.dim_theta = int(dim_theta)
        self.dim_x = int(dim_x)
        self.n_components = int(n_components)
        self._rng = np.random.default_rng(seed)

        # Initialize per-component linear maps A_k: (dim_x, dim_theta)
        if A is None:
            self.As = [self._rng.normal(size=(self.dim_x, self.dim_theta)) for _ in range(self.n_components)]
        else:
            assert len(A) == self.n_components
            self.As = [np.asarray(a) for a in A]

        # Initialize per-component covariances Sigma_x_k: (dim_x, dim_x)
        if Sigma_x is None:
            self.Sigmas = [np.diag(self._rng.uniform(0.05, 0.5, size=self.dim_x)) for _ in range(self.n_components)]
        else:
            assert len(Sigma_x) == self.n_components
            self.Sigmas = [np.asarray(s) for s in Sigma_x]

        # Mixture weights
        if weights is None:
            self.weights = np.ones(self.n_components, dtype=float) / max(1, self.n_components)
        else:
            w = np.asarray(weights, dtype=float)
            if w.ndim != 1 or w.shape[0] != self.n_components:
                raise ValueError("weights must have shape (n_components,)")
            self.weights = w / np.sum(w)

    def sample_prior(self, n: int = 1) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.empty((0, self.dim_theta))
        return self._rng.normal(size=(n, self.dim_theta)) if n > 1 else self._rng.normal(size=(self.dim_theta,))

    def simulate(self, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> np.ndarray:
        # theta can be shape (dim_theta,) or (n, dim_theta) for batched sampling
        if theta is None:
            theta = self.sample_prior(n=1)
        theta = np.asarray(theta)

        if theta.ndim == 1:
            # Single sample
            comp = int(component) if component is not None else int(self._rng.choice(self.n_components, p=self.weights))
            mean = self.As[comp] @ theta  # (dim_x,)
            x = self._rng.multivariate_normal(mean, self.Sigmas[comp])
            return x
        elif theta.ndim == 2:
            n = theta.shape[0]
            out = np.zeros((n, self.dim_x))
            for i in range(n):
                comp = int(component) if component is not None else int(self._rng.choice(self.n_components, p=self.weights))
                mean = self.As[comp] @ theta[i]
                out[i] = self._rng.multivariate_normal(mean, self.Sigmas[comp])
            return out
        else:
            raise ValueError("theta must be 1D or 2D array")

    def log_likelihood(self, x: np.ndarray, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> float:
        x = np.asarray(x)
        if theta is None:
            theta = np.zeros(self.dim_theta)
        theta = np.asarray(theta)
        # Single sample case
        def _log_pdf_for_comp(k: int, th: np.ndarray, xv: np.ndarray) -> float:
            mean = self.As[k] @ th
            return _logpdf_gaussian(xv, mean, self.Sigmas[k])

        if theta.ndim == 1:
            if component is not None:
                comp = int(component)
                return self.weights[comp] * np.exp(_log_pdf_for_comp(comp, theta, x))
            # Mixture across components: log-sum-exp of weight_k + log_pdf_k
            logs = []
            for k in range(self.n_components):
                logp = np.log(self.weights[k] + 1e-20) + _log_pdf_gaussian(x, self.As[k] @ theta, self.Sigmas[k])
                logs.append(logp)
            # logsumexp
            a = max(logs)
            return a + np.log(np.sum(np.exp(np.array(logs) - a)))
        elif theta.ndim == 2:
            n = theta.shape[0]
            if x.ndim != 1 or x.shape[0] != self.dim_x:
                raise ValueError("x must be a 1D vector when theta is batched")
            res = np.zeros(n, dtype=float)
            for i in range(n):
                if component is not None:
                    comp = int(component)
                    res[i] = _log_pdf_gaussian(x, self.As[comp] @ theta[i], self.Sigmas[comp])
                    res[i] += np.log(self.weights[comp] + 1e-20)
                else:
                    logs = []
                    for k in range(self.n_components):
                        logp = np.log(self.weights[k] + 1e-20) + _log_pdf_gaussian(x, self.As[k] @ theta[i], self.Sigmas[k])
                        logs.append(logp)
                    a = max(logs)
                    res[i] = a + np.log(np.sum(np.exp(np.array(logs) - a)))
            return res
        else:
            raise ValueError("theta must be 1D or 2D array")

    def get_mixture_weights(self) -> np.ndarray:
        return self.weights


def _logpdf_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute log pdf of x ~ N(mean, cov).

    Supports 1D x or 1D mean with a full covariance matrix cov.
    Uses Cholesky decomposition for numerical stability.
    Returns a scalar logpdf when x is 1D, or an array of logpdfs if x is 2D.
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    diff = x - mean
    # Ensure shapes
    if diff.ndim == 0:
        diff = diff.reshape(-1)
        x = x.reshape(-1)
        mean = mean.reshape(-1)
    if cov.ndim != 2:
        raise ValueError("cov must be 2D matrix")
    d = mean.shape[0]
    # Cholesky factorization
    try:
        L = np.linalg.cholesky(cov)
        solve = np.linalg.solve(L, diff)
        maha = float(np.dot(solve, solve))
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        logpdf = -0.5 * (maha + logdet + d * np.log(2 * np.pi))
        return logpdf
    except np.linalg.LinAlgError:
        # Fallback: use general inversion (less stable but robust for tests)
        inv_cov = np.linalg.pinv(cov)
        maha = float(diff.T @ inv_cov @ diff)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # fallback to diagonal approximation if cov is not SPD
            diag_cov = np.diag(np.diag(cov))
            inv_diag = np.diag(1.0 / np.diag(diag_cov))
            maha = float(diff.T @ inv_diag @ diff)
            logdet = float(np.sum(np.log(np.diag(diag_cov))))
        logpdf = -0.5 * (maha + logdet + d * np.log(2 * np.pi))
        return logpdf
