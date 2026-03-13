# Code Implementation: SLCP data sampler
"""
Simple SLCP data sampler with a linear-Gaussian mapping from theta to x.
This provides a stable, test-friendly implementation compatible with the
project's data_samplers API: sample_prior, simulate, log_likelihood, and
get_mixture_weights.

Notes:
- Supports multiple mixture components. Each component k defines A_k (dim_x x dim_theta)
  and Sigma_x_k (dim_x x dim_x).
- If weights are not provided, a uniform mixture is assumed.
- Theta prior is standard normal N(0, I).
- Simulation draws x ~ N(A_k theta, Sigma_x_k).
- Log-likelihood supports either a specific component or all components via log-sum-exp.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List


__all__ = ["SlcpSampler"]


def _logpdf_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute log pdf of x ~ N(mean, cov).

    Supports x being 1-D (returns scalar) or 2-D (returns 1-D array of length n).
    Uses Cholesky decomposition for numerical stability when cov is positive definite.
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    d = mean.shape[0]
    if x.ndim == 1:
        diff = x - mean
        try:
            L = np.linalg.cholesky(cov)
            solve = np.linalg.solve(L, diff)
            quad = np.dot(solve, solve)
            logdet = 2.0 * np.log(np.diag(L)).sum()
        except np.linalg.LinAlgError:
            # Fallback to general purpose solver if cov is singular
            inv_cov = np.linalg.pinv(cov)
            quad = diff.T @ inv_cov @ diff
            sign, logdet = np.linalg.slogdet(cov)
            logdet = -logdet  # slogdet returns log(det), we keep consistent sign
        log_pdf = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
        return float(log_pdf)
    else:
        # x is (n, d)
        diffs = x - mean
        try:
            L = np.linalg.cholesky(cov)
            # Solve L z = diffs^T for each row
            z = np.linalg.solve(L, diffs.T).T  # shape (n, d)
            quad = np.sum(z * z, axis=1)
            logdet = 2.0 * np.log(np.diag(L)).sum()
            log_pdf = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
            quad = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
            sign, logdet = np.linalg.slogdet(cov)
            log_pdf = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
        return log_pdf


class SlcpSampler:
    """Structured Linear-Constraint Process (SLCP) data sampler.

    This sampler maps theta to x via x = A_k @ theta + noise, with a
    mixture over k components. It follows the common API found in the repo:
    sample_prior, simulate, log_likelihood, get_mixture_weights.
    """

    def __init__(
        self,
        dim_theta: int,
        dim_x: int = 3,
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

        # Initialize per-component A_k and Sigma_x_k
        if A is None:
            self.A = [self._rng.normal(size=(self.dim_x, self.dim_theta)) for _ in range(self.n_components)]
        else:
            assert len(A) == self.n_components
            self.A = [np.asarray(a) for a in A]
        if Sigma_x is None:
            self.Sigma_x = [np.diag(self._rng.uniform(0.05, 0.5, size=self.dim_x)) for _ in range(self.n_components)]
        else:
            assert len(Sigma_x) == self.n_components
            self.Sigma_x = [np.asarray(s) for s in Sigma_x]

        if weights is None:
            w = np.ones(self.n_components) / float(self.n_components)
        else:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
        self.weights = w

    def sample_prior(self, n: int = 1) -> np.ndarray:
        """Sample theta ~ N(0, I_dim_theta).
        Returns shape (n, dim_theta) or (dim_theta,) if n == 1.
        """
        n = int(n)
        thetas = self._rng.normal(size=(n, self.dim_theta)) if n > 1 else self._rng.normal(size=(self.dim_theta,))
        return thetas

    def simulate(self, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> np.ndarray:
        """Sample x given theta and optional component.
        - theta can be 1D (dim_theta,) or 2D (n, dim_theta) for batched samples.
        - If component is None, sample according to weights.
        Returns x with shape (dim_x,) or (n, dim_x).
        """
        if theta is None:
            theta = self.sample_prior(n=1)
        theta_arr = np.asarray(theta)
        batched = theta_arr.ndim == 2

        k = component if component is not None else self._rng.choice(self.n_components, size=(theta_arr.shape[0] if batched else 1), p=self.weights)
        if not batched:
            k = int(k)
            mean = self.A[k] @ theta_arr
            x = self._rng.multivariate_normal(mean, self.Sigma_x[k])
            return x
        else:
            means = np.zeros((theta_arr.shape[0], self.dim_x))
            covs = [self.Sigma_x[kk] for kk in k]
            xs = []
            for i, th in enumerate(theta_arr):
                kk = int(k[i])
                means[i] = self.A[kk] @ th
                xs.append(self._rng.multivariate_normal(means[i], covs[i]))
            return np.asarray(xs)

    def log_likelihood(self, x: np.ndarray, theta: Optional[np.ndarray] = None, component: Optional[int] = None) -> float:
        """Compute log p(x | theta) under the mixture (or a single component).
        If component is None, computes log-sum-exp across components.
        """
        x = np.asarray(x)
        if theta is None:
            # If no theta provided, assume theta is latent with prior; for our purposes,
            # we compute p(x) marginal over theta by averaging log-likelihoods across theta samples
            # drawn from prior; here we approximate with a single theta draw to keep API simple.
            theta = self.sample_prior(n=1)
        theta_arr = np.asarray(theta)

        if theta_arr.ndim == 1:
            # Single theta; return log p(x|theta) with mixture across components
            if component is not None:
                kk = int(component)
                mean = self.A[kk] @ theta_arr
                ll = _logpdf_gaussian(x, mean, self.Sigma_x[kk])
                return ll
            else:
                logs = []
                for kk in range(self.n_components):
                    mean = self.A[kk] @ theta_arr
                    logs.append(np.log(self.weights[kk]) + _logpdf_gaussian(x, mean, self.Sigma_x[kk]))
                # log-sum-exp over components
                a = max(logs)
                logs_exp = [np.exp(ll - a) for ll in logs]
                return a + np.log(np.sum(logs_exp))
        else:
            # Batched thetas: return an array of log-likes per row
            res = []
            for i in range(theta_arr.shape[0]):
                th = theta_arr[i]
                if component is not None:
                    kk = int(component)
                    mean = self.A[kk] @ th
                    res.append(_logpdf_gaussian(x, mean, self.Sigma_x[kk]))
                else:
                    logs = []
                    for kk in range(self.n_components):
                        mean = self.A[kk] @ th
                        logs.append(np.log(self.weights[kk]) + _logpdf_gaussian(x, mean, self.Sigma_x[kk]))
                    a = max(logs)
                    logs_exp = [np.exp(ll - a) for ll in logs]
                    res.append(a + np.log(np.sum(logs_exp)))
            return np.asarray(res)

    def get_mixture_weights(self) -> np.ndarray:
        return self.weights

    # Internal helper: ensure private access to rng if needed in future
    def _rng(self):  # pragma: no cover
        return self._rng
