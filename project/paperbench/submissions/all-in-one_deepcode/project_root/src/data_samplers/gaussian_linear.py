import numpy as np
from typing import Optional, Tuple

"""
Gaussian Linear data sampler.

API:
- GaussianLinearSampler(dim_theta, dim_x, seed=0, A=None, Sigma_x=None)
  - A: (dim_x, dim_theta) observation matrix. If None, randomly initialized.
  - Sigma_x: observation noise covariance. If 1D length dim_x, treated as diagonal.
- sample_prior(size=None) -> ndarray
  - Draws theta ~ N(0, I) by default. If size is provided, returns (size, dim_theta).
- simulate(theta) -> x
  - x = A @ theta + noise, noise ~ N(0, Sigma_x)
- log_likelihood(x, theta) -> float
  - log p(x | theta) under Gaussian noise with covariance Sigma_x.
"""

class GaussianLinearSampler:
    def __init__(self, dim_theta: int, dim_x: int, seed: int = 0,
                 A: Optional[np.ndarray] = None,
                 Sigma_x: Optional[np.ndarray] = None):
        self.dim_theta = int(dim_theta)
        self.dim_x = int(dim_x)
        self.rng = np.random.default_rng(seed)

        if A is None:
            # Random linear map (dim_x x dim_theta)
            self.A = self.rng.normal(size=(self.dim_x, self.dim_theta))
        else:
            self.A = np.asarray(A, dtype=float)
            if self.A.shape != (self.dim_x, self.dim_theta):
                raise ValueError(f"A must have shape (dim_x, dim_theta)=( {self.dim_x}, {self.dim_theta} ), got {self.A.shape}")

        # Noise covariance for x given theta
        if Sigma_x is None:
            self.Sigma_x = np.eye(self.dim_x)
            self._is_diag = True
            self._sigma_diag = np.ones(self.dim_x)
        else:
            S = np.asarray(Sigma_x, dtype=float)
            if S.ndim == 1:
                if S.size != self.dim_x:
                    raise ValueError("Sigma_x diagonal length must match dim_x")
                self.Sigma_x = np.diag(S)
                self._is_diag = True
                self._sigma_diag = S.copy()
            elif S.ndim == 2:
                if S.shape != (self.dim_x, self.dim_x):
                    raise ValueError("Sigma_x must be of shape (dim_x, dim_x)")
                self.Sigma_x = S
                # determine if diagonal for efficient computations
                if np.allclose(S, np.diag(np.diag(S))):
                    self._is_diag = True
                    self._sigma_diag = np.diag(S).copy()
                else:
                    self._is_diag = False
                    self._sigma_diag = None
            else:
                raise ValueError("Sigma_x must be 1D or 2D array")

        # Precompute a Cholesky-like factor for noise generation
        self._Sigma_chol = None
        try:
            self._Sigma_chol = np.linalg.cholesky(self.Sigma_x)
        except Exception:
            # Fallback to eigen-decomposition to construct a valid sqrt(Sigma_x)
            w, V = np.linalg.eigh(self.Sigma_x)
            w = np.maximum(w, 0.0)
            self._Sigma_chol = V @ np.diag(np.sqrt(w))

    def sample_prior(self, size: Optional[int] = None) -> np.ndarray:
        if size is None:
            return self.rng.normal(size=(self.dim_theta,))
        else:
            return self.rng.normal(size=(size, self.dim_theta))

    def simulate(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 1:
            if theta.size != self.dim_theta:
                raise ValueError("theta has incompatible dimension")
            mean = self.A @ theta  # shape (dim_x,)
            noise = self._draw_noise(self.dim_x, 1).reshape(self.dim_x,)
            return mean + noise
        elif theta.ndim == 2:
            if theta.shape[1] != self.dim_theta:
                raise ValueError("theta has incompatible dimension for batch")
            mean_batch = theta @ self.A.T  # shape (N, dim_x)
            N = theta.shape[0]
            noise_batch = self._draw_noise(self.dim_x, N).T  # shape (N, dim_x)
            return mean_batch + noise_batch
        else:
            raise ValueError("theta must be 1D or 2D array")

    def log_likelihood(self, x: np.ndarray, theta: np.ndarray) -> float:
        """Log-likelihood of x given theta under Gaussian noise with covariance Sigma_x."""
        x = np.asarray(x, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.dim_theta:
            raise ValueError("theta has incompatible dimension for log_likelihood")
        if x.size != self.dim_x:
            raise ValueError("x has incompatible dimension for log_likelihood")
        mean = self.A @ theta
        r = x - mean
        # Use diagonal vs full covariance for efficiency
        if self._is_diag:
            diag = self._sigma_diag
            if np.any(diag <= 0):
                raise ValueError("Non-positive diagonal elements in Sigma_x")
            ll = -0.5 * (np.sum((r ** 2) / diag) + np.sum(np.log(2 * np.pi * diag)))
            return float(ll)
        else:
            # Full covariance: ll = -0.5*(r^T inv(Sigma_x) r + log det(2π Σx))
            try:
                inv = np.linalg.inv(self.Sigma_x)
                sign, logdet = np.linalg.slogdet(self.Sigma_x)
                ll = -0.5 * (r.T @ inv @ r + logdet + self.dim_x * np.log(2 * np.pi))
                return float(ll)
            except np.linalg.LinAlgError:
                raise

    def _draw_noise(self, dim: int, batch_size: int) -> np.ndarray:
        # Draw Gaussian noise with covariance Sigma_x, using precomputed cholesky sqrt
        if self._Sigma_chol is None:
            raise RuntimeError("Sigma_x Cholesky decomposition not initialized")
        z = self.rng.normal(size=(batch_size, dim))  # shape (batch_size, dim)
        # noise = z @ Sigma_chol.T gives (batch_size, dim)
        if batch_size == 1:
            return (self._Sigma_chol @ z[0])
        else:
            return z @ self._Sigma_chol.T

    @staticmethod
    def _to_1d(v):
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1)
        return arr


__all__ = ["GaussianLinearSampler"]
