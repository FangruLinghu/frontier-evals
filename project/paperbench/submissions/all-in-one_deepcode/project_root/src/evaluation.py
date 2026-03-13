# Lightweight evaluation utilities for diffusion-based SBI experiments

from __future__ import annotations

from typing import Callable, Optional, Dict, Any, Tuple, Iterable
import numpy as np

__all__ = [
    "c2st_accuracy",
    "probability_flow_nll",
    "calibration_metrics",
    "posterior_predictive_samples",
]


def c2st_accuracy(
    posterior_samples: np.ndarray,
    ground_truth_samples: np.ndarray,
) -> float:
    """Compute a simple C2ST-like accuracy using a Gaussian Naive Bayes classifier.

    This function treats the two input arrays as samples from two different distributions
    and fits a diagonal-covariance Gaussian to each class. It then classifies a combined
    dataset by choosing the class with higher log-likelihood for each sample.

    Parameters
    - posterior_samples: shape (N0, D) samples from the posterior (class 0)
    - ground_truth_samples: shape (N1, D) samples from the ground-truth distribution (class 1)

    Returns
    - accuracy: float in [0, 1] representing the classifier's accuracy on the combined data
    """
    X0 = np.asarray(posterior_samples)
    X1 = np.asarray(ground_truth_samples)
    if X0.ndim != 2:
        raise ValueError("posterior_samples must be a 2D array of shape (N0, D)")
    if X1.ndim != 2:
        raise ValueError("ground_truth_samples must be a 2D array of shape (N1, D)")
    N0, D = X0.shape
    N1, D1 = X1.shape
    if D != D1:
        raise ValueError("Dimension mismatch between posterior and ground-truth samples")

    # Compute per-class means and diagonal variances with stability constant
    eps = 1e-6
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    var0 = np.var(X0, axis=0) + eps
    var1 = np.var(X1, axis=0) + eps

    # Compute log-likelihoods for each sample under both class models
    def log_likelihood(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        # X: (N, D), mu: (D,), var: (D,)
        diff = X - mu
        # diagonal Gaussian log-density
        ll = -0.5 * np.sum((diff ** 2) / var, axis=1)
        ll -= 0.5 * np.sum(np.log(2 * np.pi * var))
        return ll

    log_p0 = log_likelihood(np.vstack([X0, X1]), mu0, var0)[:N0 + N1]  # (N0+N1,)
    log_p1 = log_likelihood(np.vstack([X0, X1]), mu1, var1)[:N0 + N1]

    # Create labels: 0 for posterior, 1 for ground-truth
    labels = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])
    # Predicted class according to higher log-likelihood for each sample
    preds = (log_p1 > log_p0).astype(int)
    acc = float(np.mean(preds == labels))
    return max(0.0, min(1.0, acc))


def probability_flow_nll(
    x: np.ndarray,
    score_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    t_grid: Optional[np.ndarray] = None,
) -> float:
    """Toy proxy for NLL via probability-flow ODE. This is a lightweight placeholder
    intended for unit tests and demonstration runs where a full likelihood computation
    would be expensive to implement.

    The function returns a single scalar approximating the negative log-likelihood of x
    under a standard Gaussian prior, optionally modulated by the provided score_fn if given.

    - If score_fn is provided, this function does not attempt to integrate the true probability flow
      and instead returns a simple proxy that preserves compatibility with interfaces.

    Parameters
    - x: array-like, shape (D,) or (N, D)
    - score_fn: optional callable, not used in the proxy beyond type compatibility
    - t_grid: optional time grid descriptor (not used in the proxy)

    Returns
    - nll: scalar value representing the proxy negative log-likelihood (higher means less likely)
    """
    X = np.asarray(x)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # simple proxy: average squared norm scaled
    D = X.shape[1]
    mean = np.zeros(D)
    var = np.ones(D)
    ll = -0.5 * np.sum(((X - mean) ** 2) / (var + 1e-8), axis=1) - 0.5 * D * np.log(2 * np.pi * 1.0)
    # Return negative mean log-likelihood across samples as a scalar
    return float(-np.mean(ll))


def calibration_metrics(
    posterior_samples: np.ndarray,
    truth_values: np.ndarray,
    levels: Iterable[float] = (0.8, 0.95),
) -> Dict[str, Any]:
    """Compute simple calibration/coverage metrics for posterior samples.

    Given a set of posterior samples for each parameter (shape: N x D) and the true
    values (shape: D,), this function computes, for each confidence level, the fraction
    of true values that lie within the corresponding equal-tailed credible interval
    [lower, upper] for each dimension, then averages across dimensions.

    Parameters
    - posterior_samples: shape (N, D)
    - truth_values: shape (D,)
    - levels: iterable of confidence levels in (0,1)

    Returns
    - metrics: dict mapping level -> coverage (float in [0,1])
    """
    X = np.asarray(posterior_samples)
    truth = np.asarray(truth_values)
    if X.ndim != 2:
        raise ValueError("posterior_samples must be a 2D array of shape (N, D)")
    if truth.ndim != 1:
        raise ValueError("truth_values must be a 1D array of length D")
    N, D = X.shape
    if truth.shape[0] != D:
        raise ValueError("truth_values length must match the number of dimensions in posterior_samples")

    metrics: Dict[str, float] = {}
    X_sorted = X  # not strictly needed; keep for clarity
    for level in levels:
        if not (0.0 < level < 1.0):
            raise ValueError("levels must be in (0, 1)")
        lower = np.quantile(X_sorted, (1.0 - level) / 2.0, axis=0)
        upper = np.quantile(X_sorted, 1.0 - (1.0 - level) / 2.0, axis=0)
        within = (truth >= lower) & (truth <= upper)
        metrics[f"level_{int(level*100)}"] = float(np.mean(within))

    return metrics


def posterior_predictive_samples(
    posterior_samples: np.ndarray,
    sim_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: Optional[int] = None,
) -> np.ndarray:
    """Generate posterior predictive samples from a provided simulator function.

    For each posterior draw theta_i, call sim_fn(theta_i) to obtain a corresponding x_i.
    Returns an array of predictive samples with shape (N, ...), where N is the number of
    posterior samples. If n_samples is provided, only that many samples are generated.

    Parameters
    - posterior_samples: shape (N, D) or (N,)
    - sim_fn: function taking a theta vector and returning an observation vector
    - n_samples: optional cap on number of samples to generate

    Returns
    - preds: numpy array of predictive samples
    """
    X = np.asarray(posterior_samples)
    N = X.shape[0]
    if n_samples is not None:
        N = min(N, int(n_samples))
    preds = []
    for i in range(N):
        theta = X[i]
        pred = sim_fn(theta)
        preds.append(pred)
    return np.asarray(preds)
