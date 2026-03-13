"""
Additional evaluation metrics for simulation-based inference.

Includes:
- Log probability under true posterior (when available)
- Maximum Mean Discrepancy (MMD)
- Wasserstein distance
- Posterior moment errors
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable
from scipy import stats


def log_prob_true_posterior(
    samples: torch.Tensor,
    log_prob_fn: Callable,
) -> torch.Tensor:
    """
    Compute log probability of samples under true posterior.

    This is only possible when the true posterior is analytically tractable.

    Args:
        samples: Posterior samples, shape (n_samples, n_params)
        log_prob_fn: Function that computes log p(θ|x)

    Returns:
        Log probabilities, shape (n_samples,)
    """
    return log_prob_fn(samples)


def mmd_score(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    kernel: str = "rbf",
    bandwidth: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy between two sample sets.

    MMD is a kernel-based distance between distributions.

    Args:
        samples_p: Samples from first distribution, shape (n_p, dim)
        samples_q: Samples from second distribution, shape (n_q, dim)
        kernel: Kernel type ("rbf", "polynomial")
        bandwidth: Kernel bandwidth (auto if None)

    Returns:
        MMD score (lower is better, 0 means identical distributions)
    """
    n_p = samples_p.shape[0]
    n_q = samples_q.shape[0]

    # Auto bandwidth using median heuristic
    if bandwidth is None:
        all_samples = torch.cat([samples_p, samples_q], dim=0)
        dists = torch.cdist(all_samples, all_samples)
        bandwidth = torch.median(dists[dists > 0]).item()

    if kernel == "rbf":
        def kernel_fn(x, y):
            sq_dist = torch.cdist(x, y, p=2) ** 2
            return torch.exp(-sq_dist / (2 * bandwidth ** 2))
    elif kernel == "polynomial":
        def kernel_fn(x, y):
            return (torch.mm(x, y.T) + 1) ** 3
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Compute MMD
    K_pp = kernel_fn(samples_p, samples_p)
    K_qq = kernel_fn(samples_q, samples_q)
    K_pq = kernel_fn(samples_p, samples_q)

    mmd_squared = (
        K_pp.sum() / (n_p * (n_p - 1)) +
        K_qq.sum() / (n_q * (n_q - 1)) -
        2 * K_pq.sum() / (n_p * n_q)
    )

    return max(0, mmd_squared.item()) ** 0.5


def wasserstein_distance(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    p: int = 1,
) -> float:
    """
    Compute Wasserstein distance between sample sets.

    Uses sliced Wasserstein approximation for high dimensions.

    Args:
        samples_p: Samples from first distribution, shape (n_p, dim)
        samples_q: Samples from second distribution, shape (n_q, dim)
        p: Order of Wasserstein distance (1 or 2)

    Returns:
        Wasserstein distance
    """
    dim = samples_p.shape[1]

    if dim == 1:
        # Exact 1D Wasserstein
        samples_p_np = samples_p.detach().cpu().numpy().squeeze()
        samples_q_np = samples_q.detach().cpu().numpy().squeeze()
        return stats.wasserstein_distance(samples_p_np, samples_q_np)

    # Sliced Wasserstein for higher dimensions
    n_projections = 100
    distances = []

    for _ in range(n_projections):
        # Random projection direction
        direction = torch.randn(dim, device=samples_p.device)
        direction = direction / direction.norm()

        # Project samples
        proj_p = torch.mv(samples_p, direction)
        proj_q = torch.mv(samples_q, direction)

        # 1D Wasserstein
        proj_p_sorted = torch.sort(proj_p)[0]
        proj_q_sorted = torch.sort(proj_q)[0]

        # Interpolate to same length if needed
        n_p, n_q = len(proj_p_sorted), len(proj_q_sorted)
        if n_p != n_q:
            # Resample to same length
            indices_p = torch.linspace(0, n_p - 1, min(n_p, n_q)).long()
            indices_q = torch.linspace(0, n_q - 1, min(n_p, n_q)).long()
            proj_p_sorted = proj_p_sorted[indices_p]
            proj_q_sorted = proj_q_sorted[indices_q]

        if p == 1:
            dist = torch.abs(proj_p_sorted - proj_q_sorted).mean()
        else:
            dist = (torch.abs(proj_p_sorted - proj_q_sorted) ** p).mean() ** (1 / p)

        distances.append(dist.item())

    return np.mean(distances)


def posterior_mean_error(
    posterior_samples: torch.Tensor,
    true_values: torch.Tensor,
    relative: bool = False,
) -> torch.Tensor:
    """
    Compute error in posterior mean estimate.

    Args:
        posterior_samples: Samples, shape (n_observations, n_samples, n_params)
        true_values: True values, shape (n_observations, n_params)
        relative: Whether to compute relative error

    Returns:
        Mean error per parameter, shape (n_params,)
    """
    posterior_means = posterior_samples.mean(dim=1)  # (n_observations, n_params)
    errors = torch.abs(posterior_means - true_values)

    if relative:
        errors = errors / (torch.abs(true_values) + 1e-8)

    return errors.mean(dim=0)  # (n_params,)


def posterior_variance_error(
    posterior_samples: torch.Tensor,
    true_variances: torch.Tensor,
    relative: bool = True,
) -> torch.Tensor:
    """
    Compute error in posterior variance estimate.

    Args:
        posterior_samples: Samples, shape (n_observations, n_samples, n_params)
        true_variances: True posterior variances, shape (n_observations, n_params)
        relative: Whether to compute relative error

    Returns:
        Variance error per parameter, shape (n_params,)
    """
    posterior_vars = posterior_samples.var(dim=1)  # (n_observations, n_params)
    errors = torch.abs(posterior_vars - true_variances)

    if relative:
        errors = errors / (true_variances + 1e-8)

    return errors.mean(dim=0)  # (n_params,)


def negative_log_likelihood(
    posterior_samples: torch.Tensor,
    true_values: torch.Tensor,
    bandwidth: Optional[float] = None,
) -> float:
    """
    Estimate negative log-likelihood of true values under posterior.

    Uses kernel density estimation.

    Args:
        posterior_samples: Samples, shape (n_samples, n_params)
        true_values: True values, shape (n_params,)
        bandwidth: KDE bandwidth

    Returns:
        Negative log-likelihood
    """
    from scipy.stats import gaussian_kde

    samples_np = posterior_samples.detach().cpu().numpy().T  # (n_params, n_samples)
    true_np = true_values.detach().cpu().numpy()

    # KDE
    kde = gaussian_kde(samples_np, bw_method=bandwidth)

    # Evaluate at true value
    log_prob = kde.logpdf(true_np)

    return -float(log_prob[0])


def energy_distance(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
) -> float:
    """
    Compute energy distance between sample sets.

    Energy distance is 2E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]

    Args:
        samples_p: Samples from first distribution
        samples_q: Samples from second distribution

    Returns:
        Energy distance
    """
    # Cross term
    cross_dist = torch.cdist(samples_p, samples_q, p=2).mean()

    # Within-p term
    within_p = torch.cdist(samples_p, samples_p, p=2)
    n_p = len(samples_p)
    within_p_mean = within_p.sum() / (n_p * (n_p - 1)) if n_p > 1 else 0

    # Within-q term
    within_q = torch.cdist(samples_q, samples_q, p=2)
    n_q = len(samples_q)
    within_q_mean = within_q.sum() / (n_q * (n_q - 1)) if n_q > 1 else 0

    energy = 2 * cross_dist - within_p_mean - within_q_mean
    return max(0, energy.item())


def posterior_contraction(
    posterior_samples: torch.Tensor,
    prior_samples: torch.Tensor,
) -> torch.Tensor:
    """
    Compute posterior contraction relative to prior.

    Measures how much the posterior has narrowed compared to the prior.

    Args:
        posterior_samples: Samples, shape (n_observations, n_samples, n_params)
        prior_samples: Prior samples, shape (n_prior_samples, n_params)

    Returns:
        Contraction ratio per parameter, shape (n_params,)
    """
    prior_var = prior_samples.var(dim=0)  # (n_params,)
    posterior_var = posterior_samples.var(dim=1).mean(dim=0)  # (n_params,)

    contraction = 1 - posterior_var / (prior_var + 1e-8)
    return contraction


def z_score(
    posterior_samples: torch.Tensor,
    true_values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute z-scores of true values under posteriors.

    Z-score = (true - posterior_mean) / posterior_std

    Args:
        posterior_samples: Samples, shape (n_observations, n_samples, n_params)
        true_values: True values, shape (n_observations, n_params)

    Returns:
        Z-scores, shape (n_observations, n_params)
    """
    posterior_mean = posterior_samples.mean(dim=1)
    posterior_std = posterior_samples.std(dim=1)

    return (true_values - posterior_mean) / (posterior_std + 1e-8)


def shrinkage(
    posterior_samples: torch.Tensor,
    prior_samples: torch.Tensor,
    true_values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute shrinkage towards prior mean.

    Shrinkage = 1 - |posterior_mean - prior_mean| / |true - prior_mean|

    Args:
        posterior_samples: Samples, shape (n_observations, n_samples, n_params)
        prior_samples: Prior samples, shape (n_prior_samples, n_params)
        true_values: True values, shape (n_observations, n_params)

    Returns:
        Shrinkage values, shape (n_observations, n_params)
    """
    prior_mean = prior_samples.mean(dim=0)  # (n_params,)
    posterior_mean = posterior_samples.mean(dim=1)  # (n_observations, n_params)

    numerator = torch.abs(posterior_mean - prior_mean)
    denominator = torch.abs(true_values - prior_mean) + 1e-8

    return 1 - numerator / denominator
