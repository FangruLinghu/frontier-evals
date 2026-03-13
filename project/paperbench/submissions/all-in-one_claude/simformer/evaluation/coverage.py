"""
Expected coverage analysis for posterior calibration.

Coverage analysis evaluates whether posterior credible regions have
the correct frequentist coverage. A well-calibrated posterior should
have α-credible regions that contain the true parameter α% of the time.

Reference:
- Talts et al. (2018). Validating Bayesian Inference Algorithms with Simulation-Based Calibration.
- Hermans et al. (2022). A Trust Crisis In Simulation-Based Inference?
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Callable
from scipy import stats


def expected_coverage(
    posterior_samples: torch.Tensor,
    true_values: torch.Tensor,
    confidence_levels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute expected coverage for a set of posteriors.

    For each confidence level α, computes the fraction of times the
    true parameter falls within the α-credible interval.

    Args:
        posterior_samples: Samples from posteriors, shape (n_observations, n_samples, n_params)
        true_values: True parameter values, shape (n_observations, n_params)
        confidence_levels: Confidence levels to evaluate (default: 0.05, 0.1, ..., 0.95)

    Returns:
        Tuple of (confidence_levels, empirical_coverage)
    """
    if confidence_levels is None:
        confidence_levels = torch.linspace(0.05, 0.95, 19)

    n_observations, n_samples, n_params = posterior_samples.shape
    device = posterior_samples.device

    empirical_coverage = torch.zeros(len(confidence_levels), device=device)

    for i, alpha in enumerate(confidence_levels):
        # Compute credible intervals for each observation and parameter
        coverage_count = 0
        total_count = 0

        for obs_idx in range(n_observations):
            for param_idx in range(n_params):
                samples = posterior_samples[obs_idx, :, param_idx]
                true_val = true_values[obs_idx, param_idx]

                # Compute symmetric credible interval
                lower_q = (1 - alpha) / 2
                upper_q = 1 - lower_q

                lower = torch.quantile(samples, lower_q)
                upper = torch.quantile(samples, upper_q)

                if lower <= true_val <= upper:
                    coverage_count += 1
                total_count += 1

        empirical_coverage[i] = coverage_count / total_count

    return confidence_levels, empirical_coverage


def coverage_probability(
    posterior_samples: torch.Tensor,
    true_value: torch.Tensor,
    alpha: float = 0.95,
    method: str = "symmetric",
) -> bool:
    """
    Check if true value is covered by credible interval.

    Args:
        posterior_samples: Samples from posterior, shape (n_samples, n_params)
        true_value: True parameter value, shape (n_params,)
        alpha: Confidence level
        method: Interval method ("symmetric", "hpd")

    Returns:
        True if covered, False otherwise
    """
    n_params = true_value.shape[0]

    for param_idx in range(n_params):
        samples = posterior_samples[:, param_idx]

        if method == "symmetric":
            lower_q = (1 - alpha) / 2
            upper_q = 1 - lower_q
            lower = torch.quantile(samples, lower_q)
            upper = torch.quantile(samples, upper_q)
        elif method == "hpd":
            lower, upper = highest_posterior_density_region(samples, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        if not (lower <= true_value[param_idx] <= upper):
            return False

    return True


def highest_posterior_density_region(
    samples: torch.Tensor,
    alpha: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute the highest posterior density (HPD) region.

    The HPD region is the shortest interval containing α% of the probability mass.

    Args:
        samples: 1D tensor of posterior samples
        alpha: Confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    samples_np = samples.detach().cpu().numpy()
    samples_sorted = np.sort(samples_np)
    n = len(samples_sorted)

    # Number of samples in the interval
    n_in_interval = int(np.ceil(alpha * n))

    # Find shortest interval
    widths = samples_sorted[n_in_interval:] - samples_sorted[:-n_in_interval]
    best_idx = np.argmin(widths)

    lower = samples_sorted[best_idx]
    upper = samples_sorted[best_idx + n_in_interval - 1]

    return float(lower), float(upper)


def credible_interval_coverage(
    posterior_samples: torch.Tensor,
    true_values: torch.Tensor,
    alpha: float = 0.95,
    per_parameter: bool = False,
) -> torch.Tensor:
    """
    Compute empirical coverage at a specific confidence level.

    Args:
        posterior_samples: Samples, shape (n_observations, n_samples, n_params)
        true_values: True values, shape (n_observations, n_params)
        alpha: Confidence level
        per_parameter: Whether to return per-parameter coverage

    Returns:
        Empirical coverage (scalar or per-parameter tensor)
    """
    n_observations, n_samples, n_params = posterior_samples.shape
    device = posterior_samples.device

    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q

    if per_parameter:
        coverage = torch.zeros(n_params, device=device)

        for param_idx in range(n_params):
            covered = 0
            for obs_idx in range(n_observations):
                samples = posterior_samples[obs_idx, :, param_idx]
                true_val = true_values[obs_idx, param_idx]

                lower = torch.quantile(samples, lower_q)
                upper = torch.quantile(samples, upper_q)

                if lower <= true_val <= upper:
                    covered += 1

            coverage[param_idx] = covered / n_observations
    else:
        covered = 0
        total = n_observations * n_params

        for obs_idx in range(n_observations):
            for param_idx in range(n_params):
                samples = posterior_samples[obs_idx, :, param_idx]
                true_val = true_values[obs_idx, param_idx]

                lower = torch.quantile(samples, lower_q)
                upper = torch.quantile(samples, upper_q)

                if lower <= true_val <= upper:
                    covered += 1

        coverage = torch.tensor(covered / total, device=device)

    return coverage


def simulation_based_calibration(
    prior_sampler: Callable,
    simulator: Callable,
    posterior_sampler: Callable,
    n_simulations: int = 1000,
    n_posterior_samples: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform simulation-based calibration (SBC).

    SBC checks posterior calibration by verifying that ranks of true
    parameters within posterior samples are uniformly distributed.

    Args:
        prior_sampler: Function returning samples from prior
        simulator: Function returning observations given parameters
        posterior_sampler: Function returning posterior samples given observation
        n_simulations: Number of SBC simulations
        n_posterior_samples: Samples per posterior

    Returns:
        Tuple of (ranks, expected_uniform) for histogram comparison
    """
    ranks_list = []

    for _ in range(n_simulations):
        # Sample from prior
        theta_true = prior_sampler(1)  # (1, n_params)

        # Simulate observation
        x = simulator(theta_true)  # (1, n_data)

        # Sample from posterior
        theta_samples = posterior_sampler(x, n_posterior_samples)  # (n_samples, n_params)

        # Compute ranks
        ranks = (theta_samples < theta_true).sum(dim=0)  # (n_params,)
        ranks_list.append(ranks)

    ranks = torch.stack(ranks_list)  # (n_simulations, n_params)

    # Expected uniform distribution
    expected = torch.ones(n_posterior_samples + 1) * n_simulations / (n_posterior_samples + 1)

    return ranks, expected


def coverage_plot_data(
    confidence_levels: torch.Tensor,
    empirical_coverage: torch.Tensor,
) -> dict:
    """
    Prepare data for coverage plot.

    Args:
        confidence_levels: Expected coverage levels
        empirical_coverage: Empirical coverage at each level

    Returns:
        Dictionary with plot data
    """
    return {
        "confidence_levels": confidence_levels.cpu().numpy(),
        "empirical_coverage": empirical_coverage.cpu().numpy(),
        "ideal_coverage": confidence_levels.cpu().numpy(),  # Diagonal line
    }


def calibration_error(
    confidence_levels: torch.Tensor,
    empirical_coverage: torch.Tensor,
) -> float:
    """
    Compute expected calibration error (ECE).

    ECE is the mean absolute difference between expected and empirical coverage.

    Args:
        confidence_levels: Expected coverage levels
        empirical_coverage: Empirical coverage at each level

    Returns:
        Expected calibration error
    """
    return torch.abs(confidence_levels - empirical_coverage).mean().item()


def coverage_width(
    posterior_samples: torch.Tensor,
    alpha: float = 0.95,
) -> torch.Tensor:
    """
    Compute credible interval widths.

    Args:
        posterior_samples: Samples, shape (n_observations, n_samples, n_params)
        alpha: Confidence level

    Returns:
        Widths, shape (n_observations, n_params)
    """
    n_observations, n_samples, n_params = posterior_samples.shape
    device = posterior_samples.device

    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q

    widths = torch.zeros(n_observations, n_params, device=device)

    for obs_idx in range(n_observations):
        for param_idx in range(n_params):
            samples = posterior_samples[obs_idx, :, param_idx]
            lower = torch.quantile(samples, lower_q)
            upper = torch.quantile(samples, upper_q)
            widths[obs_idx, param_idx] = upper - lower

    return widths


def rank_histogram(
    posterior_samples: torch.Tensor,
    true_values: torch.Tensor,
    n_bins: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rank histogram for SBC.

    Args:
        posterior_samples: Samples, shape (n_observations, n_samples, n_params)
        true_values: True values, shape (n_observations, n_params)
        n_bins: Number of histogram bins

    Returns:
        Tuple of (bin_counts, bin_edges)
    """
    n_observations, n_samples, n_params = posterior_samples.shape

    # Compute ranks
    ranks = []
    for obs_idx in range(n_observations):
        for param_idx in range(n_params):
            samples = posterior_samples[obs_idx, :, param_idx]
            true_val = true_values[obs_idx, param_idx]
            rank = (samples < true_val).sum().item()
            ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float32)

    # Create histogram
    bin_edges = torch.linspace(0, n_samples, n_bins + 1)
    bin_counts = torch.histc(ranks, bins=n_bins, min=0, max=n_samples)

    return bin_counts, bin_edges


def uniformity_test(
    ranks: torch.Tensor,
    n_samples: int,
) -> Tuple[float, float]:
    """
    Test uniformity of ranks using Kolmogorov-Smirnov test.

    Args:
        ranks: Rank values, shape (n,)
        n_samples: Number of posterior samples (determines max rank)

    Returns:
        Tuple of (ks_statistic, p_value)
    """
    ranks_np = ranks.detach().cpu().numpy().flatten()

    # Normalize to [0, 1]
    ranks_normalized = ranks_np / n_samples

    # KS test against uniform
    ks_stat, p_value = stats.kstest(ranks_normalized, "uniform")

    return float(ks_stat), float(p_value)
