"""
Evaluation metrics for Simformer.

Includes C2ST (Classifier Two-Sample Test) and expected coverage analysis.
"""

from simformer.evaluation.c2st import c2st, c2st_accuracy, train_classifier
from simformer.evaluation.coverage import (
    expected_coverage,
    coverage_probability,
    highest_posterior_density_region,
    credible_interval_coverage,
)
from simformer.evaluation.metrics import (
    log_prob_true_posterior,
    mmd_score,
    wasserstein_distance,
    posterior_mean_error,
    posterior_variance_error,
)

__all__ = [
    "c2st",
    "c2st_accuracy",
    "train_classifier",
    "expected_coverage",
    "coverage_probability",
    "highest_posterior_density_region",
    "credible_interval_coverage",
    "log_prob_true_posterior",
    "mmd_score",
    "wasserstein_distance",
    "posterior_mean_error",
    "posterior_variance_error",
]
