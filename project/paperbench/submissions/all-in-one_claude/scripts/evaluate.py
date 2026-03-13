#!/usr/bin/env python
"""
Evaluation script for Simformer.

Evaluates trained models using C2ST and expected coverage metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/two_moons/final_model.pt --task two_moons
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simformer import Simformer
from simformer.diffusion import VESDE, VPSDE, sample_posterior
from simformer.evaluation import (
    c2st,
    c2st_accuracy,
    expected_coverage,
    credible_interval_coverage,
    calibration_error,
    mmd_score,
    posterior_mean_error,
)
from simformer.tasks import (
    GaussianLinearTask,
    GaussianMixtureTask,
    TwoMoonsTask,
    SLCPTask,
    TreeTask,
    HMMTask,
    LotkaVolterraTask,
    SIRDTask,
    HodgkinHuxleyTask,
)
from simformer.utils import get_device, set_seed


TASK_REGISTRY = {
    "gaussian_linear": GaussianLinearTask,
    "gaussian_mixture": GaussianMixtureTask,
    "two_moons": TwoMoonsTask,
    "slcp": SLCPTask,
    "tree": TreeTask,
    "hmm": HMMTask,
    "lotka_volterra": LotkaVolterraTask,
    "sird": SIRDTask,
    "hodgkin_huxley": HodgkinHuxleyTask,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Simformer")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_REGISTRY.keys()),
        help="Task to evaluate on",
    )

    # Evaluation settings
    parser.add_argument("--n_observations", type=int, default=100, help="Number of test observations")
    parser.add_argument("--n_posterior_samples", type=int, default=1000, help="Samples per posterior")
    parser.add_argument("--n_reference_samples", type=int, default=10000, help="Reference samples for C2ST")
    parser.add_argument("--n_diffusion_steps", type=int, default=500, help="Diffusion sampling steps")

    # Output
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--save_samples", action="store_true", help="Save posterior samples")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device")

    return parser.parse_args()


def load_model(checkpoint_path: str, task, device) -> Simformer:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    config = checkpoint.get("config", {})

    # Create SDE
    sde_type = config.get("sde", "vesde")
    if sde_type == "vesde":
        sde = VESDE()
    else:
        sde = VPSDE()

    # Create model
    model = Simformer(
        n_params=task.n_params,
        n_data=task.n_data,
        token_dim=config.get("token_dim", 50),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 4),
        widening_factor=config.get("widening_factor", 3),
        time_embed_dim=config.get("time_embed_dim", 128),
        sde=sde,
    )

    # Load weights
    if "ema_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device).eval()


def evaluate_c2st(
    model: Simformer,
    task,
    observation: torch.Tensor,
    true_theta: torch.Tensor,
    n_samples: int,
    n_steps: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate C2ST for a single observation."""
    # Sample from approximate posterior
    posterior_samples = sample_posterior(
        model,
        observation,
        n_samples=n_samples,
        n_steps=n_steps,
    )

    # Generate reference samples (from prior conditioned on observation)
    # For tasks with tractable likelihood, we can use rejection sampling
    # For now, we use samples from the prior as a baseline

    # If task has reference posterior, use it
    if hasattr(task, "sample_reference_posterior"):
        reference_samples = task.sample_reference_posterior(observation, n_samples)
    else:
        # Use prior samples (this is a weak baseline)
        reference_samples = task.sample_prior(n_samples).to(device)

    # Compute C2ST
    c2st_mean, c2st_std = c2st(posterior_samples, reference_samples)

    return {
        "c2st_mean": c2st_mean,
        "c2st_std": c2st_std,
    }


def evaluate_coverage(
    posterior_samples_all: torch.Tensor,
    true_values: torch.Tensor,
) -> Dict[str, Any]:
    """Evaluate coverage metrics."""
    # Expected coverage
    conf_levels, emp_coverage = expected_coverage(posterior_samples_all, true_values)

    # Calibration error
    cal_error = calibration_error(conf_levels, emp_coverage)

    # 95% coverage
    coverage_95 = credible_interval_coverage(
        posterior_samples_all, true_values, alpha=0.95
    )

    return {
        "confidence_levels": conf_levels.cpu().numpy().tolist(),
        "empirical_coverage": emp_coverage.cpu().numpy().tolist(),
        "calibration_error": cal_error,
        "coverage_95": coverage_95.item(),
    }


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Using device: {device}")

    # Create task
    print(f"Creating task: {args.task}")
    task = TASK_REGISTRY[args.task]()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, task, device)
    print("Model loaded successfully")

    # Generate test data
    print(f"\nGenerating {args.n_observations} test observations...")
    true_thetas = task.sample_prior(args.n_observations).to(device)
    observations = task.simulate(true_thetas).to(device)

    # Sample posteriors
    print(f"Sampling {args.n_posterior_samples} posterior samples per observation...")
    print(f"Using {args.n_diffusion_steps} diffusion steps")

    all_posterior_samples = []
    c2st_scores = []

    for i in range(args.n_observations):
        if (i + 1) % 10 == 0:
            print(f"  Processing observation {i + 1}/{args.n_observations}")

        obs = observations[i:i+1]

        # Sample posterior
        with torch.no_grad():
            posterior_samples = sample_posterior(
                model,
                obs,
                n_samples=args.n_posterior_samples,
                n_steps=args.n_diffusion_steps,
            )

        all_posterior_samples.append(posterior_samples)

    # Stack all samples: (n_observations, n_samples, n_params)
    all_posterior_samples = torch.stack(all_posterior_samples, dim=0)

    # Evaluate coverage
    print("\nEvaluating coverage...")
    coverage_results = evaluate_coverage(all_posterior_samples, true_thetas)

    print(f"  Calibration Error: {coverage_results['calibration_error']:.4f}")
    print(f"  95% Coverage: {coverage_results['coverage_95']:.4f}")

    # Evaluate C2ST (on subset to save time)
    print("\nEvaluating C2ST...")
    n_c2st = min(10, args.n_observations)

    for i in range(n_c2st):
        obs = observations[i:i+1]
        true_theta = true_thetas[i:i+1]
        posterior_samples = all_posterior_samples[i]

        # Use prior samples as reference for C2ST (weak baseline)
        reference_samples = task.sample_prior(len(posterior_samples)).to(device)
        c2st_mean, _ = c2st(posterior_samples, reference_samples)
        c2st_scores.append(c2st_mean)

    mean_c2st = np.mean(c2st_scores)
    std_c2st = np.std(c2st_scores)
    print(f"  C2ST (vs prior): {mean_c2st:.4f} +/- {std_c2st:.4f}")

    # Compute additional metrics
    print("\nComputing additional metrics...")
    mean_errors = posterior_mean_error(all_posterior_samples, true_thetas)
    print(f"  Posterior Mean Error: {mean_errors.mean().item():.4f}")

    # MMD score
    mmd_scores = []
    for i in range(n_c2st):
        reference = task.sample_prior(args.n_posterior_samples).to(device)
        mmd = mmd_score(all_posterior_samples[i], reference)
        mmd_scores.append(mmd)
    print(f"  MMD (vs prior): {np.mean(mmd_scores):.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_observations": args.n_observations,
        "n_posterior_samples": args.n_posterior_samples,
        "n_diffusion_steps": args.n_diffusion_steps,
        "coverage": coverage_results,
        "c2st_mean": mean_c2st,
        "c2st_std": std_c2st,
        "posterior_mean_error": mean_errors.cpu().numpy().tolist(),
        "mmd_vs_prior": np.mean(mmd_scores),
    }

    results_path = output_dir / f"{args.task}_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Optionally save samples
    if args.save_samples:
        samples_path = output_dir / f"{args.task}_samples.pt"
        torch.save({
            "posterior_samples": all_posterior_samples.cpu(),
            "true_thetas": true_thetas.cpu(),
            "observations": observations.cpu(),
        }, samples_path)
        print(f"Samples saved to: {samples_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
