#!/usr/bin/env python
"""
Run all benchmark tasks for Simformer.

This script trains and evaluates Simformer on all benchmark tasks
from the paper.

Usage:
    python scripts/run_benchmarks.py --tasks two_moons slcp --epochs 100
    python scripts/run_benchmarks.py --all --epochs 50
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from simformer import Simformer
from simformer.diffusion import VESDE, VPSDE, sample_posterior
from simformer.training import Trainer, TrainingConfig
from simformer.evaluation import c2st, expected_coverage, calibration_error
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

# Default configurations per task (based on paper)
TASK_CONFIGS = {
    "gaussian_linear": {"n_layers": 6, "epochs": 50},
    "gaussian_mixture": {"n_layers": 6, "epochs": 50},
    "two_moons": {"n_layers": 6, "epochs": 100},
    "slcp": {"n_layers": 6, "epochs": 100},
    "tree": {"n_layers": 6, "epochs": 100},
    "hmm": {"n_layers": 6, "epochs": 100},
    "lotka_volterra": {"n_layers": 8, "epochs": 200},
    "sird": {"n_layers": 8, "epochs": 200},
    "hodgkin_huxley": {"n_layers": 8, "epochs": 200},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Simformer benchmarks")

    # Tasks
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        choices=list(TASK_REGISTRY.keys()),
        help="Tasks to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all tasks")

    # Model
    parser.add_argument("--token_dim", type=int, default=50, help="Token dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--widening_factor", type=int, default=3, help="MLP widening factor")
    parser.add_argument("--sde", type=str, default="vesde", choices=["vesde", "vpsde"])

    # Training
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--samples_per_epoch", type=int, default=100000, help="Samples per epoch")

    # Evaluation
    parser.add_argument("--n_test_observations", type=int, default=100, help="Test observations")
    parser.add_argument("--n_posterior_samples", type=int, default=1000, help="Posterior samples")
    parser.add_argument("--n_diffusion_steps", type=int, default=500, help="Diffusion steps")

    # Output
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--skip_training", action="store_true", help="Skip training (evaluate only)")

    return parser.parse_args()


def train_task(
    task_name: str,
    args,
    device: torch.device,
    output_dir: Path,
) -> Simformer:
    """Train Simformer on a single task."""
    print(f"\n{'='*60}")
    print(f"Training on {task_name}")
    print(f"{'='*60}")

    # Get task config
    task_config = TASK_CONFIGS.get(task_name, {})
    n_layers = task_config.get("n_layers", 6)
    epochs = args.epochs if args.epochs is not None else task_config.get("epochs", 100)

    # Create task
    task = TASK_REGISTRY[task_name]()
    print(f"Task: n_params={task.n_params}, n_data={task.n_data}")

    # Create SDE
    if args.sde == "vesde":
        sde = VESDE()
    else:
        sde = VPSDE()

    # Create model
    model = Simformer(
        n_params=task.n_params,
        n_data=task.n_data,
        token_dim=args.token_dim,
        n_layers=n_layers,
        n_heads=args.n_heads,
        widening_factor=args.widening_factor,
        sde=sde,
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints" / task_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    config = TrainingConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=str(checkpoint_dir),
        device=device,
    )
    trainer = Trainer(model, config)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        # Generate data
        theta = task.sample_prior(args.samples_per_epoch).to(device)
        x = task.simulate(theta).to(device)

        # Train
        metrics = trainer.train_epoch(theta, x)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {metrics['loss']:.6f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")

    # Save model
    trainer.save_checkpoint("final_model.pt")

    return model, task


def evaluate_task(
    model: Simformer,
    task,
    task_name: str,
    args,
    device: torch.device,
) -> dict:
    """Evaluate Simformer on a single task."""
    print(f"\nEvaluating {task_name}...")

    model.eval()

    # Generate test data
    true_thetas = task.sample_prior(args.n_test_observations).to(device)
    observations = task.simulate(true_thetas).to(device)

    # Sample posteriors
    all_posterior_samples = []
    for i in range(args.n_test_observations):
        obs = observations[i:i+1]

        with torch.no_grad():
            posterior_samples = sample_posterior(
                model,
                obs,
                n_samples=args.n_posterior_samples,
                n_steps=args.n_diffusion_steps,
            )
        all_posterior_samples.append(posterior_samples)

    all_posterior_samples = torch.stack(all_posterior_samples, dim=0)

    # Compute coverage
    conf_levels, emp_coverage = expected_coverage(all_posterior_samples, true_thetas)
    cal_error = calibration_error(conf_levels, emp_coverage)

    # Compute C2ST
    c2st_scores = []
    for i in range(min(10, args.n_test_observations)):
        reference = task.sample_prior(args.n_posterior_samples).to(device)
        c2st_mean, _ = c2st(all_posterior_samples[i], reference)
        c2st_scores.append(c2st_mean)

    results = {
        "task": task_name,
        "calibration_error": cal_error,
        "c2st_mean": float(sum(c2st_scores) / len(c2st_scores)),
        "coverage_95": (emp_coverage[conf_levels >= 0.95][0] if any(conf_levels >= 0.95) else None),
    }

    print(f"  Calibration Error: {results['calibration_error']:.4f}")
    print(f"  C2ST (vs prior): {results['c2st_mean']:.4f}")

    return results


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Using device: {device}")

    # Determine tasks
    if args.all:
        tasks = list(TASK_REGISTRY.keys())
    elif args.tasks is not None:
        tasks = args.tasks
    else:
        tasks = ["two_moons"]  # Default

    print(f"\nTasks to run: {tasks}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["tasks"] = tasks
    config["timestamp"] = timestamp
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run benchmarks
    all_results = []

    for task_name in tasks:
        try:
            if not args.skip_training:
                model, task = train_task(task_name, args, device, output_dir)
            else:
                # Load existing model
                checkpoint_path = output_dir / "checkpoints" / task_name / "final_model.pt"
                if not checkpoint_path.exists():
                    print(f"Skipping {task_name}: no checkpoint found")
                    continue
                task = TASK_REGISTRY[task_name]()
                # Load model logic would go here

            results = evaluate_task(model, task, task_name, args, device)
            all_results.append(results)

        except Exception as e:
            print(f"Error on {task_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Task':<20} {'Cal. Error':<15} {'C2ST':<10}")
    print("-"*60)
    for r in all_results:
        print(f"{r['task']:<20} {r['calibration_error']:<15.4f} {r['c2st_mean']:<10.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
