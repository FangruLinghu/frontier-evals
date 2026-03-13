#!/usr/bin/env python
"""
Example of arbitrary conditional sampling with Simformer.

This script demonstrates sampling from arbitrary conditionals:
p(θ_A, x_B | θ_C, x_D)

The key feature of Simformer is its ability to sample from ANY
conditional distribution of the joint p(θ, x) without retraining.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simformer import Simformer, VESDE, sample_arbitrary_conditional
from simformer.training import Trainer, TrainingConfig
from simformer.tasks import SLCPTask
from simformer.utils import get_device, set_seed


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Create task with more parameters
    print("\nCreating SLCP task (5 params, 8 data)...")
    task = SLCPTask()

    # Create and train model (abbreviated)
    print("Creating and training model...")
    model = Simformer(
        n_params=task.n_params,
        n_data=task.n_data,
        token_dim=50,
        n_layers=6,
        sde=VESDE(),
    ).to(device)

    # Quick training (use more epochs for real applications)
    config = TrainingConfig(epochs=10, batch_size=1000, device=device)
    trainer = Trainer(model, config)

    for epoch in range(config.epochs):
        theta = task.sample_prior(10000).to(device)
        x = task.simulate(theta).to(device)
        trainer.train_epoch(theta, x)

    model.eval()

    # Generate a test case
    theta_true = task.sample_prior(1).to(device)
    x_obs = task.simulate(theta_true).to(device)

    print(f"\nTrue parameters θ: {theta_true.squeeze().cpu().numpy()}")
    print(f"Observation x: {x_obs.squeeze().cpu().numpy()}")

    # Example 1: Standard posterior p(θ | x)
    print("\n" + "="*50)
    print("Example 1: Standard posterior p(θ | x)")
    print("="*50)

    condition = {
        "x": x_obs.squeeze(),  # Condition on all observations
    }

    posterior_samples = sample_arbitrary_conditional(
        model,
        n_params=task.n_params,
        n_data=task.n_data,
        condition=condition,
        n_samples=1000,
        n_steps=200,
        device=device,
    )

    print(f"Posterior θ mean: {posterior_samples['theta'].mean(dim=0).cpu().numpy()}")
    print(f"True θ: {theta_true.squeeze().cpu().numpy()}")

    # Example 2: Partial conditioning p(θ_rest, x_rest | θ_0, x_0)
    print("\n" + "="*50)
    print("Example 2: Partial conditioning p(θ_{1:4}, x_{1:7} | θ_0, x_0)")
    print("="*50)

    condition = {
        "theta": {0: theta_true[0, 0].item()},  # Fix θ_0
        "x": {0: x_obs[0, 0].item()},           # Fix x_0
    }

    partial_samples = sample_arbitrary_conditional(
        model,
        n_params=task.n_params,
        n_data=task.n_data,
        condition=condition,
        n_samples=1000,
        n_steps=200,
        device=device,
    )

    print(f"Fixed θ_0: {theta_true[0, 0].item():.4f}")
    print(f"Sample θ mean: {partial_samples['theta'].mean(dim=0).cpu().numpy()}")

    # Example 3: Joint likelihood p(θ, x | x_0)
    print("\n" + "="*50)
    print("Example 3: Joint sampling p(θ, x_{1:7} | x_0)")
    print("="*50)

    condition = {
        "x": {0: x_obs[0, 0].item()},  # Only condition on x_0
    }

    joint_samples = sample_arbitrary_conditional(
        model,
        n_params=task.n_params,
        n_data=task.n_data,
        condition=condition,
        n_samples=1000,
        n_steps=200,
        device=device,
    )

    print(f"Fixed x_0: {x_obs[0, 0].item():.4f}")
    print(f"Sample θ mean: {joint_samples['theta'].mean(dim=0).cpu().numpy()}")
    print(f"Sample x mean: {joint_samples['x'].mean(dim=0).cpu().numpy()}")

    # Example 4: Likelihood p(x | θ)
    print("\n" + "="*50)
    print("Example 4: Likelihood sampling p(x | θ)")
    print("="*50)

    condition = {
        "theta": theta_true.squeeze(),  # Condition on all parameters
    }

    likelihood_samples = sample_arbitrary_conditional(
        model,
        n_params=task.n_params,
        n_data=task.n_data,
        condition=condition,
        n_samples=1000,
        n_steps=200,
        device=device,
    )

    print(f"Fixed θ: {theta_true.squeeze().cpu().numpy()}")
    print(f"Sample x mean: {likelihood_samples['x'].mean(dim=0).cpu().numpy()}")
    print(f"True x: {x_obs.squeeze().cpu().numpy()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
