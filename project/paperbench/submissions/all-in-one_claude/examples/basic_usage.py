#!/usr/bin/env python
"""
Basic usage example for Simformer.

This script demonstrates:
1. Creating a task
2. Training a Simformer model
3. Sampling from the posterior
4. Evaluating the samples
"""

import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simformer import Simformer, VESDE, sample_posterior
from simformer.training import Trainer, TrainingConfig
from simformer.tasks import TwoMoonsTask
from simformer.evaluation import c2st, expected_coverage
from simformer.utils import get_device, set_seed


def main():
    # Set seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # 1. Create task
    print("\n1. Creating Two Moons task...")
    task = TwoMoonsTask()
    print(f"   Parameters: {task.n_params}, Data: {task.n_data}")

    # 2. Create model
    print("\n2. Creating Simformer model...")
    model = Simformer(
        n_params=task.n_params,
        n_data=task.n_data,
        token_dim=50,
        n_layers=6,
        n_heads=4,
        widening_factor=3,
        sde=VESDE(),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {n_params:,}")

    # 3. Train model
    print("\n3. Training model...")
    config = TrainingConfig(
        epochs=20,  # Use more epochs for better results (100+)
        batch_size=1000,
        learning_rate=1e-4,
        device=device,
    )
    trainer = Trainer(model, config)

    for epoch in range(config.epochs):
        # Generate training data
        theta = task.sample_prior(10000).to(device)
        x = task.simulate(theta).to(device)

        # Train
        metrics = trainer.train_epoch(theta, x)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch + 1}: Loss = {metrics['loss']:.6f}")

    # 4. Generate test observation
    print("\n4. Generating test observation...")
    theta_true = task.sample_prior(1).to(device)
    x_obs = task.simulate(theta_true).to(device)
    print(f"   True θ: {theta_true.squeeze().cpu().numpy()}")
    print(f"   Observation x: {x_obs.squeeze().cpu().numpy()}")

    # 5. Sample posterior
    print("\n5. Sampling from posterior...")
    n_samples = 1000
    model.eval()
    with torch.no_grad():
        posterior_samples = sample_posterior(
            model,
            x_obs,
            n_samples=n_samples,
            n_steps=500,
        )
    print(f"   Generated {n_samples} posterior samples")
    print(f"   Posterior mean: {posterior_samples.mean(dim=0).cpu().numpy()}")
    print(f"   Posterior std: {posterior_samples.std(dim=0).cpu().numpy()}")

    # 6. Evaluate
    print("\n6. Evaluating samples...")

    # Compare with prior samples (simple baseline)
    prior_samples = task.sample_prior(n_samples).to(device)
    c2st_score, c2st_std = c2st(posterior_samples, prior_samples)
    print(f"   C2ST (vs prior): {c2st_score:.4f} ± {c2st_std:.4f}")
    print("   (Closer to 1.0 means samples are distinguishable from prior)")

    # 7. Visualize (if matplotlib available)
    print("\n7. Creating visualization...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Prior samples
        prior_np = prior_samples.cpu().numpy()
        axes[0].scatter(prior_np[:, 0], prior_np[:, 1], alpha=0.3, s=5, label="Prior")
        axes[0].scatter([theta_true[0, 0].item()], [theta_true[0, 1].item()],
                       color="red", s=100, marker="*", label="True θ")
        axes[0].set_xlabel("θ₁")
        axes[0].set_ylabel("θ₂")
        axes[0].set_title("Prior p(θ)")
        axes[0].legend()

        # Posterior samples
        post_np = posterior_samples.cpu().numpy()
        axes[1].scatter(post_np[:, 0], post_np[:, 1], alpha=0.3, s=5, label="Posterior")
        axes[1].scatter([theta_true[0, 0].item()], [theta_true[0, 1].item()],
                       color="red", s=100, marker="*", label="True θ")
        axes[1].set_xlabel("θ₁")
        axes[1].set_ylabel("θ₂")
        axes[1].set_title("Posterior p(θ|x)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("posterior_samples.png", dpi=150)
        print("   Saved visualization to posterior_samples.png")

    except Exception as e:
        print(f"   Visualization skipped: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
