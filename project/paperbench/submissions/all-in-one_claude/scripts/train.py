#!/usr/bin/env python
"""
Training script for Simformer.

Usage:
    python scripts/train.py --task two_moons --epochs 100
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simformer import Simformer
from simformer.training import Trainer, TrainingConfig
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
    parser = argparse.ArgumentParser(description="Train Simformer")

    # Task
    parser.add_argument(
        "--task",
        type=str,
        default="two_moons",
        choices=list(TASK_REGISTRY.keys()),
        help="Task to train on",
    )

    # Model
    parser.add_argument("--token_dim", type=int, default=50, help="Token dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--widening_factor", type=int, default=3, help="MLP widening factor")
    parser.add_argument("--time_embed_dim", type=int, default=128, help="Time embedding dimension")
    parser.add_argument("--sde", type=str, default="vesde", choices=["vesde", "vpsde"], help="SDE type")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--samples_per_epoch", type=int, default=100000, help="Samples per epoch")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Checkpoint frequency (epochs)")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency (batches)")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_task(task_name: str, **kwargs):
    """Create task instance."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_REGISTRY[task_name](**kwargs)


def create_model(task, args, device):
    """Create Simformer model."""
    from simformer.diffusion import VESDE, VPSDE

    # Get dependency structure
    dep_structure = task.get_dependency_structure()

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
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        widening_factor=args.widening_factor,
        time_embed_dim=args.time_embed_dim,
        sde=sde,
    )

    return model.to(device)


def create_dataloader(task, batch_size: int, n_samples: int):
    """Create dataloader for task."""
    from torch.utils.data import DataLoader, TensorDataset

    # Generate training data
    theta = task.sample_prior(n_samples)
    x = task.simulate(theta)

    dataset = TensorDataset(theta, x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    args = parse_args()

    # Load config if provided
    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Using device: {device}")

    # Create task
    print(f"Creating task: {args.task}")
    task = create_task(args.task)
    print(f"  n_params: {task.n_params}, n_data: {task.n_data}")

    # Create model
    print("Creating model...")
    model = create_model(task, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.task
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create training config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        gradient_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_frequency=args.checkpoint_freq,
        log_frequency=args.log_freq,
        device=device,
    )

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(model, config)

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Samples per epoch: {args.samples_per_epoch}")
    print()

    for epoch in range(trainer.current_epoch, args.epochs):
        # Generate fresh data each epoch
        theta = task.sample_prior(args.samples_per_epoch).to(device)
        x = task.simulate(theta).to(device)

        # Train epoch
        metrics = trainer.train_epoch(theta, x)

        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {metrics['loss']:.6f}")

        # Checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            trainer.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

    # Save final model
    trainer.save_checkpoint("final_model.pt")
    print(f"\nTraining complete! Model saved to {checkpoint_dir}/final_model.pt")


if __name__ == "__main__":
    main()
