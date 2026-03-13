#!/usr/bin/env python
"""
Main training script for DPMs-ANT few-shot image generation.

Reproduces the experiments from the paper:
- FFHQ -> Babies, Sunglasses, Raphael's paintings, Sketches, Modigliani
- LSUN Church -> Haunted Houses, Landscape drawings

Usage:
    # Full DPMs-ANT training
    python scripts/train.py \
        --target-dir datasets/sunglasses \
        --source-dir datasets/ffhq \
        --pretrained-model checkpoints/pretrained/ffhq_ddpm.pt \
        --method ant \
        --iterations 300

    # Baseline fine-tuning
    python scripts/train.py \
        --target-dir datasets/sunglasses \
        --pretrained-model checkpoints/pretrained/ffhq_ddpm.pt \
        --method baseline \
        --iterations 5000

    # ANT without adversarial noise (similarity-guided only)
    python scripts/train.py \
        --target-dir datasets/sunglasses \
        --pretrained-model checkpoints/pretrained/ffhq_ddpm.pt \
        --method ant_no_an \
        --iterations 300
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from dpms_ant.models.unet import UNetModel, create_ffhq256_model
from dpms_ant.adaptor.adaptor import UNetWithAdaptor
from dpms_ant.classifier.noise_classifier import NoisyImageClassifier, train_binary_classifier
from dpms_ant.diffusion.gaussian_diffusion import GaussianDiffusion, create_diffusion
from dpms_ant.training import ANTTrainer, BaselineTrainer
from dpms_ant.data.datasets import load_target_images, load_source_images, save_images


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPMs-ANT")

    # Data
    parser.add_argument("--target-dir", type=str, required=True, help="Target domain images directory")
    parser.add_argument("--source-dir", type=str, default=None, help="Source domain images directory (for classifier)")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution")

    # Model
    parser.add_argument("--pretrained-model", type=str, default=None, help="Path to pre-trained DDPM checkpoint")
    parser.add_argument("--model-channels", type=int, default=128, help="U-Net base channels")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="ResBlocks per level")

    # Method
    parser.add_argument("--method", type=str, default="ant",
                        choices=["ant", "ant_no_an", "baseline", "adaptor_only"],
                        help="Training method")

    # ANT hyperparameters
    parser.add_argument("--gamma", type=float, default=5.0, help="Similarity guidance scale")
    parser.add_argument("--J", type=int, default=10, help="Adversarial noise steps")
    parser.add_argument("--omega", type=float, default=0.02, help="Adversarial noise step size")

    # Adaptor
    parser.add_argument("--spatial-downscale", type=int, default=4, help="Adaptor spatial downscale (c)")
    parser.add_argument("--bottleneck-dim", type=int, default=8, help="Adaptor bottleneck dim (d)")

    # Training
    parser.add_argument("--iterations", type=int, default=300, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=40, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")

    # Classifier
    parser.add_argument("--classifier-epochs", type=int, default=50, help="Classifier training epochs")
    parser.add_argument("--classifier-lr", type=float, default=1e-4, help="Classifier learning rate")

    # Diffusion
    parser.add_argument("--diffusion-steps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--noise-schedule", type=str, default="linear", help="Noise schedule")

    # Output
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="Checkpoint frequency")

    # Hardware
    parser.add_argument("--device", type=str, default=None, help="Device (auto if None)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def get_device(device_str=None):
    """Get the best available device."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = Path(args.target_dir).name
    output_dir = Path(args.output_dir) / f"{target_name}_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load target images
    print(f"\nLoading target images from {args.target_dir}...")
    target_images = load_target_images(args.target_dir, args.image_size)
    print(f"  Loaded {len(target_images)} target images")

    # Create diffusion
    diffusion = create_diffusion(
        noise_schedule=args.noise_schedule,
        num_diffusion_timesteps=args.diffusion_steps,
    )

    # Create or load model
    print("\nSetting up model...")
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"  Loading pre-trained model from {args.pretrained_model}")
        unet = create_ffhq256_model()
        checkpoint = torch.load(args.pretrained_model, map_location="cpu")
        if "model_state_dict" in checkpoint:
            unet.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            unet.load_state_dict(checkpoint["state_dict"])
        else:
            unet.load_state_dict(checkpoint)
    else:
        print("  Creating new U-Net model (no pre-trained weights)")
        unet = create_ffhq256_model()

    unet = unet.to(device)

    if args.method in ["ant", "ant_no_an", "adaptor_only"]:
        # Wrap with adaptor
        model = UNetWithAdaptor(
            unet,
            spatial_downscale=args.spatial_downscale,
            bottleneck_dim=args.bottleneck_dim,
        ).to(device)
        print(f"  Adaptor parameters: {model.count_adaptor_parameters():,}")
        print(f"  Total parameters: {model.count_total_parameters():,}")
        print(f"  Parameter rate: {model.parameter_rate():.2%}")
    else:
        model = unet

    # Train classifier if needed
    classifier = None
    if args.method in ["ant", "ant_no_an"]:
        print("\nTraining binary classifier...")

        classifier = NoisyImageClassifier(
            image_size=args.image_size,
            in_channels=3,
            model_channels=64,
            channel_mult=(1, 2, 4),
            num_res_blocks=1,
            attention_resolutions=(16, 8),
            num_heads=4,
            num_classes=2,
        ).to(device)

        # Load source images for classifier training
        if args.source_dir:
            source_images = load_source_images(args.source_dir, args.image_size, max_images=500)
        else:
            # Use random noise as proxy for source domain
            print("  Warning: No source images provided. Using noise as source proxy.")
            source_images = torch.randn(100, 3, args.image_size, args.image_size).clamp(-1, 1)

        classifier = train_binary_classifier(
            classifier,
            source_images,
            target_images,
            diffusion,
            epochs=args.classifier_epochs,
            lr=args.classifier_lr,
            device=device,
        )

    # Training
    print(f"\nStarting {args.method} training...")
    print(f"  Iterations: {args.iterations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    if args.method == "ant":
        trainer = ANTTrainer(
            model=model,
            diffusion=diffusion,
            classifier=classifier,
            gamma=args.gamma,
            J=args.J,
            omega=args.omega,
            lr=args.lr,
            device=device,
        )
    elif args.method == "ant_no_an":
        trainer = ANTTrainer(
            model=model,
            diffusion=diffusion,
            classifier=classifier,
            gamma=args.gamma,
            J=0,  # No adversarial noise
            omega=0,
            lr=args.lr,
            device=device,
        )
    elif args.method == "adaptor_only":
        trainer = BaselineTrainer(
            model=model,
            diffusion=diffusion,
            lr=args.lr,
            device=device,
            adaptor_only=True,
        )
    else:  # baseline
        trainer = BaselineTrainer(
            model=model,
            diffusion=diffusion,
            lr=args.lr,
            device=device,
            adaptor_only=False,
        )

    if isinstance(trainer, ANTTrainer):
        losses = trainer.train(
            target_images=target_images,
            n_iterations=args.iterations,
            batch_size=args.batch_size,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=args.checkpoint_freq,
        )
    else:
        losses = trainer.train(
            target_images=target_images,
            n_iterations=args.iterations,
            batch_size=args.batch_size,
        )

    # Save loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(losses, alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Training Loss ({args.method})")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "training_loss.png", dpi=150)

    # Generate samples
    print("\nGenerating samples...")
    model.eval()
    with torch.no_grad():
        samples = diffusion.ddim_sample(
            model,
            shape=(16, 3, args.image_size, args.image_size),
            device=device,
            ddim_steps=50,
            progress=True,
        )

    save_images(samples, str(output_dir / "generated"), prefix="sample")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
