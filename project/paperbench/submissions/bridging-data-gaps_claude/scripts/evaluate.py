#!/usr/bin/env python
"""
Evaluation script for DPMs-ANT.

Computes FID and Intra-LPIPS metrics on generated images.

Usage:
    python scripts/evaluate.py \
        --model-checkpoint results/sunglasses_ant/checkpoints/final.pt \
        --target-dir datasets/sunglasses \
        --reference-dir datasets/sunglasses_full \
        --n-generated 1000
"""

import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from dpms_ant.models.unet import create_ffhq256_model
from dpms_ant.adaptor.adaptor import UNetWithAdaptor
from dpms_ant.diffusion.gaussian_diffusion import create_diffusion
from dpms_ant.evaluation.metrics import IntraLPIPS, FIDCalculator, evaluate_model
from dpms_ant.data.datasets import load_target_images, save_images


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DPMs-ANT")

    parser.add_argument("--model-checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--target-dir", type=str, required=True, help="Target domain images")
    parser.add_argument("--reference-dir", type=str, default=None, help="Larger reference set for FID")
    parser.add_argument("--pretrained-model", type=str, default=None, help="Pre-trained base model")

    parser.add_argument("--n-generated", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--ddim-steps", type=int, default=50, help="DDIM sampling steps")

    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-images", action="store_true", help="Save generated images")

    return parser.parse_args()


def get_device(device_str=None):
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load target images
    print(f"Loading target images from {args.target_dir}...")
    target_images = load_target_images(args.target_dir, args.image_size)

    # Load reference images for FID (if available)
    reference_images = None
    if args.reference_dir and os.path.exists(args.reference_dir):
        print(f"Loading reference images from {args.reference_dir}...")
        reference_images = load_target_images(args.reference_dir, args.image_size)
        print(f"  Loaded {len(reference_images)} reference images")

    # Create model
    print("Creating model...")
    unet = create_ffhq256_model()

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        checkpoint = torch.load(args.pretrained_model, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            unet.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            unet.load_state_dict(checkpoint["state_dict"])
        else:
            unet.load_state_dict(checkpoint)

    model = UNetWithAdaptor(unet).to(device)

    # Load checkpoint (adaptor weights)
    print(f"Loading checkpoint from {args.model_checkpoint}...")
    checkpoint = torch.load(args.model_checkpoint, map_location=device)

    if "adaptor_state_dict" in checkpoint:
        model_dict = model.state_dict()
        for name, param in checkpoint["adaptor_state_dict"].items():
            if name in model_dict:
                model_dict[name] = param
        model.load_state_dict(model_dict)
    else:
        # Try loading full state dict
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # Create diffusion
    diffusion = create_diffusion()

    # Generate images
    print(f"\nGenerating {args.n_generated} images (DDIM {args.ddim_steps} steps)...")
    all_samples = []

    for i in tqdm(range(0, args.n_generated, args.batch_size)):
        bs = min(args.batch_size, args.n_generated - i)
        shape = (bs, 3, args.image_size, args.image_size)

        with torch.no_grad():
            samples = diffusion.ddim_sample(
                model, shape, device,
                ddim_steps=args.ddim_steps,
                progress=False,
            )
        all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=0)

    # Save generated images if requested
    if args.save_images:
        save_images(all_samples, str(output_dir / "generated"), prefix="sample")

    # Compute metrics
    results = {}

    # Intra-LPIPS
    print("\nComputing Intra-LPIPS...")
    intra_lpips = IntraLPIPS(device=device)
    mean_ilpips, std_ilpips = intra_lpips.compute(
        all_samples.to(device), target_images.to(device),
        batch_size=args.batch_size,
    )
    results["intra_lpips_mean"] = mean_ilpips
    results["intra_lpips_std"] = std_ilpips
    print(f"  Intra-LPIPS: {mean_ilpips:.3f} +/- {std_ilpips:.3f}")

    # FID
    if reference_images is not None:
        print("Computing FID...")
        fid_calc = FIDCalculator(device=device)
        fid = fid_calc.compute_fid(all_samples, reference_images, batch_size=args.batch_size)
        results["fid"] = fid
        print(f"  FID: {fid:.2f}")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)
    print(f"  Intra-LPIPS: {results['intra_lpips_mean']:.3f} +/- {results['intra_lpips_std']:.3f}")
    if "fid" in results:
        print(f"  FID: {results['fid']:.2f}")
    print()

    # Paper reference values (FFHQ -> Sunglasses)
    print("Paper reference (FFHQ -> Sunglasses):")
    print("  DDPM-PA:   Intra-LPIPS=0.604, FID=34.75")
    print("  DDPM-ANT:  Intra-LPIPS=0.613, FID=20.06")
    print("  LDM-ANT:   Intra-LPIPS=0.613, FID=N/A")


if __name__ == "__main__":
    main()
