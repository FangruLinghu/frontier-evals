#!/usr/bin/env python
"""
Download pre-trained models for DPMs-ANT.

Models needed:
1. Pre-trained DDPM on FFHQ 256x256
   - From guided-diffusion (Dhariwal & Nichol, 2021)
   - Used as source model for transfer learning

2. Pre-trained DDPM on LSUN Church 256x256
   - From guided-diffusion
   - Used as source model for church-related transfers

3. Pre-trained classifier on ImageNet 256x256
   - From guided-diffusion
   - Used as backbone for the binary classifier

Note: These are the models from OpenAI's guided-diffusion project.
The FFHQ model may need to be sourced from DDPM-PA or trained.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Model URLs from guided-diffusion (OpenAI)
# https://github.com/openai/guided-diffusion
MODEL_URLS = {
    # 256x256 classifier for ImageNet (used as backbone)
    "imagenet_256_classifier": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt",
        "256x256_classifier.pt",
    ),
    # 256x256 unconditional diffusion model (ImageNet)
    "imagenet_256_diffusion": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt",
        "256x256_diffusion.pt",
    ),
    # LSUN bedroom 256x256 (closest available for church)
    "lsun_bedroom_256": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt",
        "lsun_bedroom.pt",
    ),
}

# Note on FFHQ model:
# The guided-diffusion project doesn't provide an FFHQ checkpoint.
# For FFHQ, you have two options:
# 1. Use the model from DDPM-PA (Zhu et al., 2022) which provides FFHQ checkpoints
# 2. Train from scratch on FFHQ using guided-diffusion code
# 3. Use an available FFHQ DDPM checkpoint from other sources
#
# DDPM (Ho et al., 2020) checkpoints for FFHQ can be found at:
# https://github.com/pesser/pytorch_diffusion
# Note: This uses a different architecture (original DDPM, not improved DDPM)


def download_file(url: str, output_path: str):
    """Download a file with progress reporting."""
    print(f"Downloading: {url}")
    print(f"  -> {output_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print("\n  Done!")


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_URLS.keys()) + ["all"],
        default=["all"],
        help="Models to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/pretrained",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_download = list(MODEL_URLS.keys()) if "all" in args.models else args.models

    for model_name in models_to_download:
        url, filename = MODEL_URLS[model_name]
        output_path = output_dir / filename

        if output_path.exists():
            print(f"Skipping {model_name}: already exists at {output_path}")
            continue

        try:
            download_file(url, str(output_path))
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            print("You may need to download manually.")

    print("\n--- Setup Notes ---")
    print("For FFHQ pre-trained DDPM, you need to obtain the checkpoint separately.")
    print("Options:")
    print("  1. Use pytorch_diffusion FFHQ checkpoint:")
    print("     https://github.com/pesser/pytorch_diffusion")
    print("  2. Train on FFHQ yourself using this codebase")
    print("  3. Use the ImageNet model as a starting point")
    print()
    print("For 10-shot target datasets, place images in:")
    print("  datasets/<target_name>/  (e.g., datasets/sunglasses/)")


if __name__ == "__main__":
    main()
