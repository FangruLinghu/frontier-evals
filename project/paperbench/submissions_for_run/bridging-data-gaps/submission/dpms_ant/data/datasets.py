"""
Dataset utilities for DPMs-ANT.

Handles loading few-shot target domain images and source domain images.

Datasets from the paper:
- Source: FFHQ 256x256, LSUN Church 256x256
- Target: 10-shot Babies, Sunglasses, Raphael, Sketches, Modigliani,
          Haunted Houses, Landscape drawings
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Tuple, List
from pathlib import Path


class FewShotDataset(Dataset):
    """
    Dataset for few-shot image generation.

    Loads images from a directory, applies transforms, and
    returns normalized images in [-1, 1].

    Args:
        root: Directory containing images
        image_size: Target image size
        max_images: Maximum number of images to load (None = all)
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        max_images: Optional[int] = None,
    ):
        self.root = root
        self.image_size = image_size

        # Find all image files
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if Path(f).suffix.lower() in valid_extensions
        ])

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

    def load_all(self) -> torch.Tensor:
        """Load all images as a single tensor."""
        images = [self[i] for i in range(len(self))]
        return torch.stack(images)


def load_target_images(
    target_dir: str,
    image_size: int = 256,
) -> torch.Tensor:
    """
    Load target domain few-shot images.

    Args:
        target_dir: Directory with target images
        image_size: Image size

    Returns:
        Tensor of images (N, C, H, W) in [-1, 1]
    """
    dataset = FewShotDataset(target_dir, image_size)
    print(f"Loaded {len(dataset)} target images from {target_dir}")
    return dataset.load_all()


def load_source_images(
    source_dir: str,
    image_size: int = 256,
    max_images: int = 1000,
) -> torch.Tensor:
    """
    Load source domain images for classifier training.

    Args:
        source_dir: Directory with source images
        image_size: Image size
        max_images: Max number of source images to load

    Returns:
        Tensor of images (N, C, H, W) in [-1, 1]
    """
    dataset = FewShotDataset(source_dir, image_size, max_images)
    print(f"Loaded {len(dataset)} source images from {source_dir}")
    return dataset.load_all()


def save_images(
    images: torch.Tensor,
    output_dir: str,
    prefix: str = "sample",
    nrow: int = 8,
):
    """
    Save generated images to disk.

    Args:
        images: Generated images (N, C, H, W) in [-1, 1]
        output_dir: Output directory
        prefix: Filename prefix
        nrow: Number of images per row for grid
    """
    from torchvision.utils import save_image, make_grid

    os.makedirs(output_dir, exist_ok=True)

    # Save individual images
    for i, img in enumerate(images):
        img_pil = transforms.ToPILImage()((img + 1) / 2)  # [-1,1] -> [0,1]
        img_pil.save(os.path.join(output_dir, f"{prefix}_{i:04d}.png"))

    # Save grid
    grid = make_grid((images + 1) / 2, nrow=nrow, padding=2)
    save_image(grid, os.path.join(output_dir, f"{prefix}_grid.png"))


def create_synthetic_target(
    n_images: int = 10,
    image_size: int = 256,
    style: str = "colored_noise",
) -> torch.Tensor:
    """
    Create synthetic target images for testing.

    Args:
        n_images: Number of images
        image_size: Image resolution
        style: Type of synthetic images

    Returns:
        Tensor of synthetic images
    """
    if style == "colored_noise":
        # Simple colored noise patterns
        images = []
        for i in range(n_images):
            img = torch.randn(3, image_size, image_size) * 0.3
            # Add a color bias
            color = torch.rand(3, 1, 1) * 0.5
            img = img + color
            img = img.clamp(-1, 1)
            images.append(img)
        return torch.stack(images)
    elif style == "gradients":
        images = []
        for i in range(n_images):
            angle = np.random.uniform(0, 2 * np.pi)
            x = torch.linspace(-1, 1, image_size)
            y = torch.linspace(-1, 1, image_size)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            gradient = torch.cos(angle) * xx + torch.sin(angle) * yy
            img = gradient.unsqueeze(0).repeat(3, 1, 1)
            # Add random color
            color = torch.rand(3, 1, 1) * 0.5
            img = (img * 0.5 + color).clamp(-1, 1)
            images.append(img)
        return torch.stack(images)
    else:
        return torch.randn(n_images, 3, image_size, image_size).clamp(-1, 1)
