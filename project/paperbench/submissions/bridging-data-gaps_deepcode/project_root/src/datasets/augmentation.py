"""
Lightweight dataset augmentation utilities for 10-shot diffusion-model training.

This module provides a single helper:
- get_augmentation_transforms(size=None): returns a torchvision transforms.Compose
  pipeline that applies common augmentations suitable for image-based diffusion
  tasks and outputs tensors normalized to [-1, 1].
"""
from typing import Optional

try:
    from torchvision import transforms
except Exception:  # pragma: no cover - optional dependency handling
    transforms = None  # type: ignore


def get_augmentation_transforms(size: Optional[int] = None):
    """Return a torchvision augmentation pipeline.

    The pipeline performs a random crop/resize, horizontal flip, color jitter,
    followed by conversion to a tensor and normalization to [-1, 1].

    Args:
        size: Target image size (height and width). If None, defaults to 128.

    Returns:
        A torchvision.transforms.Compose object implementing the augmentation.
    """
    if transforms is None:
        raise ImportError("torchvision is required for augmentation transforms.")

    target_size = int(size) if size is not None else 128

    # Compose common augmentations for robust target-domain reception
    pipeline = transforms.Compose(
        [
            transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.ToTensor(),  # converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # maps to [-1, 1]
        ]
    )
    return pipeline


__all__ = ["get_augmentation_transforms"]
