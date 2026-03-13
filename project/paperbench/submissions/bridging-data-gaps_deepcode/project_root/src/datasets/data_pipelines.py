from __future__ import annotations

from typing import Callable, List, Optional, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

__all__ = ["ImageFilesDataset", "get_augmentation_transforms", "make_10shot_loader", "collate_tenshot"]


class ImageFilesDataset(Dataset):
    """A lightweight dataset that loads images from disk and applies a transform.

    This is a minimal utility to wrap a list of image file paths. It expects each
    item to be an image (RGB). If a transform is provided, it will be applied to
    the PIL image before returning. By default, the item is returned as a tensor if
    the transform ends with ToTensor().
    """

    def __init__(self, image_paths: List[str], transform: Optional[Callable] = None):
        self.image_paths = list(image_paths)
        self.transform = transform if transform is not None else self._default_transform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        path = self.image_paths[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    @staticmethod
    def _default_transform() -> Callable:
        # Basic default: convert to tensor and normalize to [-1, 1]
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def get_augmentation_transforms(size: Optional[int] = None) -> transforms.Compose:
    """Return a lightweight data augmentation pipeline for target samples.

    The transforms operate on PIL Images. The returned composition can be plugged into
    ImageFilesDataset or any other dataset that yields PIL Images.

    Args:
        size: Optional target size to resize images to. If None, defaults to 256.
    """
    target_size = int(size) if size is not None else 256
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.Resize(target_size),  # Optional: ensure output size consistency
        transforms.ToTensor(),  # Return tensor; downstream code can further normalize if needed
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def collate_tenshot(batch: List[Any]) -> Dict[str, torch.Tensor]:
    """Collate function supporting 10-shot style batches.

    This collate function is designed to work with the TenShotTargetDataset API used
    in the project. It gracefully handles two common patterns:
    - Batch items are tensors representing a single image: stacks into [B, ...]
    - Batch items are dicts containing a key 'shots' with a tensor of shape
      [n_shots, C, H, W]; stacks into [B, n_shots, C, H, W]. Other keys are preserved
      as lists per-batch element.

    Returns a dictionary with a unified structure to ease downstream training loops.
    """
    if len(batch) == 0:
        return {}

    first = batch[0]
    # Case: items are dicts containing 'shots'
    if isinstance(first, dict) and "shots" in first:
        shots_list = [item["shots"] for item in batch]
        shots_tensor = torch.stack(shots_list, dim=0)  # [B, n_shots, C, H, W]
        out: Dict[str, torch.Tensor] = {"shots": shots_tensor}
        # append any per-item auxiliary fields as lists
        for key in first.keys():
            if key == "shots":
                continue
            out[key] = [item.get(key) for item in batch]
        return out

    # Case: items are tensors (images, or features)
    if isinstance(first, torch.Tensor):
        return {"images": torch.stack(batch, dim=0)}

    # Case: items are tuples/lists of tensors (e.g., (image, label))
    if isinstance(first, (tuple, list)):
        transposed = list(zip(*batch))  # tuple of lists per position
        stacked = [torch.stack(items, dim=0) for items in transposed]
        return {f"item_{i}": t for i, t in enumerate(stacked)}

    # Fallback: try to convert batch to tensor
    try:
        return {"items": torch.tensor(batch)}
    except Exception:
        # Last resort: return as-is in a dict
        return {"batch": batch}


def make_10shot_loader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader suitable for 10-shot style datasets.

    The returned DataLoader uses collate_tenshot to preserve the 10-shot dimension
    across the batch. If the provided dataset already returns 10-shot samples via
    __getitem__, this loader will transparently batch them.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tenshot,
        pin_memory=True
    )
