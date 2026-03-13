## data.py
"""
Data handling module for DPMs-ANT reproduction.

This module provides a deterministic data pipeline to support 10-shot
transfer tasks between source and target domains as described in the paper.

Key components:
- SimpleImageDataset: generic image dataset with a transform in CPU/PIL space.
- TenShotTargetDataset: tiny target exemplar dataset (10-shot) that can be
  wrapped by a DataLoader to yield batches larger than 10 via cyclic indexing.
- DataLoaderManager: orchestrates loading source domains, deterministic 10-shot
  target exemplars, and provides DataLoader instances for both source and target
  data. It uses a transform that converts PIL images to tensors in [-1, 1] range.

Notes:
- No external configuration parsing is performed here; the manager accepts
  paths and hyperparameters via initialization and/or explicit method calls.
- Image loading is implemented with Pillow; no torchvision dependency is required.
- The 10-shot target loader is designed to fill large batch sizes by cycling over
  the 10 exemplar images (via a large __len__ on the dataset).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import os
import random
import math
import hashlib
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler


#############################
# Utilities for image data  #
#############################

def _collect_images(root_dir: str, extensions: Optional[List[str]] = None) -> List[str]:
    """Recursively collect image file paths from root_dir with given extensions."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in extensions):
                paths.append(os.path.join(dirpath, fname))
    paths.sort()  # deterministic order
    return paths


def _default_transform(image_size: int) -> Callable:
    """Return a transform that resizes, center-crops, and normalizes PIL images
    to [-1, 1] tensor with shape (C, H, W)."""

    def _transform(img: Image.Image) -> torch.Tensor:
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to a size that preserves aspect ratio, then center-crop to image_size
        w, h = img.size
        scale = image_size / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - image_size) // 2
        top = (new_h - image_size) // 2
        img = img.crop((left, top, left + image_size, top + image_size))

        # To tensor and normalize to [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [C, H, W] after transpose
        arr = arr.transpose(2, 0, 1)  # to (C, H, W)
        return torch.from_numpy(arr)

    return _transform


#############################
# Dataset wrappers          #
#############################


class SimpleImageDataset(Dataset):
    """A minimal image dataset that returns (image_tensor, domain_label)."""

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        domain_label: int = 0,
    ) -> None:
        """
        Args:
            image_paths: list of image file paths.
            transform: callable to convert PIL.Image to a tensor.
            domain_label: label indicating domain (e.g., 0 for FFHQ, 1 for LSUN Church).
        """
        self.image_paths = image_paths
        self.transform = transform
        self.domain_label = domain_label

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx % len(self.image_paths)]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                # Fallback: simple tensor conversion
                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                img = torch.from_numpy(arr).permute(2, 0, 1)
        return img, self.domain_label


class TenShotTargetDataset(Dataset):
    """Tiny target exemplar dataset (assumes 10 images)."""

    def __init__(self, paths: List[str], transform: Optional[Callable] = None, length: int = 1000) -> None:
        """
        Args:
            paths: list of 10 exemplar image paths.
            transform: image transform to apply.
            length: logical length of the dataset; used to allow large batch sizes via cycling.
        """
        self.paths = paths
        self.transform = transform
        self._length = max(1, int(length))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        idx0 = idx % len(self.paths)
        path = self.paths[idx0]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                img = torch.from_numpy(arr).permute(2, 0, 1)
        # Return a dummy label since some training loops expect (image, label)
        return img, 0


#############################
# DataLoader manager        #
#############################


class DataLoaderManager:
    """Manager for source and target data loaders for 10-shot diffusion training.

    This class does not perform model training; it only prepares deterministic
    data pipelines that can be consumed by the trainer.
    """

    def __init__(
        self,
        sources_dirs: List[str],
        targets_map: Dict[str, str],
        shot_count: int = 10,
        global_seed: int = 0,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        exemplar_length: int = 1000,
    ) -> None:
        """
        Args:
            sources_dirs: list of directories containing source-domain images (FFHQ, LSUN_Church, etc.).
            targets_map: mapping from task_id string to target-domain root directory.
            shot_count: number of target exemplars to use per task (default 10).
            global_seed: global seed for reproducibility.
            image_size: square image size to transform images to.
            transform: optional prebuilt transform; if None, a default transform is created.
            exemplar_length: internal length for the TenShotTargetDataset (to enable large batch sizes).
        """
        self.sources_dirs = sources_dirs
        self.targets_map = targets_map
        self.shot_count = int(shot_count)
        self.global_seed = int(global_seed)
        self.image_size = int(image_size)
        self.transform = transform or _default_transform(self.image_size)
        self.exemplar_length = int(exemplar_length)

        # Internal caches to avoid reloading datasets
        self._loaded_sources: Optional[List[SimpleImageDataset]] = None
        self._target_datasets: Dict[str, TenShotTargetDataset] = {}

    # Public API

    def load_source_data(self) -> List[SimpleImageDataset]:
        """Load and return source-domain datasets as a list of SimpleImageDataset.
        Each dataset is paired with a domain_label (0..n-1)."""
        if self._loaded_sources is not None:
            return self._loaded_sources

        datasets: List[SimpleImageDataset] = []
        for i, dir_path in enumerate(self.sources_dirs):
            image_paths = _collect_images(dir_path)
            ds = SimpleImageDataset(image_paths=image_paths, transform=self.transform, domain_label=i)
            datasets.append(ds)

        if len(datasets) == 0:
            raise RuntimeError("No source datasets found. Check sources_dirs.")

        self._loaded_sources = datasets
        return datasets

    def prepare_target_10shot(self, task_id: str) -> TenShotTargetDataset:
        """Deterministically prepare a 10-shot target exemplar dataset for a given task.

        The 10 images are chosen from the target directory with a fixed seed derived
        from global_seed and task_id to ensure reproducibility across runs.

        Returns:
            TenShotTargetDataset: a tiny dataset with 10 exemplars.
        """
        if task_id in self._target_datasets:
            return self._target_datasets[task_id]

        if task_id not in self.targets_map:
            raise KeyError(f"Task id '{task_id}' not found in targets_map.")

        target_root = self.targets_map[task_id]
        all_paths = _collect_images(target_root)
        if len(all_paths) < max(self.shot_count, 1):
            raise ValueError(
                f"Target root '{target_root}' does not contain enough images for 10-shot sampling (found {len(all_paths)})."
            )

        # Deterministic seed per task
        seed = self._seed_for_task(task_id)
        rnd = random.Random(seed)
        # Deterministically pick shot_count samples, then sort for stability
        chosen = sorted(rnd.sample(all_paths, self.shot_count))

        dataset = TenShotTargetDataset(paths=chosen, transform=self.transform, length=self.exemplar_length)
        self._target_datasets[task_id] = dataset
        return dataset

    def get_source_loader(self, batch_size: int) -> DataLoader:
        """Create a DataLoader that yields batches by balancing two (or more) source datasets.

        If there is only one source dataset, a simple DataLoader is returned.
        If two or more sources are present, a balanced sampling strategy is used
        via a WeightedRandomSampler over a ConcatDataset to ensure approximately
        equal contributions from each domain in every batch.
        """
        ds_list = self.load_source_data()

        if len(ds_list) == 1:
            ds = ds_list[0]
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

        # Build a ConcatDataset and a per-sample weighting to balance domains
        concat_ds = ConcatDataset(ds_list)

        # Build weights so each domain contributes proportionally to 1/len(ds)
        weights: List[float] = []
        for ds in ds_list:
            w = 1.0 / max(1, len(ds))
            weights.extend([w] * len(ds))

        sampler = WeightedRandomSampler(weights, num_samples=batch_size, replacement=True)

        loader = DataLoader(
            concat_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )
        return loader

    def get_target_loader(self, task_id: str, batch_size: int, split: str = "train") -> DataLoader:
        """Get a DataLoader for the target 10-shot exemplars.

        The dataset is constructed lazily and then wrapped by a DataLoader.
        The loader is designed to support large batch sizes by cycling through
        the 10 exemplars (via TenShotTargetDataset length > 10).
        """
        dataset = self.prepare_target_10shot(task_id)

        # If needed, we could support different splits ('train'/'val') by using
        # separate deterministic exemplar sets; for simplicity, we reuse the same 10-shot set.
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )

    # Helpers

    def _seed_for_task(self, task_id: str) -> int:
        """Derive a stable integer seed from the global seed and the task id."""
        # Use a hash function to derive a 32-bit seed from (global_seed, task_id)
        to_hash = f"{self.global_seed}:{task_id}"
        digest = hashlib.sha256(to_hash.encode("utf-8")).hexdigest()
        seed = int(digest, 16) % (2**32)
        return max(0, seed)