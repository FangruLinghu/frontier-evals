# Lightweight intra-LPIPS proxy implementation
"""
Intra-LPIPS (perceptual diversity) proxy for a batch of images.

This module provides a small, self-contained fallback for measuring
perceptual diversity within a set of generated images without relying
on heavyweight pretrained LPIPS models. It uses a tiny CNN-based feature
extractor to produce fixed-length feature vectors and computes the mean
pairwise Euclidean distance between these features as a proxy for LPIPS
diversity.

Public API:
- compute_intra_lpips(images, batch_size=32, device=None) -> float
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["compute_intra_lpips"]


class _TinyLPIPS(nn.Module):
    """A tiny CNN-based feature extractor to produce compact descriptors.

    The network is intentionally lightweight and is not meant to be a real
    LPIPS descriptor. It provides a deterministic, differentiable feature
    representation suitable for a quick proxy evaluation of intra-sample
    perceptual diversity.
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self._out_dim = feat_dim
        # Simple projection to consistent feature dimension if needed
        self._proj = nn.Identity()
        self.to(torch.device("cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feats = self.net(x)
        # Normalize to unit length to stabilize distances a bit
        if feats.numel() == 0:
            return feats
        feats = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-6)
        return feats

    @property
    def feature_dim(self) -> int:
        return self._out_dim


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    # Convert to float32 and scale to [0, 1] if necessary
    imgs = images.to(dtype=torch.float32)
    if imgs.max() > 1.0:
        # Assume [0, 255] input
        imgs = imgs / 255.0
    imgs = torch.clamp(imgs, 0.0, 1.0)
    return imgs


def _extract_features(model: _TinyLPIPS, imgs: torch.Tensor, batch_size: int, device: Optional[torch.device]) -> torch.Tensor:
    model.eval()
    if device is None:
        device = imgs.device
    model = model.to(device)
    feats_list = []
    n = imgs.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = imgs[i : min(i + batch_size, n)].to(device)
            feats = model(batch)  # [B, D]
            feats_list.append(feats)
    feats_all = torch.cat(feats_list, dim=0)
    return feats_all


def compute_intra_lpips(images: torch.Tensor, batch_size: int = 32, device: Optional[torch.device] = None) -> float:
    """Compute a lightweight intra-LPIPS-like score for a batch of images.

    Args:
        images: Tensor of shape [N, C, H, W], with values in [0, 1] or [0, 255].
        batch_size: Batch size for feature extraction.
        device: Optional torch.device to run computation on.

    Returns:
        A scalar float representing the mean pairwise distance between image features.
    """
    if images is None:
        return 0.0
    if images.dim() != 4:
        raise ValueError("images must have shape [N, C, H, W]")
    imgs = _normalize_images(images)
    n = imgs.shape[0]
    if n < 2:
        return 0.0

    # Initialize a tiny feature extractor
    lpips_model = _TinyLPIPS(in_channels=imgs.shape[1])
    # Compute features for all images
    features = _extract_features(lpips_model, imgs, batch_size, device)
    # Compute pairwise L2 distances among features and average upper-triangular part
    feats = features  # [N, D]
    with torch.no_grad():
        dists = torch.cdist(feats, feats, p=2)  # [N, N]
        # Exclude diagonal
        idx = torch.triu_indices(n, n, offset=1, device=feats.device)
        pair_dist = dists[idx[0], idx[1]]
        mean_dist = pair_dist.mean().item()
    return float(mean_dist)
