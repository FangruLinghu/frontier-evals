"""eval_loop.py - lightweight evaluation harness for DPMs-ANT components.

This module provides a minimal, self-contained evaluation runner that can be
executed without requiring full diffusion backbones or heavy datasets. It is
intended as a wiring-validation tool and demonstration harness that composes a
frozen backbone (DummyBackbone) with a toy evaluation loop, returning simple
FID/Intra-LPIPS-like metrics for sanity checks.

Usage:
  python -m src.eval_loop  # uses default seed
"""

from __future__ import annotations

import math
import random
from typing import Optional, Callable, Dict, Any

import torch
import numpy as np


def _safe_imports() -> tuple:
    """Attempt lightweight imports of ANT trainer components if available.

    Returns a tuple of (ANTTrainerClass, build_adaptors_for_backbone) where
    each may be None if the import fails. This keeps eval_loop.py usable even
    when heavier dependencies are not installed in the execution environment.
    """
    ANTTrainer = None
    build_adaptors_for_backbone = None
    try:
        from training.ant_trainer import ANTTrainer  # type: ignore
        ANTTrainer = ANTTrainer
    except Exception:
        ANTTrainer = None

    try:
        from adaptor.adaptor import build_adaptors_for_backbone  # type: ignore
        build_adaptors_for_backbone = build_adaptors_for_backbone
    except Exception:
        build_adaptors_for_backbone = None

    return ANTTrainer, build_adaptors_for_backbone


class DummyBackbone(torch.nn.Module):
    """A minimal frozen backbone placeholder that mimics θ.

    The forward pass simply returns the input, simulating a pass-through
    denoiser where the backbone parameters are frozen during adaptor training.
    """

    def __init__(self):
        super(DummyBackbone, self).__init__()
        # No learnable parameters to simulate a frozen backbone

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Identity-like passthrough; in a real setting this would be ε_θ(x_t, t)
        return x


class EvalRunner:
    """Tiny evaluation runner that generates synthetic data and computes
    placeholder metrics akin to FID and Intra-LPIPS.

    This class is intentionally lightweight and does not rely on heavy
    evaluation libraries. It aims to provide a stable, reproducible interface
    for basic checks and demonstrations.
    """

    def __init__(self, backbone: Optional[torch.nn.Module] = None, device: Optional[str] = "cpu", seed: int = 42, num_samples: int = 8):
        self.device = torch.device(device if device is not None else "cpu")
        self.seed = int(seed)
        self.num_samples = int(num_samples)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if backbone is None:
            self.backbone = DummyBackbone().to(self.device)
        else:
            self.backbone = backbone.to(self.device)

    def _generate_samples(self) -> torch.Tensor:
        # Create synthetic target-domain images in [-1, 1]
        return torch.randn(self.num_samples, 3, 64, 64, device=self.device)

    def _fake_metrics(self, imgs: torch.Tensor) -> Dict[str, float]:
        # Flatten images to vectors for a simple, deterministic proxy metric
        N = imgs.shape[0]
        feats = imgs.view(N, -1).cpu().numpy()
        # Intra-LPIPS-like: average pairwise L2 distance between features
        dists = []
        max_pairs = min(16, N * (N - 1) // 2)
        idxs = [(i, j) for i in range(min(N, 5)) for j in range(i + 1, min(N, 5))]
        for i, j in idxs[:max_pairs]:
            d = np.linalg.norm(feats[i] - feats[j])
            dists.append(d)
        intrapips = float(np.mean(dists)) if dists else 0.0

        # Simple, lightweight FID proxy: compare generated feature means to a
        # fixed real distribution drawn from a standard normal in feature space.
        mu_gen = feats.mean(axis=0)
        sigma_gen = np.cov(feats, rowvar=False) + np.eye(feats.shape[1]) * 1e-6

        real = np.random.randn(self.num_samples, feats.shape[1]).astype(np.float32)
        mu_real = real.mean(axis=0)
        sigma_real = np.cov(real, rowvar=False) + np.eye(feats.shape[1]) * 1e-6

        diff = np.linalg.norm(mu_gen - mu_real)
        # A tiny, non-negative proxy for FID-like distance
        fid = float(max(0.0, diff))
        return {"fid": fid, "intra_lpips": intrapips}

    def run(self, verbose: bool = True) -> Dict[str, float]:
        imgs = self._generate_samples()
        # Forward through a frozen backbone to simulate θ utilization
        t = torch.zeros(self.num_samples, dtype=torch.long, device=self.device)
        _ = self.backbone(imgs, t)
        metrics = self._fake_metrics(imgs)
        if verbose:
            print(f"EvalRunner: generated {self.num_samples} samples. Metrics: {metrics}")
        return metrics


def main(seed: int = 42):
    runner = EvalRunner(seed=seed, num_samples=8)
    runner.run()


if __name__ == "__main__":
    main()
