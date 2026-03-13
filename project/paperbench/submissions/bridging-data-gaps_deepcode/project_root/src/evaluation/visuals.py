# Lightweight visualization helpers for evaluation in DPMs-ANT pipeline
"""evaluation.visuals

A compact collection of utilities to visualize evaluation metrics and sample images
without pulling heavy dependencies. These helpers are designed to be robust and work
with common shapes used in diffusion model evaluation.

Public API:
- save_convergence_plots(metric_series, path)
- save_image_grid(images, path, nrow=8, normalize=True)

These utilities are intentionally small and dependency-light. They rely on
matplotlib (for plotting) and torchvision (for image grid creation) when available,
but gracefully degrade if torchvision is not installed.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import math
import torch
import numpy as np

__all__ = ["save_convergence_plots", "save_image_grid"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_convergence_plots(metric_series: Dict[str, List[float]], path: str) -> None:
    """Plot convergence curves for given metrics and save to a file.

    Args:
        metric_series: Mapping from metric name to a list of values per iteration/step.
        path: Destination file path to save the plot (e.g., plots/convergence.png).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("matplotlib is required for save_convergence_plots but is not installed.") from e

    if not metric_series:
        # Nothing to plot
        return

    _ensure_dir(os.path.dirname(path) or ".")
    plt.figure(figsize=(8, 4.5))
    for name, series in metric_series.items():
        if isinstance(series, (list, tuple)):
            y = list(series)
        else:
            # Fallback: try to convert to list
            y = list(series)
        plt.plot(range(len(y)), y, label=str(name))
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(path, dpi=150)
    plt.close()


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
) -> None:
    """Save a grid of images as a single image file.

    Accepts images of shape:
      - [N, C, H, W]
      - [N, H, W, C] (will be transposed to [N, C, H, W])
    The function will move data to CPU, optionally normalize to [0, 1], and then
    construct a grid using torchvision if available, otherwise falls back to a simple
    numpy-based grid.

    Args:
        images: Torch tensor containing image batch.
        path: Path to save the grid image.
        nrow: Number of images per row in the grid.
        normalize: If True, map image values to [0, 1] for visualization.
    """
    if images is None:
        return

    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)

    # Ensure 4D: [N, C, H, W]
    if images.dim() == 3:
        # Assume [N, H, W] or [C, H, W'?], try to infer
        if images.shape[1] in {1, 3}:
            images = images.unsqueeze(1)  # [N,1,H,W]? not ideal; leave to caller
        else:
            images = images.unsqueeze(0)
    if images.dim() == 2:
        # [N, N?] not a valid image batch; skip gracefully
        raise ValueError("save_image_grid requires a 3D or 4D tensor with image data.")

    if images.dim() == 4:
        imgs = images
    else:
        raise ValueError("save_image_grid encountered an unsupported tensor shape: {}".format(images.shape))

    imgs = imgs.detach().cpu()
    # Normalize to [0, 1] if requested
    if normalize:
        # Expect values roughly in [-1, 1] or [0, 1], detect range
        minv = float(imgs.min())
        maxv = float(imgs.max())
        if minv < -0.5 or maxv > 1.5:
            # Likely in [-1, 1]
            imgs = (imgs + 1.0) / 2.0
        else:
            # Already in [0,1]
            imgs = torch.clamp(imgs, 0.0, 1.0)
    # Try torchvision grid first
    grid = None
    try:
        import torchvision
        import torchvision.utils as vutils
        if imgs.size(1) == 1:
            # replicate grayscale to 3 channels for nicer visualization
            imgs_rgb = imgs.repeat(1, 3, 1, 1)
        else:
            imgs_rgb = imgs
        grid = vutils.make_grid(imgs_rgb, nrow=nrow, padding=2, normalize=False, scale_each=True)
        # grid: [C, H, W] tensor in [0,1]
        grid_np = grid.permute(1, 2, 0).numpy()
        # Convert to uint8
        grid_uint8 = (grid_np * 255).astype(np.uint8)
        from PIL import Image
        im = Image.fromarray(grid_uint8)
        im.save(path)
        return
    except Exception:
        # Fallback to numpy-based grid if torchvision is not available
        pass

    # Fallback: simple grid assembly
    N, C, H, W = imgs.shape
    if C != 3:
        imgs = imgs.repeat(1, 3, 1, 1)
    # Compute grid dimensions
    n = int(math.ceil(math.sqrt(N)))
    grid_h = int(n) * H
    grid_w = int(n) * W
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # [N, H, W, C]
    imgs_np = (imgs_np * 255).astype(np.uint8)
    idx = 0
    for i in range(n):
        for j in range(n):
            if idx >= N:
                break
            top = i * H
            left = j * W
            grid_img[top : top + H, left : left + W, :] = imgs_np[idx]
            idx += 1
    from PIL import Image
    Image.fromarray(grid_img).save(path)


# Exposed helper alias for backward compatibility if needed by other modules
def make_gallery(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """Backward-compatible alias to save_image_grid with default normalization."""
    save_image_grid(images, path, nrow=nrow, normalize=True)


__all__ = ["save_convergence_plots", "save_image_grid", "make_gallery"]
