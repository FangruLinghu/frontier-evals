"""
Lightweight FID (Fréchet Inception Distance) approximation utility.

This module provides a small, backbone-agnostic implementation of a Frechet-distance
based similarity metric between two sets of images. To avoid heavy dependencies
on pretrained classifiers (e.g., Inception), features are extracted using a simple
residual pooling approach:
- Images are resized (via PyTorch) to a fixed size and passed through a simple
  adaptive average pooling to produce per-image feature vectors of shape [N, C].
- The mean and covariance of these features are used to compute a Frechet distance
  between the two distributions.

Notes:
- This is a lightweight proxy for real FID. In production, use a proper InceptionV3
  or an equivalent feature extractor.
- Inputs can be PyTorch tensors (N, C, H, W) in [0, 1] or [0, 255], or numpy arrays
  with the same shapes. The function handles normalization internally.

Public API:
- fid_score(real_images, generated_images, batch_size=32, device=None) -> float
  Compute the approximate FID between two image batches.
- compute_fid_between_images(real_images, generated_images, batch_size=32, device=None) -> float
  Alias for fid_score for readability.

Exports:
- __all__ = ["fid_score", "compute_fid_between_images"]
"""

from __future__ import annotations

from typing import Union
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["fid_score", "compute_fid_between_images"]


def _matrix_sqrt_psd(A: np.ndarray) -> np.ndarray:
    """Compute the matrix square root of a PSD matrix A using a symmetric eigendecomposition.

    The function first symmetrizes A to mitigate numerical asymmetries, then performs
    an eigen decomposition and returns V * sqrt(D) * V^T.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return A
    # Ensure symmetry for a robust sqrt
    A = (A + A.T) / 2.0
    try:
        vals, vecs = np.linalg.eigh(A)
        # Clip negative eigenvalues for numerical stability
        vals = np.clip(vals, 0, None)
        sqrt_vals = np.sqrt(vals)
        return vecs @ np.diag(sqrt_vals) @ vecs.T
    except np.linalg.LinAlgError:
        # Fallback to identity if decomposition fails
        return np.eye(A.shape[0], dtype=A.dtype)


def _calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                                mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute the Frechet distance between two Gaussian distributions.

    mu1, mu2: (D,) means
    sigma1, sigma2: (D, D) covariance matrices
    Returns a scalar FID score.
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    covmean = _matrix_sqrt_psd((sigma1 @ sigma2) / 1.0)
    # If covmean is not real due to numerical issues, symmetrize earlier step
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = float(np.trace(covmean))
    fid = float(np.dot(diff, diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return max(fid, 0.0)


def _extract_features(images: Union[torch.Tensor, np.ndarray], device: torch.device | None,
                      target_size: int = 64) -> np.ndarray:
    """Extract per-image feature vectors from images using a simple pooling-based extractor.

    - Images are expected to be in shape [N, C, H, W]. If a different shape is provided,
      it will be coerced to 4D where possible.
    - The features are global average pooled to [N, C] and used as activations for FID.
    - Outputs a numpy array of shape [N, C].
    """
    if isinstance(images, np.ndarray):
        x = torch.from_numpy(images)
    else:
        x = images
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected images with shape [N, C, H, W], got {list(x.shape)}")
    x = x.to(dtype=torch.float32)
    if device is not None:
        x = x.to(device)
    # Normalize to [0,1] if needed
    if x.max() > 1.0:
        x = x / 255.0
    with torch.no_grad():
        # Resize to target_size x target_size and then pool to [N, C]
        if (x.shape[2], x.shape[3]) != (target_size, target_size):
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        feats = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # [N, C]
        feats_np = feats.cpu().numpy()
    return feats_np


def fid_score(real_images: Union[torch.Tensor, np.ndarray], generated_images: Union[torch.Tensor, np.ndarray],
              batch_size: int = 32, device: torch.device | None = None) -> float:
    """Compute an approximate FID score between two image batches.

    This implementation uses a lightweight feature extractor based on per-image
    channel-wise statistics (mean pooling) and computes the Frechet distance
    between the two Gaussian approximations of the activations.

    real_images and generated_images may be PyTorch tensors of shape [N, C, H, W]
    or numpy arrays of the same shape. Pixel values can be in [0, 1] or [0, 255].
    """
    # Convert to numpy feature representations
    feats_real = _extract_features(real_images, device, target_size=64)  # [N, C]
    feats_gen = _extract_features(generated_images, device, target_size=64)  # [N, C]

    if feats_real.shape[0] < 2 or feats_gen.shape[0] < 2:
        raise ValueError("Need at least 2 samples per set to compute FID.")

    # Compute means and covariances with numpy for both sets
    mu_real = feats_real.mean(axis=0)
    mu_gen = feats_gen.mean(axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    sigma_gen = np.cov(feats_gen, rowvar=False)

    fid = _calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return float(fid)


def compute_fid_between_images(real_images: Union[torch.Tensor, np.ndarray], generated_images: Union[torch.Tensor, np.ndarray],
                             batch_size: int = 32, device: torch.device | None = None) -> float:
    """Alias wrapper for fid_score for readability in codebases.

    Delegates to fid_score and preserves the same semantics.
    """
    return fid_score(real_images, generated_images, batch_size=batch_size, device=device)


if __name__ == "__main__":  # pragma: no cover - quick sanity test
    import torch
    # Generate two random image batches: 16 samples, 3 channels, 64x64
    a = torch.randn(16, 3, 64, 64)
    b = torch.randn(16, 3, 64, 64)
    print("Approximate FID between random batches:", fid_score(a, b))
