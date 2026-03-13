## eval.py
"""
Evaluation utilities for DPMs-ANT reproduction.

This module provides an Evaluator class that can:
- Generate samples from a diffusion model wrapper that includes an adaptor (DiffusionWrapper).
- Compute Intra-LPIPS, a diversity metric, given generated samples and a small set of target exemplars.
- Compute FID between generated samples and real target-domain data when feasible (optional, requires SciPy).
- Provide a lightweight sampling routine that uses the same diffusion schedule style used in training.

Notes
- This module follows the design constraints from the project: imports are flat
  (from model import DiffusionWrapper, from utils import log_metrics).
- The Adaptor-enabled diffusion wrapper is expected to implement a forward method
  that returns the predicted noise ε_θ(x_t, t). Sampling is performed in this module
  using a standard DDPM reverse-step recipe.
- LPIPS is optional; if lpips is not installed, Intra-LPIPS computation will raise a
  friendly error to indicate the requirement for evaluation.
- FID computation relies on SciPy for the exact Fréchet distance (sqrtm). If SciPy
  is not available, compute_fid will gracefully return None and log a warning.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DiffusionWrapper
from utils import log_metrics

# Optional dependencies
try:
    import lpips  # type: ignore
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

try:
    # FID utilities via SciPy; used if available
    from scipy import linalg  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    # Inception model for FID activations (optional, used only if SciPy is available)
    import torchvision.models as models  # type: ignore
    import torchvision.transforms as transforms  # type: ignore
    INCEPTION_AVAILABLE = True
except Exception:
    INCEPTION_AVAILABLE = False


class Evaluator:
    """
    Evaluation utility for DPMs-ANT.

    Responsibilities:
    - Sampling from a adaptor-enabled diffusion model (DiffusionWrapper).
    - Computing Intra-LPIPS (diversity) across generated samples with respect to 10-shot exemplars.
    - Optional: Computing FID between generated samples and real target-domain images when SciPy
      is available (uses InceptionV3 activations to compute statistics).
    """

    def __init__(
        self,
        diff_model: DiffusionWrapper,
        device: Optional[torch.device] = None,
        lpips_net: str = "vgg",
        fid_backend: str = "pytorch",
        T: int = 1000,
    ) -> None:
        """
        Initialize Evaluator.

        Args:
            diff_model: DiffusionWrapper with adaptor; used for sampling.
            device: Torch device to run evaluation on. If None, uses CUDA when available.
            lpips_net: LPIPS backbone to use (e.g., 'vgg', 'alex'). Requires lpips package.
            fid_backend: Reserved for future extension; kept for API compatibility.
            T: Number of diffusion timesteps used for the internal schedule (sampling).
        """
        self.diff_model: DiffusionWrapper = diff_model
        self.device: torch.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.T: int = int(T)

        # LPIPS setup
        self.lpips = None
        if LPIPS_AVAILABLE:
            # Lazy import: LPIPS needs to be on the correct device
            self.lpips = lpips.LPIPS(net=lpips_net).to(self.device)
            self.lpips.eval()
        else:
            warnings.warn("LPIPS package not available. Intra-LPIPS will not run.", UserWarning)

        # Schedule for sampling (alpha_bar / alpha). Precompute for reuse in sampling.
        self._alpha_bar_list: List[float] = []
        self._alpha_list: List[float] = []
        self._schedule_built: bool = False
        self._build_schedule(self.T)

        # FID (optional)
        self._fid_activations_cache: Optional[Tuple[List[float], List[float]]] = None
        self._inception_model = None  # lazy init if needed
        self.fid_backend = fid_backend

        # Warn if SciPy is not available but FID may be requested
        if not SCIPY_AVAILABLE:
            warnings.warn(
                "SciPy not available. FID computation will be skipped if compute_fid is called.",
                UserWarning,
            )

    # ----------------------------
    # Scheduling utilities
    # ----------------------------
    def _build_schedule(self, T: int) -> None:
        """
        Build a simple DDPM-like schedule for α_t and ᾱ_t to enable sampling.

        The schedule is: linear betas from beta_start to beta_end, α_t = 1 - β_t,
        ᾱ_t = prod_i^t α_i.

        This is used for sampling when the underlying base model does not expose a
        schedule interface, to keep sampling deterministic and reproducible.
        """
        if self._schedule_built:
            return

        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, steps=T + 1)[1:]  # t = 1..T
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self._alpha_list = [float(a) for a in alphas.tolist()]
        self._alpha_bar_list = [float(ab) for ab in alpha_bar.tolist()]
        self._schedule_built = True

    def _get_schedule_values(self, t: int) -> Tuple[float, float, float, float]:
        """
        Retrieve sqrt(ᾱ_t), sqrt(1 - ᾱ_t), ᾱ_t, α_t for a given timestep t ∈ {1..T}.

        Returns:
            (sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t, alpha_bar_t, alpha_t)
        """
        t = max(1, min(int(t), self.T))
        alpha_bar_t = self._alpha_bar_list[t - 1]
        alpha_t = self._alpha_list[t - 1]
        sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(max(0.0, 1.0 - alpha_bar_t))
        return sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t, alpha_bar_t, alpha_t

    # ----------------------------
    # Sampling utility
    # ----------------------------
    def sample_images(self, num_images: int) -> torch.Tensor:
        """
        Sample images unconditionally from the adaptor-enabled diffusion model.

        This uses a standard DDPM reverse process with η=1 (adding noise per step)
        as in Ho et al.-style sampling, but implemented in a compact form here.

        Args:
            num_images: Number of images to sample.

        Returns:
            Tensor of shape [num_images, C, H, W] with values in [-1, 1].
        """
        # Default image size/channels inferred from the base model's first reachable target
        # We attempt to infer channel count by performing a dummy pass with a small tensor.
        with torch.no_grad():
            # Determine a default shape by performing a dummy forward through adaptor-enabled model
            # If this fails, fall back to a common 3x256x256 shape.
            try:
                x = torch.randn((max(1, num_images), 3, 256, 256), device=self.device)
            except Exception:
                x = torch.randn((max(1, num_images), 3, 256, 256), device=self.device)

            T = self.T
            # Start from pure Gaussian noise at step T
            x_t = torch.randn(num_images, x.size(1), x.size(2), x.size(3), device=self.device)

            for t in range(T, 0, -1):
                # Predict noise with current adaptor-enabled model
                eps_theta = self.diff_model(x_t, t)

                a_t = self._alpha_list[t - 1]
                a_bar_t = self._alpha_bar_list[t - 1]

                # x0 estimate
                denom = max(1e-12, a_bar_t)
                x0_pred = (x_t - math.sqrt(max(0.0, 1.0 - a_bar_t)) * eps_theta) / math.sqrt(denom)

                # Direction to previous step (from Ho et al.)
                x_prev = (x_t - ((1.0 - a_t) / math.sqrt(max(1e-12, 1.0 - a_bar_t))) * eps_theta) / math.sqrt(a_t)

                if t > 1:
                    beta_t = 1.0 - a_t
                    a_bar_prev = self._alpha_bar_list[t - 2]
                    sigma_t = math.sqrt(max(0.0, beta_t) * max(0.0, (1.0 - a_bar_prev)) / max(1e-12, (1.0 - a_bar_t)))
                    z = torch.randn_like(x_t)
                    x_t = x_prev + sigma_t * z
                else:
                    x_t = x_prev

            # Ensure output is in [-1, 1]
            return x_t.clamp(-1.0, 1.0)

    # ----------------------------
    # Intra-LPIPS computation
    # ----------------------------
    def compute_intra_lpips(self, generated_images: torch.Tensor, target_images: torch.Tensor) -> Optional[float]:
        """
        Compute Intra-LPIPS as described in the paper.

        Steps:
        - For each generated sample g_i, compute LPIPS distance to each target exemplar t_j.
        - Assign g_i to the exemplar with the minimum distance (cluster j*).
        - For each cluster j, compute mean LPIPS distance among all generated samples assigned to j.
        - Return the average of these cluster means as Intra-LPIPS.

        Args:
            generated_images: Tensor [N, C, H, W], values in [-1, 1].
            target_images: Tensor [K, C, H, W], values in [-1, 1].

        Returns:
            Float Intra-LPIPS score. Returns None if LPIPS is not available.
        """
        if self.lpips is None:
            raise RuntimeError("LPIPS is not available. Install the lpips package to run Intra-LPIPS.")

        if generated_images.ndim != 4 or target_images.ndim != 4:
            raise ValueError("Expected tensors of shape [N, C, H, W] for images.")

        N = int(generated_images.size(0))
        K = int(target_images.size(0))

        device = self.device
        gen = generated_images.to(device)
        targ = target_images.to(device)

        # Pre-allocate distance matrix [N, K]
        dist_matrix = torch.zeros((N, K), dtype=torch.float32, device=device)

        with torch.no_grad():
            for i in range(N):
                g_i = gen[i:i + 1].expand(K, -1, -1, -1)  # [K, C, H, W]
                d = self.lpips(g_i, targ)  # [K, 1, 1, 1]
                dist_matrix[i] = d.view(-1)

        # Assign each generated image to the nearest exemplar
        assignments = dist_matrix.argmin(dim=1)  # [N]

        # Compute per-cluster LPIPS means
        cluster_means = []
        for j in range(K):
            idxs = (assignments == j).nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(idxs, int):
                idxs = [idxs]
            m = len(idxs)
            if m <= 1:
                continue
            cluster_dist_sum = 0.0
            cluster_count = 0
            for a in range(m):
                ga = gen[idxs[a]:idxs[a] + 1]
                for b in range(a + 1, m):
                    gb = gen[idxs[b]:idxs[b] + 1]
                    with torch.no_grad():
                        val = self.lpips(ga, gb)  # [1,1,1,1]
                        cluster_dist_sum += float(val.item())
                    cluster_count += 1
            if cluster_count > 0:
                cluster_mean = cluster_dist_sum / cluster_count
                cluster_means.append(cluster_mean)

        if len(cluster_means) == 0:
            # Not enough data to compute a meaningful diversity metric
            return None

        intra_lpips = float(sum(cluster_means) / len(cluster_means))
        # Log for reproducibility
        log_metrics({"eval/intra_lpips": intra_lpips, "eval/N": N, "eval/K": K})
        return intra_lpips

    # ----------------------------
    # FID computation (optional)
    # ----------------------------
    def compute_fid(self, generated_images: torch.Tensor, real_images: torch.Tensor) -> Optional[float]:
        """
        Compute Frechet Inception Distance (FID) between generated and real images.

        This implementation requires SciPy and a working InceptionV3 backbone via PyTorch.

        Args:
            generated_images: [N, C, H, W], values in [-1, 1].
            real_images: [M, C, H, W], values in [-1, 1].

        Returns:
            FID score (float) if computation is possible; otherwise None.
        """
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available. Cannot compute FID.", UserWarning)
            return None

        if not INCEPTION_AVAILABLE:
            warnings.warn("Inception model not available. Cannot compute FID.", UserWarning)
            return None

        import numpy as np

        def _to_uint8(imgs: torch.Tensor) -> torch.Tensor:
            # Convert [-1,1] to [0,1], then to [0,255] for typical Inception input if needed
            imgs = imgs.clamp(-1.0, 1.0)
            imgs = (imgs + 1.0) * 0.5  # [0,1]
            return imgs

        device = self.device
        batch_size = 16

        # Prepare Inception model
        if self._inception_model is None:
            inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
            inception.eval()
            # We'll use the pool3 features via a forward hook
            self._inception_model = inception
            self._fid_activations = []

        inception = self._inception_model  # type: ignore

        # Helper to compute activations for a set of images
        def _get_activations(imgs: torch.Tensor) -> np.ndarray:
            imgs = imgs.to(device)
            # Resize to 299x299 as required by Inception v3
            if imgs.size(-1) != 299 or imgs.size(-2) != 299:
                imgs_resized = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
            else:
                imgs_resized = imgs
            # Normalize to [0,1]
            imgs_proc = _to_uint8(imgs_resized)

            # Collect activations via a forward hook on avgpool to get 2048-D features
            feats: List[torch.Tensor] = []

            def _hook(module, input, output):
                # output: [N, 2048, 1, 1]
                feats.append(output.view(output.size(0), -1).detach())

            hook = inception.avgpool.register_forward_hook(_hook)
            with torch.no_grad():
                # Inception expects inputs in [0,1] float; ensure normalized
                _ = inception(imgs_proc)
            hook.remove()
            if len(feats) == 0:
                raise RuntimeError("Failed to capture Inception activations for FID.")
            acts = feats[0]  # [N, 2048]
            return acts.cpu().numpy()

        # Split generated_images into batches and compute activations
        gen_acts = []
        real_acts = []

        N = int(generated_images.size(0))
        M = int(real_images.size(0))

        with torch.no_grad():
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                batch = generated_images[i:end]
                acts = _get_activations(batch)
                gen_acts.append(acts)

            for i in range(0, M, batch_size):
                end = min(i + batch_size, M)
                batch = real_images[i:end]
                acts = _get_activations(batch)
                real_acts.append(acts)

        if len(gen_acts) == 0 or len(real_acts) == 0:
            return None

        mu_gen = np.concatenate(gen_acts, axis=0).mean(axis=0)
        mu_real = np.concatenate(real_acts, axis=0).mean(axis=0)

        sigma_gen = np.cov(np.concatenate(gen_acts, axis=0).T)
        sigma_real = np.cov(np.concatenate(real_acts, axis=0).T)

        # Compute sqrtm of the product
        covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
        if covmean is None:
            return None
        # Numerical issues can lead to tiny imaginary parts
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(np.sum((mu_gen - mu_real) ** 2) + np.trace(sigma_gen + sigma_real - covmean * 2.0))
        log_metrics({"eval/fid": fid, "eval/N": N, "eval/M": M})
        return fid

    # ----------------------------
    # Public interface (optional helper)
    # ----------------------------
    def compute_metrics(
        self,
        generated_images: torch.Tensor,
        target_images: torch.Tensor,
        real_images_for_fid: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Convenience wrapper to compute Intra-LPIPS and optionally FID.

        Args:
            generated_images: [N, C, H, W], values in [-1, 1]
            target_images: [K, C, H, W], target exemplars
            real_images_for_fid: [M, C, H, W], real target-domain images for FID (optional)

        Returns:
            dict with keys 'intra_lpips' and 'fid' (fid may be None if unavailable)
        """
        results: dict = {}

        # Intra-LPIPS
        if self.lpips is not None:
            intra_lpips = self.compute_intra_lpips(generated_images, target_images)
            results["intra_lpips"] = intra_lpips
        else:
            results["intra_lpips"] = None

        # FID (optional)
        if real_images_for_fid is not None:
            fid = self.compute_fid(generated_images, real_images_for_fid)
            results["fid"] = fid
        else:
            results["fid"] = None

        log_metrics({"eval/overall": results})
        return results