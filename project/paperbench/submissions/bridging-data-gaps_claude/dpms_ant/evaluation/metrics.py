"""
Evaluation metrics for DPMs-ANT.

From the paper (Section 5.2):

1. Intra-LPIPS (higher is better):
   - Generate 1000 images
   - Assign each generated image to its nearest training sample (by LPIPS distance)
   - Average pairwise LPIPS distances within each cluster
   - Average across clusters
   - Score of 0 = perfect copies (no diversity)
   - Higher = more diversity

2. FID (lower is better):
   - Standard Frechet Inception Distance
   - Distribution distance between generated and real samples
   - Computed against larger target datasets (e.g., 2.5k Sunglasses, 2.7k Babies)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
import lpips


class IntraLPIPS:
    """
    Intra-LPIPS metric for measuring diversity of generated images.

    From CDC (Ojha et al., 2021):
    1. Generate N images
    2. For each generated image, find nearest training image by LPIPS
    3. Group generated images by their nearest training image
    4. Compute average pairwise LPIPS within each group
    5. Average across groups
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.lpips_fn = lpips.LPIPS(net="alex").to(device)
        self.lpips_fn.eval()
        self.device = device

    @torch.no_grad()
    def compute_lpips_distance(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LPIPS distance between two images."""
        return self.lpips_fn(img1, img2).squeeze()

    @torch.no_grad()
    def compute(
        self,
        generated_images: torch.Tensor,
        training_images: torch.Tensor,
        batch_size: int = 16,
    ) -> Tuple[float, float]:
        """
        Compute Intra-LPIPS score.

        Args:
            generated_images: Generated images (N_gen, C, H, W) in [-1, 1]
            training_images: Training images (N_train, C, H, W) in [-1, 1]
            batch_size: Batch size for LPIPS computation

        Returns:
            Tuple of (mean_intra_lpips, std_intra_lpips)
        """
        n_gen = len(generated_images)
        n_train = len(training_images)

        generated_images = generated_images.to(self.device)
        training_images = training_images.to(self.device)

        # Step 1: Assign each generated image to nearest training image
        assignments = []  # Index of nearest training image for each generated image

        for i in tqdm(range(0, n_gen, batch_size), desc="Computing LPIPS assignments"):
            batch_gen = generated_images[i:i + batch_size]
            min_dists = torch.full((len(batch_gen),), float("inf"), device=self.device)
            min_indices = torch.zeros(len(batch_gen), dtype=torch.long, device=self.device)

            for j in range(n_train):
                train_img = training_images[j:j + 1].expand_as(batch_gen)
                dists = self.lpips_fn(batch_gen, train_img).squeeze(-1).squeeze(-1).squeeze(-1)

                closer = dists < min_dists
                min_dists[closer] = dists[closer]
                min_indices[closer] = j

            assignments.extend(min_indices.cpu().tolist())

        # Step 2: Group by assignment
        clusters = {}
        for gen_idx, train_idx in enumerate(assignments):
            if train_idx not in clusters:
                clusters[train_idx] = []
            clusters[train_idx].append(gen_idx)

        # Step 3: Compute pairwise LPIPS within each cluster
        cluster_scores = []

        for train_idx, gen_indices in clusters.items():
            if len(gen_indices) < 2:
                continue

            cluster_imgs = generated_images[gen_indices]
            n = len(cluster_imgs)
            pairwise_dists = []

            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.compute_lpips_distance(
                        cluster_imgs[i:i + 1], cluster_imgs[j:j + 1]
                    )
                    pairwise_dists.append(dist.item())

            if pairwise_dists:
                cluster_scores.append(np.mean(pairwise_dists))

        # Step 4: Average across clusters
        if cluster_scores:
            return float(np.mean(cluster_scores)), float(np.std(cluster_scores))
        else:
            return 0.0, 0.0


class FIDCalculator:
    """
    FID computation using PyTorch.

    Uses InceptionV3 features to compute Frechet Inception Distance.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self._inception = None

    def _get_inception(self):
        """Lazy load InceptionV3."""
        if self._inception is None:
            from torchvision.models import inception_v3
            self._inception = inception_v3(pretrained=True, transform_input=False).to(self.device)
            self._inception.fc = nn.Identity()
            self._inception.eval()
        return self._inception

    @torch.no_grad()
    def get_features(
        self,
        images: torch.Tensor,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract InceptionV3 features from images.

        Args:
            images: Images tensor (N, C, H, W) in [-1, 1]
            batch_size: Batch size

        Returns:
            Features array (N, 2048)
        """
        model = self._get_inception()

        # Resize to 299x299 and normalize to [0, 1]
        from torchvision.transforms.functional import resize

        features_list = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)

            # Normalize from [-1, 1] to [0, 1]
            batch = (batch + 1) / 2

            # Resize to 299x299
            batch = torch.nn.functional.interpolate(
                batch, size=(299, 299), mode="bilinear", align_corners=False
            )

            feats = model(batch)
            features_list.append(feats.cpu().numpy())

        return np.concatenate(features_list, axis=0)

    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """Compute Frechet distance between two Gaussians."""
        from scipy import linalg

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def compute_fid(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        batch_size: int = 32,
    ) -> float:
        """
        Compute FID between two sets of images.

        Args:
            images1: First set of images (N1, C, H, W) in [-1, 1]
            images2: Second set of images (N2, C, H, W) in [-1, 1]
            batch_size: Batch size

        Returns:
            FID score
        """
        feats1 = self.get_features(images1, batch_size)
        feats2 = self.get_features(images2, batch_size)

        mu1, sigma1 = self.compute_statistics(feats1)
        mu2, sigma2 = self.compute_statistics(feats2)

        return self.frechet_distance(mu1, sigma1, mu2, sigma2)


def evaluate_model(
    model: nn.Module,
    diffusion,
    training_images: torch.Tensor,
    reference_images: Optional[torch.Tensor] = None,
    n_generated: int = 1000,
    image_size: int = 256,
    batch_size: int = 16,
    ddim_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Full evaluation: compute FID and Intra-LPIPS.

    Args:
        model: Trained model
        diffusion: GaussianDiffusion
        training_images: Few-shot training images
        reference_images: Larger reference set for FID (optional)
        n_generated: Number of images to generate
        image_size: Image resolution
        batch_size: Generation batch size
        ddim_steps: DDIM steps for faster sampling
        device: Device

    Returns:
        Dict with "fid" and "intra_lpips" scores
    """
    model.eval()

    # Generate images
    print(f"Generating {n_generated} images...")
    generated = []

    for i in tqdm(range(0, n_generated, batch_size)):
        bs = min(batch_size, n_generated - i)
        shape = (bs, 3, image_size, image_size)

        samples = diffusion.ddim_sample(
            model, shape, device,
            ddim_steps=ddim_steps,
            progress=False,
        )
        generated.append(samples.cpu())

    generated = torch.cat(generated, dim=0)
    print(f"Generated {len(generated)} images")

    results = {}

    # Intra-LPIPS
    print("Computing Intra-LPIPS...")
    intra_lpips = IntraLPIPS(device=device)
    mean_ilpips, std_ilpips = intra_lpips.compute(generated, training_images)
    results["intra_lpips_mean"] = mean_ilpips
    results["intra_lpips_std"] = std_ilpips
    print(f"  Intra-LPIPS: {mean_ilpips:.3f} ± {std_ilpips:.3f}")

    # FID (if reference images provided)
    if reference_images is not None and len(reference_images) >= 100:
        print("Computing FID...")
        fid_calc = FIDCalculator(device=device)
        fid_score = fid_calc.compute_fid(generated, reference_images)
        results["fid"] = fid_score
        print(f"  FID: {fid_score:.2f}")

    return results
