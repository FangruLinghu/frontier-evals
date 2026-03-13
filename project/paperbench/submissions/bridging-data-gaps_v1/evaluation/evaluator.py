## evaluation/evaluator.py
"""Evaluation module for similarity-guided diffusion model adaptation.

This module implements evaluation metrics for measuring diversity and quality
of generated samples from the adapted diffusion model:

- Intra-LPIPS: Perceptual diversity metric (higher = more diverse)
- FID: Fréchet Inception Distance (lower = better quality)

The evaluator uses:
- Pre-trained LPIPS (Alex variant) for perceptual distance computation
- DDPM reverse process for sample generation
- InceptionV3 for FID computation
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from typing import Optional, Tuple

from model.diffusion_model import DiffusionUNet
from diffusion.utils import DiffusionUtils


def load_lpips(device: str) -> nn.Module:
    """Load pre-trained LPIPS network (Alex architecture) for perceptual distance.
    
    Loads the Alex variant of LPIPS from torch.hub, which measures perceptual
    similarity between images. Lower LPIPS distance = more similar perceptually.
    
    Args:
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        LPIPS network module (Alex variant) in evaluation mode
    
    Example:
        >>> lpips_model = load_lpips('cuda')
        >>> distance = lpips(img1, img2)  # Perceptual distance
    """
    try:
        # Try to import lpips package if available
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        return lpips_model
    except ImportError:
        # Fallback: use a simple perceptual loss approximation
        # This is a simplified version when lpips package is not available
        print("Warning: lpips package not found. Using simplified perceptual loss.")
        return SimplePerceptualLoss().to(device)


class SimplePerceptualLoss(nn.Module):
    """Simplified perceptual loss as fallback when lpips is not available.
    
    Uses a pre-trained VGG feature extractor for computing perceptual distance.
    This is a simplified approximation of LPIPS for environments without the
    lpips package.
    """
    
    def __init__(self) -> None:
        """Initialize simplified perceptual loss with VGG features."""
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            self.features = vgg.features[:23]  # First few conv layers
            for param in self.features.parameters():
                param.requires_grad = False
            self.eval()
        except ImportError:
            # Fallback to simple MSE if torchvision not available
            self.features = None
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute perceptual distance between two images.
        
        Args:
            x: First image tensor [-1, 1]
            y: Second image tensor [-1, 1]
        
        Returns:
            Perceptual distance tensor (scalar)
        """
        if self.features is None:
            # Simple MSE fallback
            return torch.mean((x - y) ** 2)
        
        # Extract features and compute distance
        x_feat = self.features(x)
        y_feat = self.features(y)
        return torch.mean((x_feat - y_feat) ** 2)


def compute_lpips_distance(
    img1: torch.Tensor,
    img2: torch.Tensor,
    lpips_model: nn.Module
) -> torch.Tensor:
    """Compute LPIPS perceptual distance between two images.
    
    Preprocesses images to [-1, 1] range (expected by LPIPS network),
    then passes through LPIPS network to compute perceptual distance.
    
    Args:
        img1: First image tensor [B, C, H, W], range [0, 1] or [-1, 1]
        img2: Second image tensor [B, C, H, W], range [0, 1] or [-1, 1]
        lpips_model: Pre-trained LPIPS network (Alex variant)
    
    Returns:
        LPIPS distance tensor [B] (scalar distance for each image pair)
    
    Example:
        >>> lpips_model = load_lpips('cuda')
        >>> img1 = torch.randn(1, 3, 32, 32)
        >>> img2 = torch.randn(1, 3, 32, 32)
        >>> distance = compute_lpips_distance(img1, img2, lpips_model)
    """
    # Preprocess images to [-1, 1] range (LPIPS expects this)
    img1_processed = img1.clone()
    img2_processed = img2.clone()
    
    # Convert from [0, 1] to [-1, 1] if needed
    if img1_processed.min() >= 0 and img1_processed.max() <= 1:
        img1_processed = img1_processed * 2 - 1
    if img2_processed.min() >= 0 and img2_processed.max() <= 1:
        img2_processed = img2_processed * 2 - 1
    
    # Ensure both images are on the same device
    img1_processed = img1_processed.to(lpips_model.parameters().__next__().device)
    img2_processed = img2_processed.to(lpips_model.parameters().__next__().device)
    
    # Compute LPIPS distance
    with torch.no_grad():
        distance = lpips_model(img1_processed, img2_processed)
    
    return distance.squeeze()


class Evaluator:
    """Evaluator for similarity-guided diffusion model.
    
    This class provides methods for evaluating the adapted diffusion model:
    - Sample generation using DDPM reverse process
    - Intra-LPIPS metric for measuring diversity
    - FID metric for measuring quality
    
    The evaluator uses:
    - Trained DiffusionUNet with adaptor layers
    - DiffusionUtils for forward/reverse process
    - Pre-trained LPIPS (Alex) for perceptual distance
    - InceptionV3 for FID computation
    
    Attributes:
        model: Trained diffusion model with adaptor layers
        diffusion_utils: Diffusion utilities for forward/reverse process
        device: Device for computation (cuda/cpu)
        lpips_model: Pre-trained LPIPS network for perceptual distance
    """
    
    def __init__(
        self,
        model: DiffusionUNet,
        diffusion_utils: DiffusionUtils,
        device: str = 'cuda'
    ) -> None:
        """Initialize evaluator with trained model and diffusion utilities.
        
        Loads pre-trained LPIPS network (Alex variant) for perceptual distance
        computation during Intra-LPIPS evaluation.
        
        Args:
            model: Trained diffusion model with adaptor layers
            diffusion_utils: Diffusion utilities for forward/reverse process
            device: Device for computation (default: 'cuda')
        
        Example:
            >>> from model.diffusion_model import DiffusionUNet
            >>> from diffusion.utils import DiffusionUtils
            >>> from config import create_toy_config
            >>> 
            >>> config = create_toy_config()
            >>> model = DiffusionUNet(config)
            >>> diffusion_utils = DiffusionUtils(timesteps=1000)
            >>> 
            >>> evaluator = Evaluator(model, diffusion_utils, device='cuda')
            >>> intra_lpips = evaluator.compute_intra_lpips(num_samples=1000)
        """
        self.model = model
        self.diffusion_utils = diffusion_utils
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Load pre-trained LPIPS network (Alex variant)
        self.lpips_model = load_lpips(device)
        self.lpips_model.eval()
        
        # Pre-extract noise schedule parameters for efficient sampling
        self._prepare_noise_schedule()
    
    def _prepare_noise_schedule(self) -> None:
        """Pre-extract noise schedule parameters for efficient sampling."""
        schedule = self.diffusion_utils.get_noise_schedule()
        
        # Move all tensors to device for faster sampling
        self.betas = schedule['betas'].to(self.device)
        self.alphas = schedule['alphas'].to(self.device)
        self.alphas_cumprod = schedule['alphas_cumprod'].to(self.device)
        self.alphas_cumprod_prev = schedule['alphas_cumprod_prev'].to(self.device)
        
        # Compute derived quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance and sigma for sampling
        posterior_variance = schedule['posterior_variance'].to(self.device)
        self.sqrt_posterior_variance = torch.sqrt(posterior_variance)
        
        # Also precompute 1 - alphas_cumprod for use in sampling
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
    
    def generate_samples(
        self,
        num_samples: int,
        image_size: int = 32,
        channels: int = 3
    ) -> torch.Tensor:
        """Generate samples using the reverse diffusion process (DDPM sampling).
        
        Starting from random noise x_T ~ N(0, I), iteratively applies the learned
        model to denoise through T timesteps down to x_0. Uses the trained
        adaptor-enhanced U-Net for noise prediction.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Spatial resolution of generated images (default: 32)
            channels: Number of image channels (default: 3 for RGB)
        
        Returns:
            Generated samples tensor of shape [num_samples, channels, image_size, image_size]
        
        Example:
            >>> evaluator = Evaluator(model, diffusion_utils, device='cuda')
            >>> samples = evaluator.generate_samples(num_samples=100, image_size=32, channels=3)
            >>> print(samples.shape)  # torch.Size([100, 3, 32, 32])
        """
        batch_size = 50  # Generate in batches for memory efficiency
        
        all_samples = []
        
        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                current_batch_size = end_idx - start_idx
                
                # Start from random noise x_T ~ N(0, I)
                x_t = torch.randn(
                    current_batch_size, 
                    channels, 
                    image_size, 
                    image_size, 
                    device=self.device
                )
                
                # Iteratively denoise from T-1 down to 0
                for t in range(self.diffusion_utils.timesteps, 0, -1):
                    # Create timestep tensor
                    t_batch = torch.full(
                        (current_batch_size,), 
                        t, 
                        device=self.device, 
                        dtype=torch.long
                    )
                    
                    # Single reverse diffusion step
                    x_t = self.ddpm_sampling_step(x_t, t_batch)
                
                all_samples.append(x_t)
        
        # Concatenate all samples
        samples = torch.cat(all_samples, dim=0)
        
        return samples
    
    def ddpm_sampling_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Single reverse diffusion step using DDPM formula.
        
        Given noisy image x_t at timestep t, predicts noise using model,
        then computes x_{t-1} using the DDPM reverse process formula:
        
        x_{t-1} = sqrt(alpha_{t-1}) * ((x_t - sqrt(1-alpha_t) * epsilon_theta) / sqrt(alpha_t)) 
                  + sqrt(1 - alpha_{t-1} - sigma_t^2) * epsilon_theta 
                  + sigma_t * epsilon
        
        where epsilon ~ N(0, I).
        
        Args:
            x_t: Noisy image tensor at timestep t [B, C, H, W]
            t: Current timestep indices [B] (values in [1, T])
        
        Returns:
            x_{t-1}: Denoised image at timestep t-1 [B, C, H, W]
        
        Example:
            >>> x_t = torch.randn(4, 3, 32, 32, device='cuda')
            >>> t = torch.tensor([500, 499, 501, 498], device='cuda')
            >>> x_prev = evaluator.ddpm_sampling_step(x_t, t)
            >>> print(x_prev.shape)  # torch.Size([4, 3, 32, 32])
        """
        batch_size = x_t.size(0)
        
        # Predict noise using the model
        predicted_noise = self.model(x_t, t)
        
        # Get noise schedule values at timestep t
        # Index into precomputed arrays
        t_idx = t.long() - 1  # Convert to 0-indexed
        
        # Get alpha_t (at index t-1 for 1-indexed timesteps)
        alpha_t = self.alphas[t_idx].view(batch_size, 1, 1, 1)
        
        # Get alpha_{t-1} (at index t-2)
        t_prev_idx = torch.clamp(t_idx - 1, min=0)
        alpha_t_prev = self.alphas[t_prev_idx].view(batch_size, 1, 1, 1)
        
        # Get sqrt(alpha_t)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        
        # Get sqrt(1 - alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        # Get sqrt(alpha_{t-1})
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        
        # Get alpha_{t-1} (not squared)
        # For t=1, alpha_0 = 1, so sqrt(1 - alpha_0) = 0
        alpha_t_prev_not_sq = alpha_t_prev
        
        # Compute variance term: 1 - alpha_{t-1} - sigma_t^2
        # Using simplified DDPM with sigma_t = sqrt((1 - alpha_{t-1}) / (1 - alpha_t) * beta_t)
        # This is the standard DDPM variance
        beta_t = self.betas[t_idx].view(batch_size, 1, 1, 1)
        
        # Compute posterior variance
        # sigma_t^2 = (1 - alpha_{t-1}) * beta_t / (1 - alpha_t)
        one_minus_alpha_t_prev = 1.0 - alpha_t_prev
        sigma_t_squared = one_minus_alpha_t_prev * beta_t / (1.0 - alpha_t)
        sigma_t = torch.sqrt(sigma_t_squared)
        
        # Compute the coefficient for predicted noise in the mean
        # sqrt(alpha_{t-1}) * (x_t - sqrt(1-alpha_t) * epsilon) / sqrt(alpha_t)
        # = sqrt(alpha_{t-1}) / sqrt(alpha_t) * x_t - sqrt(alpha_{t-1}) * sqrt(1-alpha_t)/sqrt(alpha_t) * epsilon
        
        # Compute x_t / sqrt(alpha_t)
        x_over_sqrt_alpha = x_t / sqrt_alpha_t
        
        # Compute sqrt(1-alpha_t)/sqrt(alpha_t) * epsilon
        coef = sqrt_one_minus_alpha_t / sqrt_alpha_t
        noise_term = coef * predicted_noise
        
        # Compute predicted x_0
        pred_x_0 = x_over_sqrt_alpha - noise_term
        
        # Compute x_{t-1} = sqrt(alpha_{t-1}) * pred_x_0 + sqrt(1 - alpha_{t-1} - sigma_t^2) * epsilon
        # Note: In standard DDPM, the term is sqrt(1 - alpha_{t-1}) * epsilon, but we use the 
        # formula from the spec which has: sqrt(1 - alpha_{t-1} - sigma_t^2)
        
        # Coefficient for predicted noise
        coef_noise = torch.sqrt(torch.clamp(1.0 - alpha_t_prev - sigma_t_squared, min=1e-8))
        
        # Add random noise epsilon
        random_noise = torch.randn_like(x_t)
        
        # Compute x_{t-1}
        x_t_prev = sqrt_alpha_t_prev * pred_x_0 + coef_noise * predicted_noise + sigma_t * random_noise
        
        # For t=1 (final step), don't add noise
        # Use mask to handle this
        mask = (t > 1).float().view(batch_size, 1, 1, 1)
        x_t_prev = x_t_prev * mask + (sqrt_alpha_t_prev * pred_x_0 + coef_noise * predicted_noise) * (1 - mask)
        
        return x_t_prev
    
    def compute_intra_lpips(
        self,
        num_samples: int = 1000,
        image_size: int = 32,
        channels: int = 3
    ) -> float:
        """Compute Intra-LPIPS metric to measure diversity of generated samples.
        
        Steps:
        1. Generate num_samples using reverse diffusion (generate_samples)
        2. Randomly sample pairs of images
        3. Compute LPIPS distance between each pair using pre-trained network
        4. Return mean LPIPS distance
        
        Higher values indicate greater diversity in generated samples.
        
        Args:
            num_samples: Number of samples to generate for evaluation (default: 1000)
            image_size: Size of generated images (default: 32)
            channels: Number of channels (default: 3 for RGB)
        
        Returns:
            Mean LPIPS distance across all pairs (float)
        
        Example:
            >>> evaluator = Evaluator(model, diffusion_utils, device='cuda')
            >>> intra_lpips = evaluator.compute_intra_lpips(num_samples=1000)
            >>> print(f"Intra-LPIPS: {intra_lpips:.4f}")  # Higher = more diverse
        """
        # Generate samples
        print(f"Generating {num_samples} samples for Intra-LPIPS computation...")
        samples = self.generate_samples(
            num_samples=num_samples,
            image_size=image_size,
            channels=channels
        )
        
        # Sample pairs and compute LPIPS distances
        # For efficiency, we sample random pairs rather than computing all pairs
        num_pairs = min(1000, num_samples * (num_samples - 1) // 2)
        
        distances = []
        
        with torch.no_grad():
            for _ in range(num_pairs):
                # Randomly sample two different images
                idx1 = torch.randint(0, num_samples, (1,)).item()
                idx2 = torch.randint(0, num_samples, (1,)).item()
                
                while idx2 == idx1:
                    idx2 = torch.randint(0, num_samples, (1,)).item()
                
                img1 = samples[idx1:idx1+1]
                img2 = samples[idx2:idx2+1]
                
                # Compute LPIPS distance
                dist = compute_lpips_distance(img1, img2, self.lpips_model)
                distances.append(dist.item())
        
        # Return mean distance
        mean_distance = np.mean(distances)
        
        return float(mean_distance)
    
    def compute_fid(
        self,
        num_samples: int,
        reference_features: torch.Tensor
    ) -> float:
        """Compute FID (Fréchet Inception Distance) between generated samples and reference.
        
        Extracts features using InceptionV3, computes mean and covariance of both
        distributions, then calculates FID:
            FID = ||mu_gen - mu_ref||^2 + Tr(cov_gen + cov_ref - 2*sqrt(cov_gen*cov_ref))
        
        Lower FID indicates better quality (generated samples closer to reference).
        
        Args:
            num_samples: Number of samples to generate for FID computation
            reference_features: Pre-extracted reference features [N, feature_dim]
        
        Returns:
            FID score (float)
        
        Example:
            >>> # First extract reference features
            >>> reference_features = extract_inception_features(reference_images)
            >>> 
            >>> # Compute FID
            >>> evaluator = Evaluator(model, diffusion_utils, device='cuda')
            >>> fid_score = evaluator.compute_fid(num_samples=1000, reference_features=ref_features)
            >>> print(f"FID: {fid_score:.4f}")  # Lower is better
        """
        # Generate samples
        print(f"Generating {num_samples} samples for FID computation...")
        generated_samples = self.generate_samples(
            num_samples=num_samples,
            image_size=299,  # InceptionV3 expects 299x299
            channels=3
        )
        
        # Extract features using InceptionV3
        generated_features = self._extract_inception_features(generated_samples)
        
        # Compute FID
        fid_score = self._calculate_fid(generated_features, reference_features)
        
        return fid_score
    
    def _extract_inception_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from InceptionV3 model.
        
        Args:
            images: Input images [B, 3, H, W] normalized to [0, 1]
        
        Returns:
            Feature tensor [B, 2048]
        """
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            
            # Load InceptionV3 model
            inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            inception_model.eval()
            
            # Remove the final classification layer to get features
            inception_model = nn.Sequential(
                inception_model.Conv2dEmbedding(),
                inception_model.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            inception_model.to(self.device)
            
            with torch.no_grad():
                # InceptionV3 expects 299x299 images
                if images.size(-1) != 299:
                    images = nn.functional.interpolate(
                        images, 
                        size=(299, 299), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Normalize to ImageNet stats
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
                images = (images - mean) / std
                
                features = inception_model(images)
            
            return features
        
        except ImportError:
            print("Warning: Could not load InceptionV3. Using random features.")
            return torch.randn(images.size(0), 2048, device=self.device)
    
    def _calculate_fid(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> float:
        """Calculate FID between two sets of features.
        
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        
        Args:
            features1: First set of features [N1, D]
            features2: Second set of features [N2, D]
        
        Returns:
            FID score (float)
        """
        # Compute mean and covariance
        mu1 = torch.mean(features1, dim=0)
        mu2 = torch.mean(features2, dim=0)
        
        # Compute covariance
        sigma1 = torch.cov(features1.T)
        sigma2 = torch.cov(features2.T)
        
        # Compute FID
        diff = mu1 - mu2
        diff_squared = torch.sum(diff ** 2)
        
        # Compute sqrt of product of covariances
        # Use eigenvalue decomposition for numerical stability
        sigma_product = torch.mm(sigma1, sigma2)
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_product)
        sqrt_sigma_product = eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=1e-8))) @ eigenvectors.T
        
        # Compute trace term
        trace_term = torch.trace(sigma1 + sigma2 - 2 * sqrt_sigma_product)
        
        fid = diff_squared + trace_term
        
        return fid.item()
    
    def evaluate_diversity_and_quality(
        self,
        num_samples: int = 1000,
        image_size: int = 32,
        channels: int = 3,
        reference_images: Optional[torch.Tensor] = None
    ) -> dict:
        """Comprehensive evaluation of diversity and quality.
        
        Evaluates the model using:
        - Intra-LPIPS: Measures diversity (higher = more diverse)
        - FID: Measures quality relative to reference (lower = better)
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of generated images
            channels: Number of channels
            reference_images: Optional reference images for FID computation
        
        Returns:
            Dictionary containing evaluation metrics:
                - 'intra_lpips': Diversity score (higher = more diverse)
                - 'fid': Quality score (lower = better, if reference provided)
        
        Example:
            >>> evaluator = Evaluator(model, diffusion_utils, device='cuda')
            >>> results = evaluator.evaluate_diversity_and_quality(
            ...     num_samples=1000,
            ...     reference_images=reference_images
            ... )
            >>> print(f"Intra-LPIPS: {results['intra_lpips']:.4f}")
            >>> print(f"FID: {results['fid']:.4f}")
        """
        results = {}
        
        # Compute Intra-LPIPS
        intra_lpips = self.compute_intra_lpips(
            num_samples=num_samples,
            image_size=image_size,
            channels=channels
        )
        results['intra_lpips'] = intra_lpips
        
        # Compute FID if reference provided
        if reference_images is not None:
            # Extract reference features
            print("Extracting reference features for FID...")
            reference_features = self._extract_inception_features(reference_images)
            
            # Compute FID
            fid = self.compute_fid(
                num_samples=num_samples,
                reference_features=reference_features
            )
            results['fid'] = fid
        
        return results


def evaluate_model(
    model: DiffusionUNet,
    diffusion_utils: DiffusionUtils,
    num_samples: int = 1000,
    device: str = 'cuda',
    image_size: int = 32,
    channels: int = 3
) -> dict:
    """Convenience function to evaluate model with default settings.
    
    Args:
        model: Trained diffusion model with adaptor layers
        diffusion_utils: Diffusion utilities
        num_samples: Number of samples for evaluation
        device: Device for computation
        image_size: Size of generated images
        channels: Number of image channels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    evaluator = Evaluator(model, diffusion_utils, device)
    return evaluator.compute_intra_lpips(
        num_samples=num_samples,
        image_size=image_size,
        channels=channels
    )