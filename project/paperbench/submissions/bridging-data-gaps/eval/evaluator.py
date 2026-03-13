## eval/evaluator.py
"""
Evaluator for DPMs-ANT that computes standard few-shot image generation metrics.

Implements the evaluation pipeline described in Section 5.2 of the paper:
- Generates images using trained diffusion model
- Computes Intra-LPIPS (diversity metric)
- Computes FID (quality metric) when sufficient real data available

All configurations are sourced from config.yaml to ensure consistency across the pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple, Optional
import lpips
import subprocess
import tempfile
from PIL import Image

# Import dependencies
from config import config
from utils.noise_scheduler import DDPMNoiseScheduler


class Evaluator:
    """
    Evaluator class for assessing performance of trained DPMs-ANT model.
    
    Computes two key metrics:
    - Intra-LPIPS: Measures diversity within clusters of generated images
    - FID: Measures quality by comparing feature distributions (when large real dataset available)
    
    Also handles image generation via ancestral sampling or DDIM.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda'):
        """
        Initialize evaluator with trained model and device.
        
        Args:
            model: Trained diffusion model (UNetWithAdaptor)
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Create noise scheduler
        self.scheduler = DDPMNoiseScheduler()
        self.scheduler.to(self.device)
        
        # Initialize LPIPS metric
        self.lpips_metric = lpips.LPIPS(net='alex').to(self.device)
        
        # Create results directory
        self.results_dir = Path(config.logging.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"Initialized Evaluator on {self.device}")
    
    def generate_images(self, num_samples: int) -> List[torch.Tensor]:
        """
        Generate images using ancestral sampling from the trained diffusion model.
        
        Implements reverse diffusion process:
        x_{t-1} = (x_t - sqrt(1-ᾱ_t)*ε_θ(x_t,t)) / sqrt(ᾱ_t) * sqrt(ᾱ_{t-1}) + sqrt(1-ᾱ_{t-1}-σ^2)*ε_θ + σ*ε
        
        For simplicity, uses DDPM-style sampling with η=0 (deterministic component).
        
        Args:
            num_samples: Number of images to generate
            
        Returns:
            List of generated image tensors in range [-1, 1]
        """
        # Get configuration values
        image_size = config.dataset.image_size
        batch_size = min(8, num_samples)  # Limit batch size for memory efficiency
        
        # Extract coefficients from scheduler
        alphas_dict = self.scheduler.get_alphas()
        alphas_cumprod = alphas_dict['alphas_cumprod']  # Shape: (T,)
        sqrt_alphas_cumprod = alphas_dict['sqrt_alphas_cumprod']
        sqrt_one_minus_alphas_cumprod = alphas_dict['sqrt_one_minus_alphas_cumprod']
        
        generated_images = []
        
        with torch.no_grad():
            # Process in batches
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                current_batch_size = min(batch_size, num_samples - len(generated_images))
                
                # Start from pure noise at timestep T
                x_t = torch.randn(current_batch_size, 3, image_size, image_size, device=self.device)
                
                # Reverse diffusion process: t = T, T-1, ..., 1
                for t in reversed(range(1, self.scheduler.T + 1)):
                    t_tensor = torch.full((current_batch_size,), t, device=self.device, dtype=torch.long)
                    
                    # Predict noise
                    eps_pred = self.model(x_t, t_tensor)
                    
                    if t > 1:
                        # Compute coefficients for this timestep
                        alpha_t = alphas_dict['alphas'][t-1]  # α_t
                        alpha_cumprod_t = alphas_cumprod[t-1]  # ᾱ_t
                        alpha_cumprod_tm1 = alphas_cumprod[t-2] if t > 1 else torch.tensor(1.0, device=self.device)
                        
                        # Compute mean and variance components
                        # Mean: (sqrt(ᾱ_{t-1}) * β_t / (1 - ᾱ_t)) * ε_θ + (sqrt(α_t) * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)) * x_t
                        coef_eps = (torch.sqrt(alpha_cumprod_tm1) * (1 - alpha_t)) / (1 - alpha_cumprod_t)
                        coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_cumprod_tm1)) / (1 - alpha_cumprod_t)
                        
                        mean = coef_eps * eps_pred + coef_xt * x_t
                        
                        # Variance: σ^2 = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
                        sigma = torch.sqrt((1 - alpha_cumprod_tm1) / (1 - alpha_cumprod_t) * (1 - alpha_t))
                        
                        # Add noise if not last step
                        z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
                        x_t = mean + sigma * z
                    else:
                        # Final step (t=1): deterministic
                        x_t = (x_t - sqrt_one_minus_alphas_cumprod[t-1] * eps_pred) / sqrt_alphas_cumprod[t-1]
                
                # Clamp to valid range and convert to list
                x_0 = torch.clamp(x_t, -1, 1)
                generated_images.extend([img.cpu() for img in x_0])
                
                print(f"Generated batch {batch_idx+1}/{num_batches}")
        
        return generated_images
    
    def compute_intra_lpips(self, 
                           gen_images: List[torch.Tensor], 
                           real_images: List[torch.Tensor]) -> float:
        """
        Compute Intra-LPIPS score measuring diversity within clusters.
        
        Procedure:
        1. Assign each generated image to nearest real image (min LPIPS)
        2. For each cluster, compute average pairwise LPIPS among generated images
        3. Average across all clusters
        
        Higher value indicates greater diversity.
        
        Args:
            gen_images: List of generated image tensors in range [-1, 1]
            real_images: List of real target images (10-shot) in range [-1, 1]
            
        Returns:
            Intra-LPIPS score (float)
        """
        if len(gen_images) == 0:
            raise ValueError("No generated images provided")
        if len(real_images) == 0:
            raise ValueError("No real images provided")
        
        # Convert lists to tensors and move to device
        gen_tensors = torch.stack([img for img in gen_images]).to(self.device)
        real_tensors = torch.stack([img for img in real_images]).to(self.device)
        
        # Resize images if necessary
        image_size = config.dataset.image_size
        if gen_tensors.shape[-1] != image_size or real_tensors.shape[-1] != image_size:
            from torchvision.transforms import Resize
            resize = Resize((image_size, image_size))
            gen_tensors = resize(gen_tensors)
            real_tensors = resize(real_tensors)
        
        # Normalize to [0, 1] for LPIPS
        gen_norm = (gen_tensors + 1) / 2
        real_norm = (real_tensors + 1) / 2
        
        # Ensure 4D tensors
        if gen_norm.dim() == 3:
            gen_norm = gen_norm.unsqueeze(0)
        if real_norm.dim() == 3:
            real_norm = real_norm.unsqueeze(0)
        
        # Compute distance matrix: [G, R]
        dist_matrix = torch.zeros(len(gen_images), len(real_images), device=self.device)
        
        with torch.no_grad():
            for i, gen_img in enumerate(gen_norm):
                for j, real_img in enumerate(real_norm):
                    dist = self.lpips_metric(gen_img.unsqueeze(0), real_img.unsqueeze(0))
                    dist_matrix[i, j] = dist.item()
        
        # Assign each generated image to closest real image
        assignments = torch.argmin(dist_matrix, dim=1).cpu().numpy()  # Shape: (G,)
        
        # Compute intra-cluster diversity
        intra_cluster_scores = []
        
        for cluster_id in range(len(real_images)):
            cluster_mask = (assignments == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < 2:
                continue  # Need at least 2 images for pairwise distance
                
            # Extract images in this cluster
            cluster_imgs = gen_norm[cluster_indices]
            
            # Compute pairwise LPIPS distances within cluster
            n = len(cluster_imgs)
            total_dist = 0.0
            count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.lpips_metric(cluster_imgs[i].unsqueeze(0), cluster_imgs[j].unsqueeze(0))
                    total_dist += dist.item()
                    count += 1
            
            if count > 0:
                intra_cluster_avg = total_dist / count
                intra_cluster_scores.append(intra_cluster_avg)
        
        # Final score: average across clusters
        if len(intra_cluster_scores) == 0:
            return 0.0  # All clusters have fewer than 2 images
            
        final_score = float(np.mean(intra_cluster_scores))
        return final_score
    
    def compute_fid(self, gen_features: torch.Tensor, real_features: torch.Tensor) -> float:
        """
        Compute Fréchet Inception Distance (FID) between generated and real features.
        
        This method provides the core FID computation logic. In practice, it's recommended
        to use external tools like pytorch-fid which handle feature extraction properly.
        
        Args:
            gen_features: Generated image features from Inception-V3 (N_gen, 2048)
            real_features: Real image features from Inception-V3 (N_real, 2048)
            
        Returns:
            FID score (float)
        """
        # Convert to numpy
        g = gen_features.cpu().numpy()
        r = real_features.cpu().numpy()
        
        # Handle different sizes by truncating larger set
        min_size = min(g.shape[0], r.shape[0])
        g = g[:min_size]
        r = r[:min_size]
        
        # Compute means
        mu_g = np.mean(g, axis=0)
        mu_r = np.mean(r, axis=0)
        
        # Compute covariance matrices
        sigma_g = np.cov(g, rowvar=False)
        sigma_r = np.cov(r, rowvar=False)
        
        # Compute squared distance between means
        diff = mu_g - mu_r
        mean_diff = diff.dot(diff)
        
        # Compute trace term
        # Use SVD for numerical stability
        try:
            sqrt_sigma_r = self._sqrt_cov(sigma_r)
            sqrt_prod = self._sqrt_cov(sqrt_sigma_r @ sigma_g @ sqrt_sigma_r)
            trace_term = np.trace(sigma_g) + np.trace(sigma_r) - 2 * np.trace(sqrt_prod)
        except Exception as e:
            print(f"Numerical error in FID computation: {e}")
            trace_term = 0.0
        
        fid = mean_diff + trace_term
        return float(fid)
    
    def _sqrt_cov(self, matrix: np.ndarray) -> np.ndarray:
        """Compute square root of covariance matrix using SVD."""
        u, s, vh = np.linalg.svd(matrix)
        return u @ np.diag(np.sqrt(s)) @ vh
    
    def save_images(self, images: List[torch.Tensor], directory: str) -> None:
        """
        Save generated images to disk for FID computation.
        
        Args:
            images: List of image tensors in range [-1, 1]
            directory: Directory path to save images
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Denormalize from [-1,1] to [0,1]
        for i, img_tensor in enumerate(images):
            # Convert tensor to PIL Image
            img_np = img_tensor.cpu().numpy()
            # Convert from CHW to HWC and [-1,1] to [0,1]
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np + 1) / 2
            img_np = np.clip(img_np, 0, 1)
            
            # Convert to uint8
            img_uint8 = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            
            # Save
            pil_img.save(dir_path / f"sample_{i:06d}.png")
        
        print(f"Saved {len(images)} images to {directory}")


# Example usage and testing
if __name__ == "__main__":
    try:
        # Create dummy model for testing
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
            
            def forward(self, x, timesteps=None):
                return self.conv(x)
        
        # Initialize evaluator
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dummy_model = DummyModel()
        evaluator = Evaluator(dummy_model, device=device)
        
        # Test image generation
        print("Testing image generation...")
        gen_images = evaluator.generate_images(num_samples=4)
        print(f"Generated {len(gen_images)} images, each of shape {gen_images[0].shape}")
        
        # Test Intra-LPIPS with dummy data
        print("\nTesting Intra-LPIPS...")
        real_images = [torch.randn(3, 256, 256) for _ in range(10)]
        intra_lpips_score = evaluator.compute_intra_lpips(gen_images, real_images)
        print(f"Intra-LPIPS score: {intra_lpips_score:.6f}")
        
        # Test saving images
        test_dir = "test_output"
        evaluator.save_images(gen_images, test_dir)
        print(f"Images saved to {test_dir}")
        
        # Cleanup
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
    except Exception as e:
        print(f"Error during testing: {e}")
