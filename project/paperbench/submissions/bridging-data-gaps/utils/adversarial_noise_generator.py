## utils/adversarial_noise_generator.py
"""
Adversarial Noise Generator for DPMs-ANT.

Implements the adversarial noise selection strategy described in Section 4.2 of the paper.
Generates 'worst-case' noise inputs that maximize reconstruction error for the current model,
thereby accelerating convergence and improving robustness in few-shot transfer learning.

Key features:
- Finite-step gradient ascent to find adversarial noise ε* (Equation 6)
- Normalization after each step to maintain noise statistics (Eq 7, Norm function)
- Compatible with frozen pre-trained diffusion models
- Configurable via config.yaml: J (steps), omega (step size), norm_method

All configuration values are sourced from config.yaml to ensure consistency across the pipeline.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from dataclasses import dataclass

# Import dependencies
from config import config, get_adaptor_reduction_factor
from utils.noise_scheduler import DDPMNoiseScheduler


NormMethod = Literal["batch_norm", "clip_and_scale", "project_sphere"]


class AdversarialNoiseGenerator:
    """
    Generates adversarial noise through finite-step gradient ascent to identify 'worst-case'
    noise patterns that are hardest for the current model to denoise.

    This forces the adaptor module to focus on correcting the most challenging cases first,
    leading to faster convergence and improved performance under limited data conditions.

    Implements Equation (6) and (7) from the paper:
        max_ε ||ε - ϵ_θ(x_t, t)||^2
        ε^{j+1} = Norm(ε^j + ω * ∇_{ε^j} ||ε^j - ϵ_θ(x_t^j, t)||^2)

    Where Norm() maintains zero mean and unit standard deviation of the noise tensor.
    """

    def __init__(self, 
                 model: nn.Module,
                 noise_scheduler: DDPMNoiseScheduler,
                 device: Optional[torch.device] = None):
        """
        Initialize adversarial noise generator.

        Args:
            model: Pre-trained diffusion model ϵ_θ (frozen during noise generation)
            noise_scheduler: DDPMNoiseScheduler instance for forward diffusion
            device: Device to run computations on. If None, uses CUDA if available
            
        Raises:
            ValueError: If model or scheduler is invalid
        """
        if model is None:
            raise ValueError("Model cannot be None")
        if noise_scheduler is None:
            raise ValueError("Noise scheduler cannot be None")

        self.model = model
        self.noise_scheduler = noise_scheduler
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device if needed (but don't modify original)
        self.model.to(self.device)
        
        # Get hyperparameters from config
        self.J = config.adversarial_noise.J  # Number of inner steps
        self.omega = config.adversarial_noise.omega  # Step size for ascent
        self.norm_method = config.adversarial_noise.norm_method  # Normalization method
        
        # Ensure model is in eval mode (frozen weights)
        self.model.eval()
        
        # Verify no parameters are being tracked for gradients
        for param in self.model.parameters():
            param.requires_grad = False
    
    def generate(self, 
                x0: torch.Tensor, 
                t: torch.Tensor, 
                eps_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate adversarial noise ε* via finite-step gradient ascent.

        For a given clean image x0 and timestep t, this method finds the noise vector ε*
        that maximizes the model's reconstruction error, making it the 'worst-case' input.

        Args:
            x0: Clean target images of shape (B, C, H, W)
            t: Timesteps of shape (B,) with values in [1, T]
            eps_init: Initial noise tensor. If None, sampled from N(0,I)
            
        Returns:
            Adversarial noise ε* of same shape as x0
            
        Raises:
            ValueError: If inputs have invalid shapes or values
            RuntimeError: If computation fails
        """
        # Validate inputs
        if x0.dim() != 4:
            raise ValueError(f"x0 must be 4D tensor (B,C,H,W), got {x0.dim()}D")
        if t.dim() == 0:
            t = t.unsqueeze(0)  # Handle scalar t
        if t.dim() != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError(f"t must be 1D tensor with length matching batch size, got {t.shape}")
        if t.min() < 1 or t.max() > self.noise_scheduler.T:
            raise ValueError(f"Timestep t must be in [1, {self.noise_scheduler.T}], got [{t.min().item()}, {t.max().item()}]")
        
        # Move inputs to device
        x0 = x0.to(self.device)
        t = t.to(self.device)
        
        # Initialize noise if not provided
        if eps_init is None:
            eps_j = torch.randn_like(x0, device=self.device)
        else:
            if eps_init.shape != x0.shape:
                raise ValueError(f"eps_init shape {eps_init.shape} doesn't match x0 shape {x0.shape}")
            eps_j = eps_init.clone().detach().to(self.device)
        
        # Ensure eps_j requires gradient
        eps_j.requires_grad_(True)
        
        # Extract coefficients from noise scheduler
        alphas_dict = self.noise_scheduler.get_alphas()
        alphas_cumprod = alphas_dict['alphas_cumprod']  # Shape: (T,)
        
        # Get coefficients at specified timesteps (convert to zero-based index)
        t_idx = t - 1  # Convert to zero-based indexing
        sqrt_alphas_cumprod_t = alphas_dict['sqrt_alphas_cumprod'][t_idx]  # Shape: (B,)
        sqrt_one_minus_alphas_cumprod_t = alphas_dict['sqrt_one_minus_alphas_cumprod'][t_idx]  # Shape: (B,)
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        # Finite-step gradient ascent loop
        for j in range(self.J):
            # Zero gradients from previous iteration
            if eps_j.grad is not None:
                eps_j.grad.zero_()
            
            # Compute noised image xt^j = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε^j
            # Gradients will flow through ε^j here
            xt_j = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * eps_j
            
            # Forward pass through frozen model to get predicted noise
            # Detach xt_j from any prior graph but keep as leaf for gradient computation
            with torch.no_grad():
                # Model expects tensors requiring grad only for input noise
                eps_pred = self.model(xt_j.detach(), t)
            
            # Compute loss: ||ε^j - ϵ_θ(xt^j, t)||^2
            # We want to maximize this, so we'll minimize the negative
            loss = torch.mean((eps_j - eps_pred) ** 2)
            
            # Backward pass to compute gradient w.r.t. ε^j
            grad = torch.autograd.grad(
                outputs=loss,
                inputs=eps_j,
                retain_graph=False,
                create_graph=False,
                allow_unused=False
            )[0]
            
            # Update noise using gradient ascent
            with torch.no_grad():
                # Apply update: ε^{j+1} = ε^j + ω * ∇_ε L
                eps_next = eps_j + self.omega * grad
                
                # Apply normalization to maintain noise statistics
                eps_next = self.norm(eps_next)
                
                # Create new tensor that requires grad for next iteration
                eps_j = eps_next.clone().detach().requires_grad_(True)
        
        # Return final adversarial noise (detached from graph)
        return eps_j.detach()

    def norm(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Normalize noise tensor to maintain zero mean and unit standard deviation.

        Implements the Norm() function from Equation (7). Different methods can be used
        based on configuration, but all aim to prevent noise from drifting too far from N(0,I).

        Args:
            noise: Input noise tensor of shape (B, C, H, W)
            
        Returns:
            Normalized noise tensor of same shape
            
        Raises:
            ValueError: If norm_method is unsupported
        """
        B, C, H, W = noise.shape
        
        if self.norm_method == "batch_norm":
            # Normalize over spatial and channel dimensions per sample
            # Subtract mean and divide by std across C, H, W for each batch element
            noise_flat = noise.view(B, -1)
            noise_normalized = (noise_flat - noise_flat.mean(dim=1, keepdim=True)) / \
                              (noise_flat.std(dim=1, keepdim=True) + 1e-8)
            return noise_normalized.view(B, C, H, W)
            
        elif self.norm_method == "clip_and_scale":
            # Clip to [-3, 3] then normalize
            noise_clipped = torch.clamp(noise, -3.0, 3.0)
            noise_flat = noise_clipped.view(B, -1)
            noise_normalized = (noise_flat - noise_flat.mean(dim=1, keepdim=True)) / \
                              (noise_flat.std(dim=1, keepdim=True) + 1e-8)
            return noise_normalized.view(B, C, H, W)
            
        elif self.norm_method == "project_sphere":
            # Project onto sphere with radius sqrt(C*H*W)
            # This preserves magnitude while allowing direction changes
            noise_flat = noise.view(B, -1)
            norms = torch.norm(noise_flat, dim=1, keepdim=True)
            target_norm = torch.sqrt(torch.tensor(C * H * W, dtype=torch.float32, device=noise.device))
            noise_normalized = noise_flat * (target_norm / (norms + 1e-8))
            return noise_normalized.view(B, C, H, W)
            
        else:
            raise ValueError(f"Unsupported norm_method: {self.norm_method}. "
                           f"Use 'batch_norm', 'clip_and_scale', or 'project_sphere'.")

    def __repr__(self) -> str:
        """String representation showing configuration."""
        return (f"AdversarialNoiseGenerator(J={self.J}, omega={self.omega}, "
                f"norm_method='{self.norm_method}', device={self.device})")


# Example usage and testing
if __name__ == "__main__":
    try:
        # Create dummy components for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dummy model (simple identity-like behavior)
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
            
            def forward(self, x, t):
                return self.conv(x)  # Just a placeholder
        
        # Initialize components
        dummy_model = DummyModel().to(device)
        noise_scheduler = DDPMNoiseScheduler().to(device)
        
        # Create adversarial noise generator
        noise_gen = AdversarialNoiseGenerator(dummy_model, noise_scheduler, device)
        print(f"Initialized {noise_gen}")
        
        # Test with dummy data
        x0 = torch.randn(2, 3, 64, 64).to(device)  # Small batch for test
        t = torch.tensor([100, 500], device=device)  # Two different timesteps
        
        # Generate adversarial noise
        eps_star = noise_gen.generate(x0, t)
        print(f"Generated adversarial noise: shape {eps_star.shape}")
        print(f"Noise stats - Mean: {eps_star.mean().item():.6f}, Std: {eps_star.std().item():.6f}")
        
        # Test edge cases
        try:
            # Test invalid timestep
            noise_gen.generate(x0, torch.tensor([0], device=device))
        except ValueError as e:
            print(f"Correctly caught invalid timestep: {e}")
            
        try:
            # Test shape mismatch
            noise_gen.generate(x0, torch.tensor([100, 200, 300], device=device))
        except ValueError as e:
            print(f"Correctly caught shape mismatch: {e}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
