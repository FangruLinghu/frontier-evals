## utils/noise_scheduler.py
"""
DDPM Noise Scheduler for DPMs-ANT.

Implements the forward diffusion process and provides essential time-dependent coefficients
used throughout training, adversarial noise generation, and evaluation. This scheduler follows
the standard DDPM framework with a linear beta schedule as defined in Ho et al. (2020).

Key functionalities:
- Precompute and store diffusion parameters (betas, alphas, cumulative products)
- Apply forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
- Provide access to derived terms like σ̂_t used in similarity-guided loss

All configuration values are sourced from config.yaml to ensure consistency across the pipeline.
"""

import torch
import numpy as np
from typing import Dict, Optional, Union
from dataclasses import dataclass

# Import configuration
from config import config


@dataclass
class NoiseScheduleConfig:
    """Configuration specific to noise scheduling."""
    T: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02


class DDPMNoiseScheduler:
    """
    Noise scheduler for Denoising Diffusion Probabilistic Models (DDPM).
    
    Implements the forward diffusion process and provides precomputed coefficients
    for efficient access during training and sampling. Uses a linear beta schedule
    over T timesteps as specified in the paper.
    
    The scheduler operates with one-based indexing (t ∈ [1, T]) to align with the
    mathematical formulation in the paper, storing arrays of size (T+1) where index 0 is unused.
    """
    
    def __init__(self, 
                 T: Optional[int] = None,
                 beta_start: Optional[float] = None,
                 beta_end: Optional[float] = None):
        """
        Initialize the noise scheduler with linear beta schedule.
        
        Args:
            T: Total number of diffusion timesteps. If None, uses value from config.
            beta_start: Starting value of beta schedule. If None, uses value from config.
            beta_end: Ending value of beta schedule. If None, uses value from config.
            
        Raises:
            ValueError: If any parameter is non-positive or if beta_start >= beta_end
        """
        # Use provided values or fall back to config
        self.T = T if T is not None else config.model.T
        self.beta_start = beta_start if beta_start is not None else config.model.beta_start
        self.beta_end = beta_end if beta_end is not None else config.model.beta_end
        
        # Validate inputs
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.beta_start <= 0 or self.beta_end <= 0:
            raise ValueError(f"Beta values must be positive, got beta_start={self.beta_start}, beta_end={self.beta_end}")
        if self.beta_start >= self.beta_end:
            raise ValueError(f"beta_start must be less than beta_end, got {self.beta_start} >= {self.beta_end}")
        
        # Generate beta schedule (linear)
        betas = torch.linspace(self.beta_start, self.beta_end, self.T)
        
        # Compute alphas: α_t = 1 - β_t
        alphas = 1.0 - betas
        
        # Compute cumulative product of alphas: ᾱ_t = ∏_{s=1}^t α_s
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Compute square roots for efficient computation
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        # Register all tensors as buffers (non-trainable, automatically moved to device)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        
        # Precompute sigma_hat_t = (1 - ᾱ_{t-1}) * sqrt(α_t / (1 - ᾱ_t))
        # Used in similarity-guided training loss (Equation 5)
        sigma_hat = torch.zeros_like(alphas_cumprod)
        
        # For t=1, use convention that ᾱ_0 = 1
        alpha_ratio = alphas / (1.0 - alphas_cumprod + 1e-8)  # Add small epsilon for numerical stability
        sigma_hat[1:] = (1.0 - alphas_cumprod[:-1]) * torch.sqrt(alpha_ratio[1:])
        sigma_hat[0] = 0.0  # Not used since t starts at 1
        
        self.register_buffer('sigma_hat', sigma_hat)
    
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """
        Register a tensor as a buffer (non-trainable parameter).
        
        This method allows us to store precomputed values that should be moved to the
        same device as the model but don't require gradients.
        
        Args:
            name: Name of the buffer
            tensor: Tensor to register
        """
        setattr(self, name, tensor)
    
    def add_noise(self, 
                  x0: torch.Tensor, 
                  t: Union[torch.Tensor, int], 
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply forward diffusion process to add noise to clean images.
        
        Computes: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        
        Args:
            x0: Clean input images of shape (B, C, H, W)
            t: Timestep(s) at which to compute x_t. Can be scalar or tensor of shape (B,)
            noise: Optional noise tensor to use. If None, generates standard Gaussian noise.
            
        Returns:
            Noisy images x_t of same shape as x0
            
        Raises:
            ValueError: If t is out of valid range [1, T]
            RuntimeError: If tensor dimensions don't match
        """
        B, C, H, W = x0.shape
        
        # Convert t to tensor if it's a scalar
        if isinstance(t, int):
            t = torch.tensor([t] * B, dtype=torch.long, device=x0.device)
        elif isinstance(t, torch.Tensor):
            t = t.to(dtype=torch.long, device=x0.device)
            if t.dim() == 0:
                t = t.unsqueeze(0).expand(B)
        else:
            raise TypeError(f"t must be int or torch.Tensor, got {type(t)}")
        
        # Validate timestep range (one-based indexing: t ∈ [1, T])
        if (t < 1).any() or (t > self.T).any():
            raise ValueError(f"Timestep t must be in [1, {self.T}], got min={t.min().item()}, max={t.max().item()}")
        
        # Adjust for one-based indexing in our arrays
        # Our buffers are indexed 0..T-1 for timesteps 1..T
        t_idx = t - 1  # Convert to zero-based index
        
        # Get coefficients at timestep t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_idx]  # Shape: (B,)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_idx]  # Shape: (B,)
        
        # Reshape coefficients to broadcast with image tensor
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(x0)
        else:
            if noise.shape != x0.shape:
                raise RuntimeError(f"Noise tensor shape {noise.shape} doesn't match x0 shape {x0.shape}")
        
        # Compute x_t
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return xt
    
    def get_alphas(self) -> Dict[str, torch.Tensor]:
        """
        Get all precomputed alpha-related tensors.
        
        Returns:
            Dictionary containing:
            - 'alphas': α_t = 1 - β_t
            - 'alphas_cumprod': ᾱ_t = ∏_{s=1}^t α_s
            - 'sqrt_alphas_cumprod': √ᾱ_t
            - 'sqrt_one_minus_alphas_cumprod': √(1 - ᾱ_t)
            - 'betas': β_t
            - 'sigma_hat': (1 - ᾱ_{t-1}) * √(α_t / (1 - ᾱ_t)) for similarity guidance
        """
        return {
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
            'betas': self.betas,
            'sigma_hat': self.sigma_hat
        }
    
    def get_sigma_hat(self, t: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        Get σ̂_t coefficient used in similarity-guided training loss.
        
        σ̂_t = (1 - ᾱ_{t-1}) * √(α_t / (1 - ᾱ_t))
        
        Args:
            t: Timestep(s) to query. Can be scalar or tensor.
            
        Returns:
            σ̂_t values of appropriate shape
            
        Raises:
            ValueError: If t is out of valid range [1, T]
        """
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.long)
        else:
            t = t.to(dtype=torch.long)
        
        # Validate range
        if (t < 1).any() or (t > self.T).any():
            raise ValueError(f"Timestep t must be in [1, {self.T}] for sigma_hat, got {t}")
        
        # Convert to zero-based index
        t_idx = t - 1
        return self.sigma_hat[t_idx].view(-1, 1, 1, 1)  # Reshape for broadcasting
    
    def timesteps(self) -> torch.Tensor:
        """
        Get all valid timesteps as a tensor.
        
        Returns:
            Tensor of integers from 1 to T inclusive
        """
        return torch.arange(1, self.T + 1, dtype=torch.long, device=self.betas.device)


# Example usage and testing
if __name__ == "__main__":
    # Create noise scheduler
    scheduler = DDPMNoiseScheduler()
    print(f"Initialized DDPMNoiseScheduler with T={scheduler.T}")
    
    # Test basic properties
    alphas_dict = scheduler.get_alphas()
    print(f"Alpha dictionary keys: {list(alphas_dict.keys())}")
    print(f"Betas shape: {alphas_dict['betas'].shape}")
    print(f"Alphas_cumprod[0]: {alphas_dict['alphas_cumprod'][0]:.6f}")
    print(f"Alphas_cumprod[-1]: {alphas_dict['alphas_cumprod'][-1]:.6f}")
    
    # Test add_noise with sample data
    try:
        # Create dummy data
        x0 = torch.randn(4, 3, 64, 64)  # Batch of 4 images
        t = torch.tensor([100, 200, 500, 900])  # Different timesteps for each sample
        noise = torch.randn_like(x0)
        
        # Apply noise
        xt = scheduler.add_noise(x0, t, noise)
        print(f"Successfully applied noise: x0 shape {x0.shape} -> xt shape {xt.shape}")
        
        # Test sigma_hat
        sigma_hats = scheduler.get_sigma_hat(t)
        print(f"Sigma_hat values: {sigma_hats.squeeze()}")
        
        # Test edge cases
        try:
            scheduler.add_noise(x0, 0)  # Should fail - t=0 invalid
        except ValueError as e:
            print(f"Correctly caught invalid timestep: {e}")
            
        try:
            scheduler.add_noise(x0, scheduler.T + 1)  # Should fail - t>T
        except ValueError as e:
            print(f"Correctly caught invalid timestep: {e}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
