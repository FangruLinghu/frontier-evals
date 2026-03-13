## diffusion/utils.py

```python
"""Diffusion utilities for similarity-guided diffusion model adaptation.

This module implements the diffusion process utilities including:
- Linear beta schedule computation
- Forward diffusion process: q(x_t | x_0) = N(x_t; ᾱ_t x_0, (1-ᾱ_t)I)
- Reverse denoising process for DDPM sampling
- Precomputed tensors for fast computation

The noise schedule follows standard DDPM (Ho et al., 2020):
- β_t: linear schedule from beta_start to beta_end
- α_t = 1 - β_t
- ᾱ_t = ∏_{i=1}^t α_i (cumulative product)

Forward process: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
Reverse process: x_{t-1} = √ᾱ_{t-1}((x_t - √(1-ᾱ_t)ε_θ)/ᾱ_t) + √(1-α_{t-1})ε_θ
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from torch import Tensor


class DiffusionUtils:
    """Diffusion utilities for noise scheduling and forward/reverse processes.
    
    This class implements the core diffusion process computations needed for
    similarity-guided diffusion model training. It precomputes all required
    tensors for efficient forward and reverse processes.
    
    Attributes:
        timesteps: Total number of diffusion timesteps T
        beta_start: Starting value for beta schedule (default: 1e-4)
        beta_end: Ending value for beta schedule (default: 0.02)
        betas: Linear beta schedule β_1, ..., β_T
        alphas: α_t = 1 - β_t for t=1,...,T
        alphas_cumprod: ᾱ_t = ∏_{i=1}^t α_i for t=1,...,T
        alphas_cumprod_prev: ᾱ_{t-1} with ᾱ_0=1
        sqrt_alphas_cumprod: √ᾱ_t for forward process
        sqrt_one_minus_alphas_cumprod: √(1-ᾱ_t) for forward process
        sqrt_recip_alphas_cumprod: 1/√ᾱ_t for reverse process
        posterior_variance: σ_t² = (1-ᾱ_{t-1})/(1-ᾱ_t) * (1-α_t/ᾱ_{t-1})
        sigma_hat_t: σ̂_t = (1-ᾱ_{t-1})√(α_t/(1-ᾱ_t)) for similarity-guided loss
    """
    
    def __init__(
        self,
        timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ) -> None:
        """Initialize diffusion utilities with noise schedule.
        
        Step 1: Create linear beta schedule from beta_start to beta_end.
        Step 2: Compute α_t = 1 - β_t.
        Step 3: Compute ᾱ_t = ∏_{i=1}^t α_i (cumulative product).
        Step 4: Precompute all required tensors for fast forward/reverse processes.
        
        Args:
            timesteps: Total number of diffusion timesteps T
            beta_start: Starting value for beta schedule β_1 (default: 1e-4)
            beta_end: Ending value for beta schedule β_T (default: 0.02)
        """
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Step 1: Create linear beta schedule from beta_start to beta_end
        # β_t: linearly interpolated noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # Step 2: Compute α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # Step 3: Compute ᾱ_t = ∏_{i=1}^t α_i (cumulative product)
        # This is the cumulative product of alphas: ᾱ_t = α_1 * α_2 * ... * α_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Compute ᾱ_{t-1} with ᾱ_0 = 1 (for t=1, ᾱ_0 = 1)
        # We prepend a 1 at the beginning for easy indexing
        self.alphas_cumprod_prev = torch.cat([
            torch.ones(1),  # ᾱ_0 = 1
            self.alphas_cumprod[:-1]  # ᾱ_{t-1} for t >= 1
        ])
        
        # Step 4: Precompute derived quantities for fast forward/reverse processes
        
        # √ᾱ_t for forward process: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        
        # √(1-ᾱ_t) for forward process
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 1/√ᾱ_t for reverse process
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        
        # Posterior variance: σ_t² = (1-ᾱ_{t-1})/(1-ᾱ_t) * (1-α_t/ᾱ_{t-1})
        # This is used in DDPM sampling: x_{t-1} = μ_t + σ_t * ε_t
        # Compute numerator: 1 - ᾱ_{t-1}
        numerator = 1.0 - self.alphas_cumprod_prev
        # Compute denominator: 1 - ᾱ_t
        denominator = 1.0 - self.alphas_cumprod
        # Compute (1 - α_t / ᾱ_{t-1})
        term = 1.0 - self.alphas / self.alphas_cumprod_prev
        # Full formula: σ_t² = numerator/denominator * term
        self.posterior_variance = numerator / denominator * term
        
        # Also compute the square root for direct use
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        
        # σ̂_t = (1-ᾱ_{t-1})√(α_t/(1-ᾱ_t)) for similarity-guided loss
        # This is used in Equation 5 for scaling the classifier gradient
        # Compute α_t / (1 - ᾱ_t)
        alpha_ratio = self.alphas / (1.0 - self.alphas_cumprod)
        self.sigma_hat_t = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alpha_ratio)
    
    def add_noise(
        self,
        x_0: Tensor,
        t: Tensor,
        epsilon: Tensor
    ) -> Tensor:
        """Forward diffusion process: q(x_t | x_0) = N(x_t; ᾱ_t x_0, (1-ᾱ_t)I).
        
        Given clean image x_0 and random noise ε ~ N(0, I), compute noisy version:
            x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        
        This is the forward noising process used in training.
        
        Step 1: Index sqrt_alphas_cumprod at timestep t to get √ᾱ_t.
        Step 2: Index sqrt_one_minus_alphas_cumprod at timestep t to get √(1-ᾱ_t).
        Step 3: Compute x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * epsilon.
        
        Args:
            x_0: Clean image tensor [B, C, H, W]
            t: Timestep indices [B] (values in [0, timesteps-1])
            epsilon: Random noise tensor [B, C, H, W] ~ N(0, I)
        
        Returns:
            x_t: Noisy image tensor [B, C, H, W]
        
        Example:
            >>> utils = DiffusionUtils(timesteps=1000)
            >>> x_0 = torch.randn(4, 3, 32, 32)
            >>> t = torch.randint(0, 1000, (4,))
            >>> epsilon = torch.randn(4, 3, 32, 32)
            >>> x_t = utils.add_noise(x_0, t, epsilon)
        """
        # Step 1: Get √ᾱ_t for each sample in batch
        # t has shape [B], need to index into sqrt_alphas_cumprod [T]
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        
        # Step 2: Get √(1-ᾱ_t) for each sample in batch
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        # Step 3: Compute x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * epsilon
        
        return x_t
    
    def get_noise_schedule(self) -> Dict[str, Tensor]:
        """Return the complete noise schedule as a dictionary.
        
        Contains all precomputed tensors for external use:
        - betas: β_t schedule
        - alphas: α_t = 1 - β_t
        - alphas_cumprod: ᾱ_t = ∏ α_i
        - alphas_cumprod_prev: ᾱ_{t-1}
        - sqrt_alphas_cumprod: √ᾱ_t
        - sqrt_one_minus_alphas_cumprod: √(1-ᾱ_t)
        - sqrt_recip_alphas_cumprod: 1/√ᾱ_t
        - posterior_variance: σ_t²
        - sigma_hat_t: σ̂_t
        
        Returns:
            Dictionary containing all noise schedule tensors
        """
        return {
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
            'alphas_cumprod_prev': self.alphas_cumprod_prev,
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
            'sqrt_recip_alphas_cumprod': self.sqrt_recip_alphas_cumprod,
            'posterior_variance': self.posterior_variance,
            'sigma_hat_t': self.sigma_hat_t
        }
    
    def denoise_step(
        self,
        x_t: Tensor,
        predicted_noise: Tensor,
        t: Tensor
    ) -> Tensor:
        """Single reverse diffusion step using DDPM sampling.
        
        Uses the DDPM reverse process (Equation 7 in Ho et al., 2020):
            x_{t-1} = √ᾱ_{t-1}((x_t - √(1-ᾱ_t)ε_θ)/ᾱ_t) + √(1-α_{t-1})ε_θ
        
        With η=0 (deterministic, no additional noise):
            x_{t-1} = √ᾱ_{t-1}(x_t/√ᾱ_t - √(t/ᾱ_{t-1})ε_θ) + √(1-α_{t-1})ε_θ
        
        This simplifies to:
            x_{t-1} = (x_t - √(1-ᾱ_t) * predicted_noise) / √α_t * √ᾱ_{t-1} + √(1-α_{t-1}) * predicted_noise
        
        Args:
            x_t: Noisy image at timestep t [B, C, H, W]
            predicted_noise: Predicted noise ε_θ(x_t, t) [B, C, H, W]
            t: Current timestep indices [B] (scalar values in [1, T])
        
        Returns:
            x_{t-1}: Denoised image at timestep t-1 [B, C, H, W]
        
        Example:
            >>> utils = DiffusionUtils(timesteps=1000)
            >>> x_t = torch.randn(4, 3, 32, 32)
            >>> predicted_noise = torch.randn(4, 3, 32, 32)
            >>> t = torch.tensor([500, 499, 501, 498])
            >>> x_prev = utils.denooise_step(x_t, predicted_noise, t)
        """
        # Get ᾱ_t (at index t)
        alphas_cumprod_t = self.get_index_from_list(
            self.alphas_cumprod, t, x_t.shape
        )
        
        # Get ᾱ_{t-1} (at index t-1)
        # Note: t is in [1, T], so we use t-1 to get ᾱ_{t-1}
        # When t=1, t-1=0 gives ᾱ_0=1 (we prepended 1 to alphas_cumprod_prev)
        alphas_cumprod_prev_t = self.get_index_from_list(
            self.alphas_cumprod_prev, t, x_t.shape
        )
        
        # Get α_t (at index t)
        alpha_t = self.get_index_from_list(
            self.alphas, t, x_t.shape
        )
        
        # Get 1 - α_{t-1} for the coefficient of predicted_noise
        one_minus_alpha_prev_t = 1.0 - self.get_index_from_list(
            self.alphas, t - 1 if t.min() > 0 else t, x_t.shape
        )
        
        # For t=1 (t-1=0), we need special handling since α_0 = 1
        # Handle this by computing 1 - α_{t-1} directly
        # When t=1, we want 1 - α_0 = 1 - 1 = 0, but for numerical stability
        # we use the formula: √(1-α_{t-1}) = √(1-α_0) = 0
        # However, we need to handle the case where t is 0 or 1 properly
        
        # Compute √α_t
        sqrt_alpha_t = torch.sqrt(alpha_t)
        
        # Compute 1/ᾱ_t for denominator
        sqrt_recip_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        
        # Compute √(1-ᾱ_t)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        # Compute √(1-α_{t-1}) - need to handle t=1 case where we use α_0=1
        # For t=1: 1 - α_0 = 0, so √(1-α_0) = 0
        # For t>1: use α_{t-1}
        # Create mask for t=1 case
        t_for_alpha = t.clone()
        t_for_alpha = torch.clamp(t_for_alpha, min=1)  # Ensure t >= 1 for indexing
        sqrt_one_minus_alpha_prev = torch.sqrt(
            1.0 - self.get_index_from_list(self.alphas, t_for_alpha - 1, x_t.shape)
        )
        # For t=1 (where original t was 0 or 1), set to 0
        sqrt_one_minus_alpha_prev = sqrt_one_minus_alpha_prev * (t > 1).float().view(-1, 1, 1, 1).to(x_t.device)
        
        # Compute the mean (predicted denoised image)
        # model_mean = √ᾱ_{t-1} * (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        #             = √ᾱ_{t-1} * (x_t/√ᾱ_t - √(1-ᾱ_t)/√ᾱ_t * ε_θ)
        #             = √ᾱ_{t-1} * (x_t/√ᾱ_t - √(t/ᾱ_{t-1}) * ε_θ)
        
        # Compute x_t / √ᾱ_t
        pred_original_sample = sqrt_recip_alphas_cumprod_t * x_t
        
        # Compute √(1-ᾱ_t)/√ᾱ_t * ε_θ = √(1-ᾱ_t)/√ᾱ_t * ε_θ
        pred_original_sample = pred_original_sample - (
            sqrt_one_minus_alphas_cumprod_t * sqrt_recip_alphas_cumprod_t * predicted_noise
        )
        
        # Multiply by √ᾱ_{t-1}
        pred_original_sample = pred_original_sample * torch.sqrt(alphas_cumprod_prev_t)
        
        # Add √(1-α_{t-1}) * ε_θ
        x_prev = pred_original_sample + sqrt_one_minus_alpha_prev * predicted_noise
        
        return x_prev
    
    def sample_timesteps(self, batch_size: int) -> Tensor:
        """Sample random timesteps from Uniform({1, ..., T}).
        
        Used in training to sample random timesteps for each batch.
        Note: We sample from {1, ..., T} rather than {0, ..., T-1}
        to match the paper's notation where timesteps start at 1.
        
        Args:
            batch_size: Number of timesteps to sample
        
        Returns:
            Tensor of shape [batch_size] with values in [1, T]
        
        Example:
            >>> utils = DiffusionUtils(timesteps=1000)
            >>> t = utils.sample_timesteps(batch_size=32)
            >>> print(t.shape)  # torch.Size([32])
            >>> print(t.min(), t.max())  # tensor(1) tensor(1000)
        """
        # Sample from Uniform({1, ..., T})
        # torch.randint gives [0, T), so we add 1 to get [1, T]
        return torch.randint(1, self.timesteps + 1, (batch_size,))
    
    def get_sigma_hat(self, t: Tensor) -> Tensor:
        """Compute σ̂_t = (1-ᾱ_{t-1})√(α_t/(1-ᾱ_t)) for similarity-guided loss.
        
        This is used as scaling factor for classifier gradient in Equation 5:
            ||ε - ε_θ(x_t,t) - σ̂_t² γ ∇_{x_t} log p_φ(y=T|x_t)||²
        
        The σ̂_t² term scales the classifier gradient appropriately at each timestep.
        
        Args:
            t: Timestep indices [B]
        
        Returns:
            σ̂_t tensor [B, 1, 1, 1] for broadcasting with input tensor
        
        Example:
            >>> utils = DiffusionUtils(timesteps=1000)
            >>> t = torch.tensor([100, 200, 300])
            >>> sigma_hat = utils.get_sigma_hat(t)
            >>> print(sigma_hat.shape)  # torch.Size([3, 1, 1, 1])
        """
        # Index into sigma_hat_t at timestep t
        sigma_hat = self.get_index_from_list(
            self.sigma_hat_t, t, (t.shape[0], 1, 1, 1)
        )
        
        return sigma_hat
    
    def get_index_from_list(
        self,
        list: Tensor,
        t: Tensor,
        x_shape: Tuple
    ) -> Tensor:
        """Helper to index into a precomputed list at timestep t, with proper broadcasting.
        
        Takes a precomputed tensor (like sqrt_alphas_cumprod) of shape [T] and
        indices t of shape [B], and returns the values at those indices reshaped
        for broadcasting with input tensor of shape x_shape.
        
        Input: list [T], t [B], x_shape (4D)
        Output: list[t] reshaped to [B, 1, 1, 1] for broadcasting
        
        Args:
            list: Precomputed tensor of shape [T] (e.g., sqrt_alphas_cumprod)
            t: Timestep indices of shape [B]
            x_shape: Shape of input tensor for broadcasting (typically [B, C, H, W])
        
        Returns:
            Indexed tensor reshaped to [B, 1, 1, 1] for broadcasting
        
        Example:
            >>> utils = DiffusionUtils(timesteps=1000)
            >>> sqrt_alphas = utils.sqrt_alphas_cumprod  # [1000]
            >>> t = torch.tensor([100, 200, 300, 400])   # [4]
            >>> result = utils.get_index_from_list(sqrt_alphas, t, (4, 3, 32, 32))
            >>> print(result.shape)  # torch.Size([4, 1, 1, 1])
        """
        # Get batch size from t
        batch_size = t.shape[0]
        
        # Index into list at timestep t
        # t is [B], list is [T], result is [B]
        out = list.to(t.device).gather(0, t.long())
        
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        # Handle different input shapes
        if len(x_shape) == 4:
            return out.view(batch_size, 1, 1, 1)
        elif len(x_shape) == 2:
            return out.view(batch_size, 1)
        elif len(x_shape) == 1:
            return out
        else:
            # Generic case
            return out.view(batch_size, *([1] * (len(x_shape) - 1)))