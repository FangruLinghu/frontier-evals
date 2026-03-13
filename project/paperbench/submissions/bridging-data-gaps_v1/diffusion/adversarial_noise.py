Now let me create the `diffusion/adversarial_noise.py` file based on the specifications:

```python
## diffusion/adversarial_noise.py
"""Adversarial noise selection via multi-step gradient ascent (Equation 7).

This module implements the adversarial noise selection algorithm for similarity-guided
diffusion model training. The algorithm finds the worst-case adversarial noise ε*
that maximizes the training loss, which is then used in the full loss (Equation 8).

The gradient ascent process follows Equation 7:
    ε^{j+1} = Norm(ε^j + ω ∇_{ε^j}||ε^j - ε_θ(√α_t x_0 + √(1-α_t)ε^j, t)||²)

Where:
- ω: Learning rate for gradient ascent (default: 0.02)
- J: Number of gradient ascent iterations (default: 10)
- Norm(·): Normalization to ensure mean=0 and std=1
- ε_θ: Noise prediction from diffusion model with adaptor layers

This adversarial noise selection implements the min-max training from Algorithm 1:
1. Sample initial noise ε_0 ~ N(0, I)
2. For j = 0 to J-1:
   a. Compute x_t^j = √α_t x_0 + √(1-α_t) ε^j
   b. Predict noise ε_θ(x_t^j, t)
   c. Compute loss L_j = ||ε^j - ε_θ||²
   d. Compute gradient ∇_{ε^j} L_j via autograd
   e. Update ε^{j+1} = ε^j + ω * gradient
   f. Normalize to ensure mean=0, std=1
3. Return ε* = ε^J

Classes:
    AdversarialNoise: Adversarial noise selector using gradient ascent
"""

import torch
import torch.nn as nn
from torch import Tensor

from diffusion.utils import DiffusionUtils


class AdversarialNoise:
    """Adversarial noise selector via multi-step gradient ascent.
    
    This class implements the adversarial noise selection algorithm from Equation 7
    of the paper. It finds the worst-case noise ε* that maximizes the similarity-guided
    loss, which is then used in the full training loss (Equation 8).
    
    The algorithm performs gradient ascent on the noise ε to find the perturbation
    that causes the largest deviation between predicted and true noise, which
    represents the most challenging case for the diffusion model to handle.
    
    Attributes:
        omega: Learning rate ω for gradient ascent in Equation 7 (default: 0.02)
        J: Number of gradient ascent iterations (default: 10)
        device: Device for computation (cuda or cpu)
    
    Example:
        >>> from diffusion.utils import DiffusionUtils
        >>> from model.diffusion_model import DiffusionUNet
        >>> 
        >>> # Initialize adversarial noise selector
        >>> adversarial_noise = AdversarialNoise(omega=0.02, J=10)
        >>> 
        >>> # Select adversarial noise for a batch
        >>> x_0 = torch.randn(4, 3, 32, 32)
        >>> t = torch.randint(1, 1000, (4,))
        >>> model = DiffusionUNet(config)
        >>> diffusion_utils = DiffusionUtils(timesteps=1000)
        >>> 
        >>> epsilon_star = adversarial_noise.select_adversarial_noise(
        ...     x_0=x_0,
        ...     t=t,
        ...     model=model,
        ...     diffusion_utils=diffusion_utils
        ... )
        >>> print(epsilon_star.shape)  # torch.Size([4, 3, 32, 32])
    """
    
    def __init__(
        self,
        omega: float = 0.02,
        J: int = 10,
        device: torch.device = None
    ) -> None:
        """Initialize adversarial noise selector.
        
        Args:
            omega: Learning rate ω for gradient ascent step in Equation 7.
                  Controls the step size when finding worst-case noise ε*.
                  Recommended value from paper Table 6: 0.02
            J: Number of gradient ascent iterations for adversarial noise selection.
               Iterates j = 0, 1, ..., J-1 to find ε*.
               Recommended value from paper: 10
            device: Device for computation. If None, defaults to cuda if available.
        
        Example:
            >>> # Create with default settings (omega=0.02, J=10)
            >>> selector = AdversarialNoise()
            >>>
            >>> # Create with custom settings
            >>> selector = AdversarialNoise(omega=0.05, J=20, device=torch.device('cuda'))
        """
        self.omega = omega
        self.J = J
        
        # Set device: use cuda if available, otherwise cpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
    
    def select_adversarial_noise(
        self,
        x_0: Tensor,
        t: Tensor,
        model: nn.Module,
        diffusion_utils: DiffusionUtils
    ) -> Tensor:
        """Select adversarial noise ε* via multi-step gradient ascent (Equation 7).
        
        This method implements the min-max training from Equation 7. It finds the
        worst-case adversarial noise that maximizes the loss, which is then used
        in the full loss (Equation 8) for training.
        
        Algorithm:
        1. Sample initial noise ε_0 ~ N(0, I)
        2. For j = 0 to J-1:
           a. Compute x_t^j = √α_t x_0 + √(1-α_t) ε^j
           b. Predict noise ε_θ(x_t^j, t)
           c. Compute loss L_j = ||ε^j - ε_θ||²
           d. Compute gradient ∇_{ε^j} L_j via autograd
           e. Update ε^{j+1} = ε^j + ω * gradient
           f. Normalize to ensure mean=0, std=1 (Norm function)
        3. Return ε* = ε^J
        
        Args:
            x_0: Clean image tensor [B, C, H, W]
            t: Timestep indices [B] (values in [1, T])
            model: Diffusion model with adaptor layers (DiffusionUNet)
            diffusion_utils: Diffusion utilities for forward process
        
        Returns:
            Adversarial noise ε* [B, C, H, W] after J iterations of gradient ascent
        
        Example:
            >>> adversarial_noise = AdversarialNoise(omega=0.02, J=10)
            >>> x_0 = torch.randn(4, 3, 32, 32)
            >>> t = torch.tensor([100, 200, 300, 400])
            >>> epsilon_star = adversarial_noise.select_adversarial_noise(
            ...     x_0=x_0,
            ...     t=t,
            ...     model=model,
            ...     diffusion_utils=diffusion_utils
            ... )
            >>> print(epsilon_star.shape)  # torch.Size([4, 3, 32, 32])
        """
        # Get batch size and shape from input
        batch_size = x_0.size(0)
        
        # Step 1: Sample initial noise ε_0 ~ N(0, I)
        # Shape: [B, C, H, W] matching x_0
        epsilon = torch.randn_like(x_0)
        
        # Move to device
        epsilon = epsilon.to(self.device)
        x_0 = x_0.to(self.device)
        t = t.to(self.device)
        
        # Ensure model is on device
        model = model.to(self.device)
        
        # Set model to eval mode for inference during gradient computation
        # but we need gradients with respect to epsilon
        model.eval()
        
        # Step 2: Gradient ascent for J iterations
        for j in range(self.J):
            # (a) Compute x_t^j = √α_t x_0 + √(1-α_t) ε^j
            # Using diffusion_utils.add_noise with the current noise
            x_t = diffusion_utils.add_noise(x_0, t, epsilon)
            
            # (b) Predict noise ε_θ(x_t^j, t)
            # We need to enable gradients for this computation
            # Clone epsilon and set requires_grad to track gradients
            epsilon_for_grad = epsilon.clone().detach().requires_grad_(True)
            
            # Recompute x_t with the grad-enabled epsilon
            x_t_for_grad = diffusion_utils.add_noise(x_0, t, epsilon_for_grad)
            
            # Get model prediction
            epsilon_pred = model(x_t_for_grad, t)
            
            # (c) Compute loss L_j = ||ε^j - ε_θ||²
            loss = torch.mean((epsilon_for_grad - epsilon_pred) ** 2)
            
            # (d) Compute gradient ∇_{ε^j} L_j via autograd
            loss.backward()
            
            # Get gradient
            gradient = epsilon_for_grad.grad
            
            # Handle case where gradient is None
            if gradient is None:
                raise RuntimeError(
                    f"Gradient computation failed at iteration {j}. "
                    "Ensure epsilon requires grad and loss is connected to computation graph."
                )
            
            # (e) Update ε^{j+1} = ε^j + ω * gradient
            epsilon = epsilon + self.omega * gradient
            
            # (f) Normalize to ensure mean=0, std=1 (Norm function)
            epsilon = self.normalize_noise(epsilon)
        
        # Step 3: Return ε* = ε^J
        return epsilon
    
    def normalize_noise(self, epsilon: Tensor) -> Tensor:
        """Normalize noise to have mean=0 and std=1.
        
        This is the Norm(·) function in Equation 7. It ensures that the noise
        maintains a standard Gaussian distribution after each gradient update,
        which stabilizes the adversarial noise selection process.
        
        Args:
            epsilon: Input noise tensor [B, C, H, W]
        
        Returns:
            Normalized noise tensor [B, C, H, W] with mean=0 and std=1
        
        Example:
            >>> noise = torch.randn(4, 3, 32, 32)
            >>> normalized = adversarial_noise.normalize_noise(noise)
            >>> # Verify normalization
            >>> mean = normalized.mean().item()
            >>> std = normalized.std().item()
            >>> print(f"Mean: {mean:.4f}, Std: {std:.4f}")  # Close to 0 and 1
        """
        # Compute mean and std per sample (across all dimensions)
        # Keep dimensions for broadcasting: [B, C, H, W] -> [B, 1, 1, 1]
        mean = epsilon.mean(dim=(1, 2, 3), keepdim=True)
        std = epsilon.std(dim=(1, 2, 3), keepdim=True)
        
        # Normalize: (epsilon - mean) / std
        # Add small epsilon to std to avoid division by zero
        epsilon_normalized = (epsilon - mean) / (std + 1e-8)
        
        return epsilon_normalized
    
    def compute_noise_gradient(
        self,
        epsilon: Tensor,
        x_0: Tensor,
        t: Tensor,
        model: nn.Module,
        diffusion_utils: DiffusionUtils
    ) -> Tensor:
        """Compute gradient of loss with respect to noise ε.
        
        Computes ∇_{ε}||ε - ε_θ(√α_t x_0 + √(1-α_t)ε, t)||² for use in the
        adversarial noise selection algorithm.
        
        The gradient indicates how the noise should be perturbed to increase
        the loss (i.e., to find the worst-case adversarial noise).
        
        Args:
            epsilon: Current noise tensor [B, C, H, W]
            x_0: Clean image tensor [B, C, H, W]
            t: Timestep indices [B] (values in [1, T])
            model: Diffusion model with adaptor layers (DiffusionUNet)
            diffusion_utils: Diffusion utilities for forward process
        
        Returns:
            Gradient tensor ∇_{ε} L with same shape as epsilon [B, C, H, W]
        
        Example:
            >>> epsilon = torch.randn(4, 3, 32, 32)
            >>> gradient = adversarial_noise.compute_noise_gradient(
            ...     epsilon=epsilon,
            ...     x_0=x_0,
            ...     t=t,
            ...     model=model,
            ...     diffusion_utils=diffusion_utils
            ... )
            >>> print(gradient.shape)  # torch.Size([4, 3, 32, 32])
        """
        # Ensure all tensors are on the correct device
        epsilon = epsilon.to(self.device)
        x_0 = x_0.to(self.device)
        t = t.to(self.device)
        model = model.to(self.device)
        
        # Clone epsilon and enable gradient tracking
        epsilon_grad = epsilon.clone().detach().requires_grad_(True)
        
        # Compute x_t = √α_t x_0 + √(1-α_t) ε
        x_t = diffusion_utils.add_noise(x_0, t, epsilon_grad)
        
        # Get model prediction ε_θ(x_t, t)
        epsilon_pred = model(x_t, t)
        
        # Compute loss L = ||ε - ε_θ||²
        loss = torch.mean((epsilon_grad - epsilon_pred) ** 2)
        
        # Backward to compute gradient w.r.t. epsilon
        loss.backward()
        
        # Get gradient
        gradient = epsilon_grad.grad
        
        # Handle case where gradient is None
        if gradient is None:
            raise RuntimeError(
                "Gradient computation failed. Ensure epsilon requires grad "
                "and loss is connected to computation graph."
            )
        
        return gradient
    
    def select_adversarial_noise_simple(
        self,
        x_0: Tensor,
        t: Tensor,
        model: nn.Module,
        diffusion_utils: DiffusionUtils
    ) -> Tensor:
        """Simplified adversarial noise selection using compute_noise_gradient.
        
        This is an alternative implementation that uses the compute_noise_gradient
        method for clarity. The result should be identical to select_adversarial_noise.
        
        Args:
            x_0: Clean image tensor [B, C, H, W]
            t: Timestep indices [B]
            model: Diffusion model with adaptor layers
            diffusion_utils: Diffusion utilities
        
        Returns:
            Adversarial noise ε* [B, C, H, W]
        """
        # Sample initial noise ε_0 ~ N(0, I)
        epsilon = torch.randn_like(x_0).to(self.device)
        
        # Gradient ascent for J iterations
        for j in range(self.J):
            # Compute gradient
            gradient = self.compute_noise_gradient(
                epsilon=epsilon,
                x_0=x_0,
                t=t,
                model=model,
                diffusion_utils=diffusion_utils
            )
            
            # Update: ε^{j+1} = ε^j + ω * gradient
            epsilon = epsilon + self.omega * gradient
            
            # Normalize
            epsilon = self.normalize_noise(epsilon)
        
        return epsilon
    
    def get_config(self) -> dict:
        """Get current configuration of adversarial noise selector.
        
        Returns:
            Dictionary containing omega and J values
        """
        return {
            'omega': self.omega,
            'J': self.J,
            'device': str(self.device)
        }
    
    def set_omega(self, omega: float) -> None:
        """Update the learning rate omega.
        
        Args:
            omega: New learning rate value
        """
        self.omega = omega
    
    def set_J(self, J: int) -> None:
        """Update the number of iterations J.
        
        Args:
            J: New number of iterations
        """
        self.J = J