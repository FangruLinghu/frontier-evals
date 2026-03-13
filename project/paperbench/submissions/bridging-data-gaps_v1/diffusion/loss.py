## diffusion/loss.py

```python
"""Similarity-guided DDPM loss for diffusion model adaptation.

This module implements the similarity-guided loss functions described in the paper:
- Equation 5: Similarity-guided DDPM loss with classifier gradient
- Equation 8: Full loss with adversarial noise selection

The key idea is to guide the diffusion model toward target domain similarity by
adding a classifier gradient term to the noise prediction loss:
    L = ||ε - ε_θ(x_t,t) - σ̂_t² γ ∇_{x_t} log p_φ(y=T|x_t)||²

where:
- ε: true noise
- ε_θ(x_t,t): predicted noise from diffusion model
- σ̂_t: standard deviation estimate from diffusion schedule
- γ: similarity guidance strength hyperparameter
- ∇_{x_t} log p_φ(y=T|x_t): classifier gradient for target class

Classes:
    SimilarityLoss: Main loss module implementing Equations 5 and 8
"""

import torch
import torch.nn as nn
from torch import Tensor

from diffusion.utils import DiffusionUtils


class SimilarityLoss(nn.Module):
    """Similarity-guided DDPM loss for diffusion model adaptation.
    
    This module implements the similarity-guided loss functions from the paper:
    - Equation 5: L = ||ε_t - ε_θ(x_t,t) - σ̂_t² γ ∇_{x_t} log p_φ(y=T|x_t)||²
    - Equation 8: Full loss with adversarial noise ε*
    
    The classifier gradient term guides the diffusion model to generate samples
    that match the target domain distribution, while the σ̂_t² scaling ensures
    proper weighting at different timesteps.
    
    Attributes:
        gamma: Hyperparameter γ controlling similarity guidance strength (default: 5.0)
        diffusion_utils: Diffusion utilities providing σ̂_t computation
        mse_loss: MSE loss for comparing noise predictions
    
    Example:
        >>> from diffusion.utils import DiffusionUtils
        >>> from classifier.binary_classifier import BinaryClassifier
        >>> 
        >>> diffusion_utils = DiffusionUtils(timesteps=1000)
        >>> similarity_loss = SimilarityLoss(gamma=5.0, diffusion_utils=diffusion_utils)
        >>> 
        >>> # Forward pass with classifier gradient
        >>> x_t = torch.randn(4, 3, 32, 32)
        >>> t = torch.randint(1, 1000, (4,))
        >>> epsilon_true = torch.randn(4, 3, 32, 32)
        >>> epsilon_pred = torch.randn(4, 3, 32, 32)
        >>> classifier = BinaryClassifier(input_dim=3072)
        >>> 
        >>> loss = similarity_loss(x_t, t, epsilon_true, epsilon_pred, classifier)
    """
    
    def __init__(
        self,
        gamma: float,
        diffusion_utils: DiffusionUtils
    ) -> None:
        """Initialize similarity-guided loss module.
        
        Args:
            gamma: Hyperparameter γ controlling strength of similarity guidance.
                  Higher values give more weight to target domain similarity.
                  Recommended value from paper: 5.0
            diffusion_utils: Diffusion utilities for computing σ̂_t scaling factor.
                           Provides access to noise schedule and σ̂_t values.
        
        Example:
            >>> utils = DiffusionUtils(timesteps=1000)
            >>> loss_fn = SimilarityLoss(gamma=5.0, diffusion_utils=utils)
            >>> print(f"Gamma: {loss_fn.gamma}")
        """
        super().__init__()
        
        self.gamma = gamma
        self.diffusion_utils = diffusion_utils
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def compute_classifier_gradient(
        self,
        x_t: Tensor,
        t: Tensor,
        classifier: nn.Module,
        target_class: int = 1
    ) -> Tensor:
        """Compute gradient of classifier log-probability with respect to input x_t.
        
        Computes ∇_{x_t} log p_φ(y=target|x_t) using autograd.
        This gradient guides the diffusion model toward target domain distribution.
        
        The gradient is computed by:
        1. Cloning x_t and setting requires_grad=True
        2. Forward pass through classifier to get logits
        3. Computing log probability for target class
        4. Backward pass to obtain gradient w.r.t. input
        
        Args:
            x_t: Noisy image tensor [B, C, H, W] at timestep t
            t: Timestep indices [B] (values in [1, T])
            classifier: Binary classifier p_φ to distinguish source vs target
            target_class: Target class for gradient computation (1 for target, 0 for source)
                         Default: 1 (target domain)
        
        Returns:
            Gradient tensor ∇_{x_t} log p_φ(y=target|x_t) with same shape as x_t [B, C, H, W]
        
        Example:
            >>> classifier = BinaryClassifier(input_dim=3072)
            >>> x_t = torch.randn(4, 3, 32, 32)
            >>> t = torch.tensor([100, 200, 300, 400])
            >>> gradient = similarity_loss.compute_classifier_gradient(x_t, t, classifier)
            >>> print(gradient.shape)  # torch.Size([4, 3, 32, 32])
        """
        # Clone x_t and enable gradient computation
        x_t_grad = x_t.clone().detach().requires_grad_(True)
        
        # Forward pass through classifier to get logits
        logits = classifier(x_t_grad, t)
        
        # Compute log probability for target class
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[:, target_class]
        
        # Backward pass to compute gradient w.r.t. input
        # Sum over batch dimension since we need gradient for each sample
        target_log_prob.sum().backward()
        
        # Get the gradient
        grad_x_t = x_t_grad.grad
        
        # Handle case where gradient is None
        if grad_x_t is None:
            raise RuntimeError(
                "Gradient computation failed. Ensure x_t requires grad and "
                "target_log_prob is not detached from computation graph."
            )
        
        return grad_x_t
    
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        epsilon_true: Tensor,
        epsilon_pred: Tensor,
        classifier: nn.Module
    ) -> Tensor:
        """Compute similarity-guided DDPM loss (Equation 5).
        
        Computes the loss:
            L = ||ε_true - ε_pred - σ̂_t² γ ∇_{x_t} log p_φ(y=T|x_t)||²
        
        Steps:
        1. Compute σ̂_t from diffusion_utils (σ̂_t = (1-α_{t-1})√(α_t/(1-α_t)))
        2. Compute classifier gradient ∇_{x_t} log p_φ(y=T|x_t)
        3. Compute adjusted target: ε_target = ε_true + σ̂_t² * γ * gradient
        4. Compute MSE between ε_pred and ε_target
        
        Args:
            x_t: Noisy image tensor at timestep t [B, C, H, W]
            t: Timestep indices [B] (values in [1, T])
            epsilon_true: True noise ε_t [B, C, H, W] added to x_0
            epsilon_pred: Predicted noise ε_θ(x_t, t) [B, C, H, W] from model
            classifier: Binary classifier p_φ to compute similarity gradient
        
        Returns:
            Scalar loss tensor (mean squared error)
        
        Example:
            >>> loss_fn = SimilarityLoss(gamma=5.0, diffusion_utils=utils)
            >>> x_t = torch.randn(4, 3, 32, 32)
            >>> t = torch.randint(1, 1000, (4,))
            >>> epsilon_true = torch.randn(4, 3, 32, 32)
            >>> epsilon_pred = torch.randn(4, 3, 32, 32)
            >>> classifier = BinaryClassifier(input_dim=3072)
            >>> 
            >>> loss = loss_fn(x_t, t, epsilon_true, epsilon_pred, classifier)
            >>> print(loss.item())  # Scalar loss value
        """
        # Step 1: Compute σ̂_t from diffusion_utils
        # σ̂_t has shape [B, 1, 1, 1] for broadcasting with x_t
        sigma_hat = self.diffusion_utils.get_sigma_hat(t)  # [B, 1, 1, 1]
        
        # Compute σ̂_t² for scaling (square the sigma_hat)
        sigma_hat_squared = sigma_hat ** 2  # [B, 1, 1, 1]
        
        # Step 2: Compute classifier gradient ∇_{x_t} log p_φ(y=T|x_t)
        classifier_gradient = self.compute_classifier_gradient(
            x_t=x_t,
            t=t,
            classifier=classifier,
            target_class=1  # Target class is 1
        )  # [B, C, H, W]
        
        # Step 3: Compute adjusted target noise
        # ε_target = ε_true + σ̂_t² * γ * gradient
        # Scale gradient by σ̂_t² * gamma
        scaled_gradient = sigma_hat_squared * self.gamma * classifier_gradient  # [B, C, H, W]
        
        # Add to true noise
        epsilon_target = epsilon_true + scaled_gradient  # [B, C, H, W]
        
        # Step 4: Compute MSE between predicted noise and adjusted target
        loss = self.mse_loss(epsilon_pred, epsilon_target)
        
        return loss
    
    def compute_full_loss(
        self,
        x_0: Tensor,
        t: Tensor,
        epsilon_star: Tensor,
        model: nn.Module,
        diffusion_utils: DiffusionUtils,
        classifier: nn.Module
    ) -> Tensor:
        """Compute full loss with adversarial noise (Equation 8).
        
        Computes the loss with adversarial noise ε*:
            L(ψ) = E_{t,x_0}[||ε* - ε_θ,ψ(x_t*, t) - σ̂_t² γ ∇_{x_t*} log p_φ(y=T|x_t*)||²]
        
        Steps:
        1. Compute x_t* = √ᾱ_t x_0 + √(1-ᾱ_t) ε* using forward process
        2. Predict noise ε_θ,ψ(x_t*, t) using model
        3. Compute classifier gradient at x_t*
        4. Compute similarity-guided loss with adversarial noise
        
        Args:
            x_0: Clean image tensor [B, C, H, W]
            t: Timestep indices [B] (values in [1, T])
            epsilon_star: Adversarial noise ε* [B, C, H, W] from gradient ascent
            model: Diffusion model with adaptor (predicts ε_θ,ψ)
            diffusion_utils: Diffusion utilities for forward process
            classifier: Binary classifier p_φ for similarity gradient
        
        Returns:
            Scalar loss tensor (mean squared error)
        
        Example:
            >>> loss_fn = SimilarityLoss(gamma=5.0, diffusion_utils=utils)
            >>> x_0 = torch.randn(4, 3, 32, 32)
            >>> t = torch.randint(1, 1000, (4,))
            >>> epsilon_star = torch.randn(4, 3, 32, 32)  # From adversarial selection
            >>> 
            >>> loss = loss_fn.compute_full_loss(
            ...     x_0=x_0,
            ...     t=t,
            ...     epsilon_star=epsilon_star,
            ...     model=model,
            ...     diffusion_utils=utils,
            ...     classifier=classifier
            ... )
            >>> print(loss.item())
        """
        # Step 1: Compute x_t* = √ᾱ_t x_0 + √(1-ᾱ_t) ε*
        # Using the forward process with adversarial noise ε*
        x_t_star = diffusion_utils.add_noise(x_0, t, epsilon_star)
        
        # Step 2: Predict noise ε_θ,ψ(x_t*, t) using model
        epsilon_pred = model(x_t_star, t)  # [B, C, H, W]
        
        # Step 3: Compute σ̂_t² for scaling
        sigma_hat = diffusion_utils.get_sigma_hat(t)  # [B, 1, 1, 1]
        sigma_hat_squared = sigma_hat ** 2  # [B, 1, 1, 1]
        
        # Step 4: Compute classifier gradient at x_t*
        classifier_gradient = self.compute_classifier_gradient(
            x_t=x_t_star,
            t=t,
            classifier=classifier,
            target_class=1  # Target class is 1
        )  # [B, C, H, W]
        
        # Step 5: Compute adjusted target: ε_target = ε* + σ̂_t² * γ * gradient
        scaled_gradient = sigma_hat_squared * self.gamma * classifier_gradient
        epsilon_target = epsilon_star + scaled_gradient
        
        # Step 6: Compute MSE between predicted noise and adjusted target
        loss = self.mse_loss(epsilon_pred, epsilon_target)
        
        return loss
    
    def compute_standard_ddpm_loss(
        self,
        epsilon_true: Tensor,
        epsilon_pred: Tensor
    ) -> Tensor:
        """Compute standard DDPM MSE loss (without similarity guidance).
        
        This is the baseline loss L = ||ε - ε_θ(x_t,t)||² used in standard DDPM training.
        Provided for comparison purposes.
        
        Args:
            epsilon_true: True noise ε_t [B, C, H, W]
            epsilon_pred: Predicted noise ε_θ(x_t, t) [B, C, H, W]
        
        Returns:
            Scalar MSE loss tensor
        
        Example:
            >>> loss_fn = SimilarityLoss(gamma=5.0, diffusion_utils=utils)
            >>> epsilon_true = torch.randn(4, 3, 32, 32)
            >>> epsilon_pred = torch.randn(4, 3, 32, 32)
            >>> loss = loss_fn.compute_standard_ddpm_loss(epsilon_true, epsilon_pred)
        """
        return self.mse_loss(epsilon_true, epsilon_pred)