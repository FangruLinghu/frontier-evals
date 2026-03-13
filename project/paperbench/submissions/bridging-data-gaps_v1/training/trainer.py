Looking at the file spec and previously generated code, I need to create the `training/trainer.py` file with the Trainer class implementing the training loop for similarity-guided diffusion model adaptation.

```python
## training/trainer.py
"""Training module for similarity-guided diffusion model adaptation.

This module implements the main training loop (Algorithm 1) for training the
adaptor parameters ψ in the similarity-guided diffusion model. The training
combines:
1. Adversarial noise selection via multi-step gradient ascent (Equation 7)
2. Similarity-guided DDPM loss (Equation 8)

The Trainer class orchestrates:
- Forward process with adversarial noise ε*
- Noise prediction from DiffusionUNet with adaptor layers
- Classifier gradient computation for similarity guidance
- Optimizer updates for only adaptor parameters ψ (backbone θ frozen)

Classes:
    Trainer: Main training class implementing Algorithm 1
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os

from config import Config
from model.diffusion_model import DiffusionUNet
from classifier.binary_classifier import BinaryClassifier
from diffusion.utils import DiffusionUtils
from diffusion.loss import SimilarityLoss
from diffusion.adversarial_noise import AdversarialNoise


class Trainer:
    """Trainer for similarity-guided diffusion model adaptation.
    
    This class implements the main training loop (Algorithm 1) for training
    the adaptor parameters ψ while keeping the frozen pre-trained U-Net θ unchanged.
    
    Training Algorithm (Algorithm 1 from paper):
        1. Sample x_0 from batch
        2. Sample t ~ Uniform({1,...,T})
        3. Sample ε ~ N(0,I)
        4. Compute adversarial noise ε* via multi-step gradient ascent (Equation 7)
        5. Compute x_t* = √α_t x_0 + √(1-α_t) ε* (forward process)
        6. Compute similarity-guided loss L(ψ) (Equation 8)
        7. Backpropagate and update adaptor parameters ψ
    
    Attributes:
        config: Configuration object with hyperparameters
        model: DiffusionUNet with adaptor layers for noise prediction
        classifier: BinaryClassifier p_φ for source vs target discrimination
        diffusion_utils: Diffusion utilities for noise scheduling and forward process
        similarity_loss: SimilarityLoss module for computing Equation 8
        adversarial_noise: AdversarialNoise selector for Equation 7
        train_loader: DataLoader for few-shot training
        optimizer: Optimizer for adaptor parameters ψ
        device: Device to run training on (cuda or cpu)
        iteration: Current training iteration counter
    
    Example:
        >>> from config import create_toy_config
        >>> from model.diffusion_model import DiffusionUNet
        >>> from classifier.binary_classifier import BinaryClassifier
        >>> from diffusion.utils import DiffusionUtils
        >>> from data.loader import get_few_shot_dataloader
        >>> from optimizer.optimizer import OptimizerFactory
        >>> 
        >>> # Setup
        >>> config = create_toy_config()
        >>> model = DiffusionUNet(config.to_dict())
        >>> model.freeze_backbone()
        >>> classifier = BinaryClassifier(input_dim=2)
        >>> diffusion_utils = DiffusionUtils(timesteps=1000)
        >>> source_loader, target_loader = get_few_shot_dataloader(config)
        >>> optimizer = OptimizerFactory().create_optimizer(model, config.learning_rate)
        >>> 
        >>> # Create trainer and train
        >>> trainer = Trainer(
        ...     config=config,
        ...     model=model,
        ...     classifier=classifier,
        ...     diffusion_utils=diffusion_utils,
        ...     train_loader=source_loader,
        ...     optimizer=optimizer
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config: Config,
        model: DiffusionUNet,
        classifier: BinaryClassifier,
        diffusion_utils: DiffusionUtils,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> None:
        """Initialize Trainer with all components.
        
        Step 1: Store configuration and components
        Step 2: Set device to cuda if available
        Step 3: Create SimilarityLoss and AdversarialNoise instances
        Step 4: Move model and classifier to device
        Step 5: Set iteration counter to 0
        
        Args:
            config: Configuration object with hyperparameters
                - gamma: similarity guidance strength
                - omega: learning rate for adversarial noise selection
                - J: number of gradient ascent iterations
                - timesteps: number of diffusion timesteps
            model: DiffusionUNet with adaptor layers for noise prediction
            classifier: BinaryClassifier p_φ for source vs target discrimination
            diffusion_utils: Diffusion utilities for forward/reverse process
            train_loader: DataLoader for few-shot training
            optimizer: Optimizer for adaptor parameters ψ
        """
        # Step 1: Store configuration and components
        self.config = config
        self.model = model
        self.classifier = classifier
        self.diffusion_utils = diffusion_utils
        self.train_loader = train_loader
        self.optimizer = optimizer
        
        # Step 2: Set device to cuda if available
        if torch.cuda.is_available() and config.device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Step 3: Create SimilarityLoss and AdversarialNoise instances
        self.similarity_loss = SimilarityLoss(
            gamma=config.gamma,
            diffusion_utils=diffusion_utils
        )
        
        self.adversarial_noise = AdversarialNoise(
            omega=config.omega,
            J=config.J,
            device=torch.device(self.device)
        )
        
        # Step 4: Move model and classifier to device
        self.model.to(self.device)
        self.model.freeze_backbone()  # Freeze backbone, keep adapters trainable
        
        self.classifier.to(self.device)
        
        # Step 5: Set iteration counter to 0
        self.iteration = 0
        
        # Print initialization info
        print(f"Trainer initialized on device: {self.device}")
        print(f"  - Gamma: {config.gamma}")
        print(f"  - Omega: {config.omega}")
        print(f"  - J: {config.J}")
        print(f"  - Learning rate: {config.learning_rate}")
        print(f"  - Iterations: {config.iterations}")
    
    def train_step(self, x_0: Tensor) -> Dict[str, float]:
        """Execute one training step (Algorithm 1).
        
        Performs the following steps:
        1. Sample x_0 from batch
        2. Sample t ~ Uniform({1,...,T})
        3. Sample ε ~ N(0,I)
        4. Compute adversarial noise ε* via gradient ascent (Equation 7)
        5. Compute forward process x_t* = √α_t x_0 + √(1-α_t) ε*
        6. Compute similarity-guided loss L(ψ) using Equation 8
        7. Backpropagate and update adaptor parameters ψ
        
        Args:
            x_0: Clean image tensor [B, C, H, W] for images or [B, 2] for toy data
        
        Returns:
            Dictionary containing:
                - total_loss: Full similarity-guided loss (Equation 8)
                - ddpm_loss: Standard DDPM loss ||ε* - ε_pred||²
                - similarity_term_magnitude: Magnitude of similarity guidance term
        
        Example:
            >>> trainer = Trainer(...)
            >>> x_0 = torch.randn(40, 3, 32, 32)
            >>> loss_dict = trainer.train_step(x_0)
            >>> print(loss_dict['total_loss'])
        """
        # Step 1: Move x_0 to device
        x_0 = x_0.to(self.device)
        
        # Get batch size
        batch_size = x_0.shape[0]
        
        # Step 2: Sample t ~ Uniform({1, ..., T})
        t = torch.randint(
            1, 
            self.config.timesteps, 
            (batch_size,), 
            device=self.device
        )
        
        # Step 3: Sample ε ~ N(0, I)
        epsilon = torch.randn_like(x_0)
        
        # Step 4: Compute adversarial noise ε* via gradient ascent (Equation 7)
        epsilon_star = self._compute_adversarial_noise(x_0, t, epsilon)
        
        # Step 5: Compute forward process x_t* = √α_t x_0 + √(1-α_t) ε*
        # Get alpha_bar_t (ᾱ_t) from diffusion_utils
        alpha_bar_t = self.diffusion_utils.get_index_from_list(
            self.diffusion_utils.alphas_cumprod, 
            t, 
            x_0.shape
        )
        
        # Compute sqrt(alpha_bar) and sqrt(1 - alpha_bar)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
        
        # Reshape for broadcasting
        if x_0.dim() == 4:  # Image data [B, C, H, W]
            sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)
        elif x_0.dim() == 2:  # Toy data [B, 2]
            sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1)
        
        # Compute x_t*
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon_star
        
        # Step 6: Compute similarity-guided loss L(ψ) (Equation 8)
        # Get noise prediction from model
        epsilon_pred = self.model(x_t, t)
        
        # Compute classifier gradient ∇_{x_t} log p_φ(y=T|x_t)
        # Using target_class=1 (target domain) as per resolution
        grad_log_prob = self.classifier.compute_gradient(
            x_t=x_t,
            t=t,
            target_class=1  # Target class is 1
        )
        
        # Compute sigma_hat_t from diffusion_utils
        sigma_hat_t = self.diffusion_utils.get_sigma_hat(t)
        
        # Reshape sigma_hat_t for broadcasting
        if x_0.dim() == 4:
            sigma_hat_t = sigma_hat_t.view(-1, 1, 1, 1)
        elif x_0.dim() == 2:
            sigma_hat_t = sigma_hat_t.view(-1, 1)
        
        # Compute similarity term: σ̂_t² * γ * ∇_{x_t} log p_φ(y=T|x_t)
        similarity_term = (sigma_hat_t ** 2) * self.config.gamma * grad_log_prob
        
        # Compute loss: ||ε* - ε_pred - similarity_term||²
        loss = torch.mean((epsilon_star - epsilon_pred - similarity_term) ** 2)
        
        # Also compute standard DDPM loss for logging
        ddpm_loss = torch.mean((epsilon_star - epsilon_pred) ** 2)
        
        # Compute similarity term magnitude for logging
        similarity_term_magnitude = torch.mean(torch.abs(similarity_term)).item()
        
        # Step 7: Backpropagate and update adaptor parameters ψ
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Increment iteration counter
        self.iteration += 1
        
        # Return loss dictionary
        return {
            'total_loss': loss.item(),
            'ddpm_loss': ddpm_loss.item(),
            'similarity_term_magnitude': similarity_term_magnitude
        }
    
    def _compute_adversarial_noise(
        self,
        x_0: Tensor,
        t: Tensor,
        epsilon: Tensor
    ) -> Tensor:
        """Compute adversarial noise ε* via multi-step gradient ascent (Equation 7).
        
        Starting from initial noise ε_0 ~ N(0, I), perform J iterations:
            ε^{j+1} = Norm(ε^j + ω ∇_{ε^j}||ε^j - ε_θ(√α_t x_0 + √(1-α_t)ε^j, t)||²)
        
        The Norm(·) operation ensures mean=0 and std=I.
        
        Args:
            x_0: Clean image tensor [B, C, H, W] or [B, 2]
            t: Timestep indices [B] (values in [1, T])
            epsilon: Initial noise tensor [B, C, H, W] or [B, 2]
        
        Returns:
            Adversarial noise ε* [B, C, H, W] or [B, 2] after J iterations
        
        Example:
            >>> x_0 = torch.randn(40, 3, 32, 32)
            >>> t = torch.randint(1, 1000, (40,))
            >>> epsilon = torch.randn_like(x_0)
            >>> epsilon_star = trainer._compute_adversarial_noise(x_0, t, epsilon)
            >>> print(epsilon_star.shape)  # torch.Size([40, 3, 32, 32])
        """
        # Get alpha_bar from diffusion_utils
        alpha_bar = self.diffusion_utils.get_index_from_list(
            self.diffusion_utils.alphas_cumprod,
            t,
            x_0.shape
        )
        
        # Compute x_t from epsilon: x_t = √α_bar * x_0 + √(1-α_bar) * epsilon
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        
        # Reshape for broadcasting
        if x_0.dim() == 4:
            sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)
        elif x_0.dim() == 2:
            sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1)
        
        # Initialize epsilon_j = epsilon.clone()
        epsilon_j = epsilon.clone()
        
        # Gradient ascent for J iterations
        for j in range(self.config.J):
            # Compute x_t_current = √α_bar * x_0 + √(1-α_bar) * epsilon_j
            x_t_current = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon_j
            
            # Get model prediction: epsilon_pred = ε_θ(x_t_current, t)
            # We need to compute gradient w.r.t. epsilon_j
            epsilon_j_detached = epsilon_j.clone().detach().requires_grad_(True)
            
            # Recompute x_t with gradient-enabled epsilon
            x_t_for_grad = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon_j_detached
            
            # Get model prediction
            epsilon_pred = self.model(x_t_for_grad, t)
            
            # Compute loss: ||ε_j - ε_pred||²
            loss = torch.mean((epsilon_j_detached - epsilon_pred) ** 2)
            
            # Backward to get gradient w.r.t. epsilon_j
            loss.backward()
            
            # Get gradient
            grad_eps = epsilon_j_detached.grad
            
            # Handle case where gradient is None
            if grad_eps is None:
                raise RuntimeError(
                    f"Gradient computation failed at iteration {j}. "
                    "Ensure epsilon requires grad and loss is connected to computation graph."
                )
            
            # Update epsilon: ε^{j+1} = ε^j + ω * grad_eps
            epsilon_j = epsilon_j + self.config.omega * grad_eps
            
            # Normalize: epsilon_j = Norm(epsilon_j)
            epsilon_j = self._normalize_noise(epsilon_j)
        
        # Return epsilon_J (final adversarial noise)
        return epsilon_j
    
    def _normalize_noise(self, epsilon: Tensor) -> Tensor:
        """Normalize noise to ensure zero mean and unit standard deviation.
        
        Implements the Norm(·) operation in Equation 7:
            Norm(ε) = (ε - mean(ε)) / std(ε)
        
        This ensures the noise maintains a standard Gaussian distribution
        after each gradient update.
        
        Args:
            epsilon: Input noise tensor [B, C, H, W] or [B, 2]
        
        Returns:
            Normalized noise tensor [B, C, H, W] or [B, 2] with mean=0, std=1
        
        Example:
            >>> noise = torch.randn(40, 3, 32, 32)
            >>> normalized = trainer._normalize_noise(noise)
            >>> print(normalized.mean(), normalized.std())  # Close to 0 and 1
        """
        # Compute mean and std across all dimensions except batch
        # For 4D tensor [B, C, H, W]: compute mean/std over dimensions 1, 2, 3
        # For 2D tensor [B, 2]: compute mean/std over dimension 1
        
        if epsilon.dim() == 4:
            # Image data: [B, C, H, W]
            mean = epsilon.mean(dim=(1, 2, 3), keepdim=True)
            std = epsilon.std(dim=(1, 2, 3), keepdim=True)
        elif epsilon.dim() == 2:
            # Toy data: [B, 2]
            mean = epsilon.mean(dim=1, keepdim=True)
            std = epsilon.std(dim=1, keepdim=True)
        else:
            # Generic case
            dim = tuple(range(1, epsilon.dim()))
            mean = epsilon.mean(dim=dim, keepdim=True)
            std = epsilon.std(dim=dim, keepdim=True)
        
        # Normalize: (epsilon - mean) / std
        # Add small epsilon to std to avoid division by zero
        epsilon_normalized = (epsilon - mean) / (std + 1e-8)
        
        return epsilon_normalized
    
    def _compute_similarity_loss(
        self,
        x_0: Tensor,
        t: Tensor,
        epsilon_star: Tensor
    ) -> Tensor:
        """Compute similarity-guided DDPM loss (Equation 8).
        
        This is an alternative implementation using the SimilarityLoss module.
        Currently not used in train_step but provided for clarity.
        
        Args:
            x_0: Clean image tensor [B, C, H, W] or [B, 2]
            t: Timestep indices [B]
            epsilon_star: Adversarial noise ε* [B, C, H, W] or [B, 2]
        
        Returns:
            Scalar loss tensor (mean squared error)
        """
        # Compute x_t* = √α_t x_0 + √(1-α_t) ε*
        alpha_bar = self.diffusion_utils.get_index_from_list(
            self.diffusion_utils.alphas_cumprod,
            t,
            x_0.shape
        )
        
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        
        # Reshape for broadcasting
        if x_0.dim() == 4:
            sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)
        elif x_0.dim() == 2:
            sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon_star
        
        # Get noise prediction
        epsilon_pred = self.model(x_t, t)
        
        # Compute loss using similarity_loss module
        loss = self.similarity_loss(
            x_t=x_t,
            t=t,
            epsilon_true=epsilon_star,
            epsilon_pred=epsilon_pred,
            classifier=self.classifier
        )
        
        return loss
    
    def train(self) -> None:
        """Main training loop for iterations (default 300).
        
        For each iteration:
        1. Get batch from train_loader
        2. Call train_step to perform one training iteration
        3. Log progress every 10 iterations
        
        Uses only adaptor parameters ψ for optimization (model backbone θ is frozen).
        
        Training continues for config.iterations (default 300 from paper Section 5.1).
        
        Example:
            >>> trainer = Trainer(...)
            >>> trainer.train()
            >>> # Training complete after 300 iterations
        """
        print(f"Starting training for {self.config.iterations} iterations...")
        
        for iteration in range(self.config.iterations):
            # Get batch from train_loader
            try:
                batch = next(self.train_iter)
            except (AttributeError, StopIteration):
                # Initialize iterator if not exists or reset if exhausted
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
            
            # Handle different batch formats
            if isinstance(batch, dict):
                # Batch is a dictionary with 'image' or 'data' key
                if 'image' in batch:
                    x_0 = batch['image']
                elif 'data' in batch:
                    x_0 = batch['data']
                else:
                    x_0 = batch[list(batch.keys())[0]]
            else:
                # Batch is directly the tensor
                x_0 = batch
            
            # Perform training step
            loss_dict = self.train_step(x_0)
            
            # Log progress every 10 iterations
            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}/{self.config.iterations}, "
                    f"Loss: {loss_dict['total_loss']:.4f}, "
                    f"DDPM Loss: {loss_dict['ddpm_loss']:.4f}, "
                    f"Similarity Magnitude: {loss_dict['similarity_term_magnitude']:.4f}"
                )
        
        print(f"Training complete! Total iterations: {self.iteration}")
        
        # Save final checkpoint
        checkpoint_path = os.path.join(self.config.save_dir, 'final_checkpoint.pt')
        self.save_checkpoint(checkpoint_path)
        print(f"Final checkpoint saved to {checkpoint_path}")
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint including adaptor parameters and training state.
        
        Saves:
        - Model state dict (including adaptor layers)
        - Classifier state dict
        - Optimizer state dict
        - Training iteration counter
        - Config (as dict)
        
        Args:
            path: Path to save the checkpoint
        
        Example:
            >>> trainer.save_checkpoint('./checkpoints/model_final.pt')
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': self.iteration,
            'config': self.config.to_dict()
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint from specified path.
        
        Restores:
        - Model state dict (including adaptor layers)
        - Classifier state dict
        - Optimizer state dict
        - Training iteration counter
        
        Args:
            path: Path to load the checkpoint from
        
        Example:
            >>> trainer.load_checkpoint('./checkpoints/model_final.pt')
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore classifier state
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore iteration counter
        self.iteration = checkpoint['iteration']
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from iteration {self.iteration}")
    
    def set_train_loader(self, train_loader: DataLoader) -> None:
        """Set a new training data loader.
        
        Args:
            train_loader: New DataLoader for training
        """
        self.train_loader = train_loader
        # Reset iterator
        if hasattr(self, 'train_iter'):
            delattr(self, 'train_iter')
    
    def get_iteration(self) -> int:
        """Get current training iteration.
        
        Returns:
            Current iteration counter value
        """
        return self.iteration
    
    def get_model(self) -> DiffusionUNet:
        """Get the model.
        
        Returns:
            The DiffusionUNet model
        """
        return self.model
    
    def get_classifier(self) -> BinaryClassifier:
        """Get the classifier.
        
        Returns:
            The BinaryClassifier
        """
        return self.classifier
    
    def get_config(self) -> Config:
        """Get the configuration.
        
        Returns:
            The Config object
        """
        return self.config