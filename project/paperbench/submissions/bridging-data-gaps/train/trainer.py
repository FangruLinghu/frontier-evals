## train/trainer.py
"""
DPMs-ANT Trainer for few-shot transfer learning in diffusion models.

Implements Algorithm 1 from the paper: Training DPMs with ANT.
Orchestrates the full training loop integrating:
- Pre-trained diffusion model with adaptor layers
- Binary classifier for similarity guidance
- Adversarial noise generator
- Noise scheduler
- Similarity-guided loss computation

All configurations are sourced from config.yaml to ensure consistency across the pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from pathlib import Path

# Import dependencies
from config import config
from model.unet_with_adaptor import UNetWithAdaptor
from classifier.binary_classifier import BinaryClassifier
from utils.adversarial_noise_generator import AdversarialNoiseGenerator
from utils.noise_scheduler import DDPMNoiseScheduler


class DPMsANTTrainer:
    """
    Trainer class for DPMs-ANT (Diffusion Probabilistic Models with Adversarial Noise-based Transfer Learning).
    
    Implements the complete training procedure described in Algorithm 1 of the paper.
    Only updates the small adaptor parameters while keeping the pre-trained diffusion model frozen.
    Combines adversarial noise selection and similarity-guided training for efficient few-shot transfer.
    """
    
    def __init__(self, 
                 model: UNetWithAdaptor,
                 classifier: BinaryClassifier,
                 noise_gen: AdversarialNoiseGenerator,
                 noise_scheduler: DDPMNoiseScheduler,
                 device: Optional[torch.device] = None):
        """
        Initialize the DPMs-ANT trainer with all required components.
        
        Args:
            model: UNetWithAdaptor instance containing pre-trained base U-Net and trainable adaptors
            classifier: Trained and frozen binary classifier p_phi(y|x_t)
            noise_gen: Adversarial noise generator for finding worst-case noise patterns
            noise_scheduler: DDPM noise scheduler providing alpha coefficients and diffusion functions
            device: Device to run training on. If None, uses CUDA if available
            
        Raises:
            ValueError: If any component is None or invalid
        """
        # Validate inputs
        if model is None:
            raise ValueError("Model cannot be None")
        if classifier is None:
            raise ValueError("Classifier cannot be None")
        if noise_gen is None:
            raise ValueError("Noise generator cannot be None")
        if noise_scheduler is None:
            raise ValueError("Noise scheduler cannot be None")
            
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move components to device
        self.model = model.to(self.device)
        self.classifier = classifier.to(self.device)
        self.noise_gen = noise_gen
        self.scheduler = noise_scheduler
        
        # Extract configuration values
        self.iterations = config.training.iterations
        self.batch_size = config.training.batch_size
        self.lr = config.training.lr_ddpm  # Default to DDPM; could be adjusted based on model type
        self.gamma = config.model.gamma
        self.J = config.adversarial_noise.J
        self.omega = config.adversarial_noise.omega
        
        # Ensure model is in correct state
        self.model.train()  # Enable training mode for adaptors
        self.classifier.eval()  # Classifier is frozen - evaluation mode
        
        # Setup optimizer for adaptor parameters only
        self.optimizer = optim.Adam(
            self.model.get_trainable_parameters().values(),
            lr=self.lr
        )
        
        # Setup logging
        self.checkpoint_dir = Path(config.logging.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Store ablation flags
        self.w_o_an = config.ablation.w_o_an  # without adversarial noise
        self.w_o_sg = config.ablation.w_o_sg  # without similarity guidance
        
        # Training statistics
        self.loss_history = []
        
        print(f"Initialized DPMsANTTrainer on {self.device}")
        print(f"Training iterations: {self.iterations}, Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}, Gamma: {self.gamma}")
        print(f"Adaptor parameter efficiency: {self.model.get_parameter_efficiency():.4f}")
    
    def step(self, batch: torch.Tensor) -> float:
        """
        Perform one training step on a batch of target domain images.
        
        Implements Algorithm 1 from the paper:
        1. Sample x0 ~ q(x0), t ~ Uniform(1..T)
        2. Generate adversarial noise ε* via J-step gradient ascent
        3. Compute xt* = sqrt(ᾱ_t)x0 + sqrt(1-ᾱ_t)ε*
        4. Compute classifier gradient ∇_{xt*} log p_ϕ(y=τ|xt*)
        5. Compute total loss ||ε* - ϵ_{θ,ψ}(xt*,t) - σ̂_t²γ∇log p_ϕ||^2
        6. Update adaptor parameters ψ only
        
        Args:
            batch: Batch of real target images x0 of shape (B, C, H, W)
            
        Returns:
            Scalar loss value for this step
            
        Raises:
            RuntimeError: If computation fails due to shape mismatch or device issues
        """
        # Move batch to device
        x0 = batch.to(self.device)
        B = x0.shape[0]
        
        # Sample timestep uniformly from [1, T]
        t = torch.randint(1, self.scheduler.T + 1, (B,), device=self.device)
        
        # Generate adversarial noise ε*
        if self.w_o_an:
            # Ablation: without adversarial noise - use random Gaussian noise
            eps_star = torch.randn_like(x0, device=self.device)
        else:
            # Normal operation: generate adversarial noise
            eps_init = torch.randn_like(x0, device=self.device)
            eps_star = self.noise_gen.generate(x0=x0, t=t, eps_init=eps_init)
        
        # Compute noised image xt* = sqrt(ᾱ_t)x0 + sqrt(1-ᾱ_t)ε*
        xt_star = self.scheduler.add_noise(x0=x0, t=t, noise=eps_star)
        
        # Compute classifier gradient ∇_{xt*} log p_ϕ(y=τ|xt*)
        if self.w_o_sg:
            # Ablation: without similarity guidance - zero gradient
            scaled_classifier_grad = torch.zeros_like(xt_star)
        else:
            # Normal operation: compute classifier gradient
            with torch.enable_grad():
                xt_star_for_grad = xt_star.detach().clone()
                xt_star_for_grad.requires_grad_(True)
                
                # Forward pass through classifier
                logits = self.classifier(xt_star_for_grad)
                # Compute log probability for target class: log sigmoid(logits)
                log_prob = -torch.nn.functional.softplus(-logits)
                
                # Compute gradient w.r.t. input
                grad = torch.autograd.grad(
                    outputs=log_prob.sum(),
                    inputs=xt_star_for_grad,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False
                )[0]
                
                # Scale by γ and σ̂_t^2
                sigma_hat_sq = self.scheduler.get_sigma_hat(t) ** 2  # Shape: (B,1,1,1)
                scaled_classifier_grad = sigma_hat_sq * self.gamma * grad
                
                # Detach to prevent backprop into classifier
                scaled_classifier_grad = scaled_classifier_grad.detach()
        
        # Forward pass through model to get predicted noise
        pred_noise = self.model(xt_star, t)
        
        # Compute target for loss: ε* - scaled_classifier_grad
        target = eps_star - scaled_classifier_grad
        
        # Compute MSE loss
        loss = torch.mean((pred_noise - target) ** 2)
        
        # Backward pass and parameter update
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters().values(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Store loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def train(self, dataloader: DataLoader) -> None:
        """
        Train the model using the provided data loader.
        
        Runs the complete training loop for specified number of iterations.
        Logs progress and saves checkpoints according to configuration.
        
        Args:
            dataloader: DataLoader providing batches of target domain images
        """
        print(f"Starting training for {self.iterations} iterations...")
        
        # Set model to train mode
        self.model.train()
        
        # Create iterator that cycles through dataset
        data_iter = iter(dataloader)
        
        log_frequency = config.logging.log_frequency
        save_checkpoints = config.logging.save_checkpoints
        
        for i in range(self.iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator if dataset exhausted
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Perform training step
            loss = self.step(batch)
            
            # Log progress
            if (i + 1) % log_frequency == 0:
                avg_loss = sum(self.loss_history[-log_frequency:]) / min(log_frequency, len(self.loss_history))
                print(f"Iteration [{i+1}/{self.iterations}], Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if save_checkpoints and (i + 1) % log_frequency == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{i+1}.pth"
                self.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_path = self.checkpoint_dir / "final_model.pth"
        self.save_checkpoint(final_path)
        print(f"Training completed. Final model saved to {final_path}")
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save trainer state including model weights and optimizer state.
        
        Args:
            filepath: Path to save checkpoint file
        """
        checkpoint = {
            'iteration': len(self.loss_history),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': {
                'iterations': self.iterations,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'J': self.J,
                'omega': self.omega,
                'w_o_an': self.w_o_an,
                'w_o_sg': self.w_o_sg
            }
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load trainer state from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        
        print(f"Loaded checkpoint from {filepath} at iteration {checkpoint['iteration']}")


# Example usage and testing
if __name__ == "__main__":
    try:
        # Create dummy components for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dummy model (this would normally be loaded from pre-trained weights)
        class DummyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
            
            def forward(self, x, timesteps=None):
                return self.conv(x)
        
        # Create dummy U-Net
        dummy_unet = DummyUNet()
        
        # Wrap with adaptor
        model = UNetWithAdaptor(dummy_unet, model_type='ddpm')
        
        # Create other components
        classifier = BinaryClassifier(device=device)
        noise_scheduler = DDPMNoiseScheduler().to(device)
        noise_gen = AdversarialNoiseGenerator(model.base_unet, noise_scheduler, device)
        
        # Create trainer
        trainer = DPMsANTTrainer(
            model=model,
            classifier=classifier,
            noise_gen=noise_gen,
            noise_scheduler=noise_scheduler,
            device=device
        )
        
        # Test step function with dummy data
        dummy_batch = torch.randn(2, 3, 64, 64)
        loss = trainer.step(dummy_batch)
        print(f"Test step completed with loss: {loss:.6f}")
        
        # Test train method (short run)
        from torch.utils.data import TensorDataset, DataLoader
        dummy_dataset = TensorDataset(torch.randn(10, 3, 64, 64))
        dummy_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
        
        print("\nStarting short training test...")
        trainer.train(dummy_loader)
        
    except Exception as e:
        print(f"Error during testing: {e}")
