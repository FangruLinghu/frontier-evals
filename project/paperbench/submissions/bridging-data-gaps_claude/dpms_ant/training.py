"""
DPMs-ANT Training Loop.

Implements Algorithm 1 from the paper:
    1. Sample x0 from target domain q(x0)
    2. Sample timestep t ~ Uniform({1,...,T})
    3. Sample initial noise ε ~ N(0,I)
    4. For J steps: update ε via adversarial noise selection (Eq 7)
    5. Compute L(ψ) with ε* via Equation (8)
    6. Update adaptor parameters: ψ = ψ - η * ∇_ψ L(ψ)

Key hyperparameters (from paper):
    - γ = 5 (similarity guidance scale)
    - J = 10 (adversarial noise steps)
    - ω = 0.02 (adversarial noise learning rate)
    - Learning rate: 5e-5 for DDPM, 1e-5 for LDM
    - ~300 iterations
    - Batch size: 40
    - Adaptor: c=4, d=8 for DDPM; c=2, d=8 for LDM
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm
import json

from dpms_ant.diffusion.gaussian_diffusion import GaussianDiffusion
from dpms_ant.diffusion.adversarial_noise import adversarial_noise_selection, normalize_noise
from dpms_ant.adaptor.adaptor import UNetWithAdaptor


class ANTTrainer:
    """
    Trainer for DPMs-ANT (Algorithm 1).

    Handles the full training loop including:
    - Adversarial noise selection
    - Similarity-guided training
    - Adaptor-only optimization

    Args:
        model: UNetWithAdaptor (pre-trained UNet + adaptor layers)
        diffusion: GaussianDiffusion instance
        classifier: Binary classifier for similarity guidance (optional)
        lr: Learning rate (5e-5 for DDPM)
        gamma: Similarity guidance strength (5.0)
        J: Adversarial noise steps (10)
        omega: Adversarial noise step size (0.02)
        device: Training device
    """

    def __init__(
        self,
        model: UNetWithAdaptor,
        diffusion: GaussianDiffusion,
        classifier: Optional[nn.Module] = None,
        base_model: Optional[nn.Module] = None,
        lr: float = 5e-5,
        gamma: float = 5.0,
        J: int = 10,
        omega: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.classifier = classifier
        self.base_model = base_model  # Frozen base model for adversarial noise
        self.gamma = gamma
        self.J = J
        self.omega = omega
        self.device = device

        # Optimizer only updates adaptor parameters
        self.optimizer = torch.optim.Adam(
            self.model.get_adaptor_parameters(), lr=lr
        )

        # If no separate base model, use the frozen UNet inside the adaptor model
        if self.base_model is None:
            self.base_model = self.model.unet

        # Move classifier to device if provided
        if self.classifier is not None:
            self.classifier = self.classifier.to(device)
            self.classifier.eval()
            for p in self.classifier.parameters():
                p.requires_grad = False

        # Logging
        self.train_losses: List[float] = []

    def _compute_adversarial_noise(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find adversarial noise using the frozen base model (Eq 7).

        Args:
            x_start: Clean target images
            t: Timesteps

        Returns:
            Adversarial noise ε*
        """
        return adversarial_noise_selection(
            model=self.base_model,
            x_start=x_start,
            t=t,
            diffusion=self.diffusion,
            J=self.J,
            omega=self.omega,
        )

    def _compute_similarity_guidance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity guidance term: σ̂²_t * γ * ∇_xt log pϕ(y=T|xt).

        Args:
            x_t: Noisy image
            t: Timestep

        Returns:
            Guidance gradient
        """
        if self.classifier is None:
            return torch.zeros_like(x_t)

        sigma_hat = self.diffusion._extract(
            self.diffusion.sigma_hat, t, x_t.shape
        )

        classifier_grad = self.classifier.get_target_gradient(
            x_t.detach(), t, target_label=1
        )

        return (sigma_hat ** 2 * self.gamma * classifier_grad).detach()

    def train_step(self, x_start: torch.Tensor) -> float:
        """
        Single training step of Algorithm 1.

        Args:
            x_start: Clean target images, shape (B, C, H, W)

        Returns:
            Loss value
        """
        self.model.train()
        x_start = x_start.to(self.device)
        batch_size = x_start.shape[0]

        # Step 1: Sample timestep t ~ Uniform({1,...,T})
        t = torch.randint(
            0, self.diffusion.num_timesteps, (batch_size,), device=self.device
        )

        # Step 2: Adversarial noise selection (Eq 7)
        with torch.no_grad():
            adv_noise = self._compute_adversarial_noise(x_start, t)

        # Step 3: Compute noisy image with adversarial noise
        x_t = self.diffusion.q_sample(x_start, t, noise=adv_noise)

        # Step 4: Model prediction (through adaptor)
        model_output = self.model(x_t, t)

        # Handle learned variance
        if model_output.shape[1] == x_start.shape[1] * 2:
            eps_pred, _ = torch.split(model_output, x_start.shape[1], dim=1)
        else:
            eps_pred = model_output

        # Step 5: Compute target with similarity guidance (Eq 5/8)
        # From Eq 5: ||ε - ε_θ(xt,t) - σ̂²γ∇xt log pϕ(y=T|xt)||²
        # Target for ε_θ = ε - σ̂²γ∇xt log pϕ(y=T|xt)
        target = adv_noise.clone()

        if self.classifier is not None:
            guidance = self._compute_similarity_guidance(x_t, t)
            target = target - guidance

        # Step 6: Compute loss and update adaptor
        loss = F.mse_loss(eps_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        target_images: torch.Tensor,
        n_iterations: int = 300,
        batch_size: int = 40,
        log_frequency: int = 10,
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 50,
    ) -> List[float]:
        """
        Full training loop.

        Args:
            target_images: Target domain images (N, C, H, W) - typically 10 images
            n_iterations: Number of training iterations (default: 300)
            batch_size: Batch size (default: 40)
            log_frequency: How often to log (iterations)
            checkpoint_dir: Where to save checkpoints
            checkpoint_frequency: How often to checkpoint

        Returns:
            List of loss values
        """
        target_images = target_images.to(self.device)
        n_target = len(target_images)

        print(f"Starting DPMs-ANT training:")
        print(f"  Target images: {n_target}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Batch size: {batch_size}")
        print(f"  γ (guidance): {self.gamma}")
        print(f"  J (AN steps): {self.J}")
        print(f"  ω (AN lr): {self.omega}")
        print(f"  Adaptor params: {self.model.count_adaptor_parameters():,}")
        print(f"  Parameter rate: {self.model.parameter_rate():.2%}")
        print()

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        losses = []

        for iteration in tqdm(range(n_iterations), desc="Training"):
            # Sample batch from target images (with replacement)
            indices = torch.randint(0, n_target, (batch_size,))
            batch = target_images[indices]

            # Training step
            loss = self.train_step(batch)
            losses.append(loss)
            self.train_losses.append(loss)

            # Logging
            if (iteration + 1) % log_frequency == 0:
                avg_loss = sum(losses[-log_frequency:]) / log_frequency
                tqdm.write(f"  Iter {iteration+1}/{n_iterations}: Loss = {avg_loss:.6f}")

            # Checkpoint
            if checkpoint_dir and (iteration + 1) % checkpoint_frequency == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_{iteration+1}.pt")
                )

        # Final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(os.path.join(checkpoint_dir, "final.pt"))

        print(f"\nTraining complete. Final loss: {losses[-1]:.6f}")
        return losses

    def save_checkpoint(self, path: str):
        """Save training checkpoint (adaptor parameters only)."""
        checkpoint = {
            "adaptor_state_dict": {},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "config": {
                "gamma": self.gamma,
                "J": self.J,
                "omega": self.omega,
            },
        }

        # Save adaptor parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                checkpoint["adaptor_state_dict"][name] = param.data.clone()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load adaptor parameters
        model_dict = self.model.state_dict()
        for name, param in checkpoint["adaptor_state_dict"].items():
            if name in model_dict:
                model_dict[name] = param
        self.model.load_state_dict(model_dict)

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])


class BaselineTrainer:
    """
    Baseline DDPM fine-tuning trainer (without ANT).

    Used for comparison: direct fine-tuning of the full model or adaptor-only.
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        lr: float = 5e-5,
        device: torch.device = torch.device("cpu"),
        adaptor_only: bool = False,
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device

        if adaptor_only and isinstance(model, UNetWithAdaptor):
            params = model.get_adaptor_parameters()
        else:
            params = model.parameters()

        self.optimizer = torch.optim.Adam(params, lr=lr)

    def train_step(self, x_start: torch.Tensor) -> float:
        """Standard DDPM training step."""
        self.model.train()
        x_start = x_start.to(self.device)

        t = torch.randint(
            0, self.diffusion.num_timesteps, (x_start.shape[0],), device=self.device
        )
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        model_output = self.model(x_t, t)
        if model_output.shape[1] == x_start.shape[1] * 2:
            eps_pred, _ = torch.split(model_output, x_start.shape[1], dim=1)
        else:
            eps_pred = model_output

        loss = F.mse_loss(eps_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        target_images: torch.Tensor,
        n_iterations: int = 5000,
        batch_size: int = 40,
    ) -> List[float]:
        """Training loop."""
        target_images = target_images.to(self.device)
        n_target = len(target_images)
        losses = []

        for iteration in tqdm(range(n_iterations), desc="Baseline Training"):
            indices = torch.randint(0, n_target, (batch_size,))
            batch = target_images[indices]
            loss = self.train_step(batch)
            losses.append(loss)

        return losses
