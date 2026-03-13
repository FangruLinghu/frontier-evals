"""
Training infrastructure for Simformer.

Provides a Trainer class that handles:
- Training loop
- Validation
- Checkpointing
- Logging
- Early stopping
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from simformer.diffusion.sde import SDE, get_sde
from simformer.training.losses import SimformerLoss
from simformer.utils.helpers import EMA, get_device


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 1000
    num_epochs: int = 1000
    grad_clip: Optional[float] = 1.0

    # Scheduler
    scheduler: str = "cosine"  # "cosine", "plateau", or "none"
    warmup_epochs: int = 0

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 100
    keep_last_n: int = 3

    # Early stopping
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 1e-6

    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "simformer"

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"


class Trainer:
    """
    Trainer for Simformer models.

    Handles the complete training pipeline including:
    - Data loading
    - Training loop
    - Validation
    - Checkpointing
    - Early stopping
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        sde: SDE,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: Simformer model to train
            sde: SDE for diffusion
            config: Training configuration
            loss_fn: Optional custom loss function
        """
        self.config = config or TrainingConfig()

        # Device setup
        if self.config.device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(self.config.device)

        self.model = model.to(self.device)
        self.sde = sde

        # Loss function
        if loss_fn is None:
            self.loss_fn = SimformerLoss(
                sde=sde,
                n_params=model.n_params,
                n_data=model.n_data,
            )
        else:
            self.loss_fn = loss_fn

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        self._setup_scheduler()

        # EMA
        if self.config.use_ema:
            self.ema = EMA(self.model, decay=self.config.ema_decay)
        else:
            self.ema = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        if self.config.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, disabling wandb logging")
                self.config.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=20,
                min_lr=1e-6,
            )
        else:
            self.scheduler = None

    def create_dataloader(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a DataLoader from theta and x tensors."""
        dataset = TensorDataset(theta, x)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, (theta, x) in enumerate(pbar):
            theta = theta.to(self.device)
            x = x.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.loss_fn(self.model, theta, x)
            loss = loss_dict["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            # Update
            self.optimizer.step()

            # EMA update
            if self.ema is not None:
                self.ema.update()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": loss.item()})

            # Periodic logging
            if self.global_step % self.config.log_every == 0:
                if self.config.use_wandb:
                    self.wandb.log({
                        "train/loss": loss.item(),
                        "train/step": self.global_step,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    })

        return total_loss / num_batches

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()

        # Use EMA weights for validation if available
        if self.ema is not None:
            self.ema.apply_shadow()

        total_loss = 0.0
        num_batches = 0

        for theta, x in val_loader:
            theta = theta.to(self.device)
            x = x.to(self.device)

            loss_dict = self.loss_fn(self.model, theta, x)
            total_loss += loss_dict["loss"].item()
            num_batches += 1

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        return total_loss / num_batches

    def train(
        self,
        train_theta: torch.Tensor,
        train_x: torch.Tensor,
        val_theta: Optional[torch.Tensor] = None,
        val_x: Optional[torch.Tensor] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_theta: Training parameters
            train_x: Training data
            val_theta: Optional validation parameters
            val_x: Optional validation data
            callbacks: Optional list of callback functions

        Returns:
            Training history
        """
        # Create data loaders
        train_loader = self.create_dataloader(train_theta, train_x, shuffle=True)

        if val_theta is not None and val_x is not None:
            val_loader = self.create_dataloader(val_theta, val_x, shuffle=False)
        else:
            val_loader = None

        # Initialize wandb
        if self.config.use_wandb:
            self.wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
            )

        print(f"Training on {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(train_theta)}")
        if val_loader:
            print(f"Validation samples: {len(val_theta)}")

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
            else:
                val_loss = train_loss

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            log_str = f"Epoch {epoch}: train_loss={train_loss:.6f}"
            if val_loader is not None:
                log_str += f", val_loss={val_loss:.6f}"
            print(log_str)

            if self.config.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                }
                if val_loader is not None:
                    log_dict["val/loss"] = val_loss
                self.wandb.log(log_dict)

            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Early stopping
            if val_loss < self.best_loss - self.config.min_delta:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1

            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_loss, val_loss)

        # Save final model
        self.save_checkpoint("final_model.pt")

        if self.config.use_wandb:
            self.wandb.finish()

        return self.history

    def save_checkpoint(self, filename: str):
        """Save a checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": vars(self.config),
            "history": self.history,
        }

        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, filename: str):
        """Load a checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]

        if self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

        while len(checkpoints) > self.config.keep_last_n:
            checkpoints[0].unlink()
            checkpoints.pop(0)

    def get_model(self, use_ema: bool = True) -> nn.Module:
        """
        Get the model for inference.

        Args:
            use_ema: Whether to use EMA weights

        Returns:
            Model with appropriate weights
        """
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
            return self.model
        return self.model


def train_simformer(
    model: nn.Module,
    train_theta: torch.Tensor,
    train_x: torch.Tensor,
    val_theta: Optional[torch.Tensor] = None,
    val_x: Optional[torch.Tensor] = None,
    sde_type: str = "vesde",
    num_epochs: int = 1000,
    batch_size: int = 1000,
    learning_rate: float = 1e-3,
    device: str = "auto",
    **kwargs,
) -> Trainer:
    """
    Convenience function to train a Simformer model.

    Args:
        model: Simformer model
        train_theta: Training parameters
        train_x: Training data
        val_theta: Optional validation parameters
        val_x: Optional validation data
        sde_type: Type of SDE ("vesde" or "vpsde")
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        **kwargs: Additional arguments for TrainingConfig

    Returns:
        Trained Trainer object
    """
    sde = get_sde(sde_type)

    config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        **kwargs,
    )

    trainer = Trainer(model, sde, config)
    trainer.train(train_theta, train_x, val_theta, val_x)

    return trainer
