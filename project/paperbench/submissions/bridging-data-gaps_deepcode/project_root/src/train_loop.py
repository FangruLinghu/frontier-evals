"""
Minimal training loop launcher for DPMs-ANT demonstration.
This script wires together a toy backbone (theta), Houlsby-style adaptors (psi),
and the ANT trainer to execute a single training step on a small batch.

Note:
- This is a lightweight, self-contained example intended for validation of wiring
  with existing components in this repository.
- It is not a full-scale training script and intentionally uses dummy components
  for the backbone and gradient signals for demonstration purposes.
"""
from __future__ import annotations

import os
import warnings
import math
import torch
from typing import Optional, Callable

# Try multiple import paths to accommodate different project layouts
ANTTrainer = None  # type: Optional[type]
try:
    from training.ant_trainer import ANTTrainer  # preferred
except Exception:
    try:
        from src.training.ant_trainer import ANTTrainer  # fallback
    except Exception:
        ANTTrainer = None

# Adaptor factory (Houlsby-style adaptors)
build_adaptors_for_backbone = None
try:
    from adaptor.adaptor import build_adaptors_for_backbone  # preferred
except Exception:
    try:
        from src.adaptor.adaptor import build_adaptors_for_backbone  # fallback
    except Exception:
        build_adaptors_for_backbone = None


class DummyBackbone(torch.nn.Module):
    """A tiny pass-through backbone that mimics a frozen θ."""

    def __init__(self):
        super().__init__()
        # simple parameter to pretend there are some weights; we won't train them
        self.param = torch.nn.Parameter(torch.zeros(1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, t):
        # Identity-like behavior; real backbones produce ε_theta(x_t, t)
        return x


def _safe_imports():
    # Helper to ensure ANTTrainer and adaptor factory are available
    global ANTTrainer, build_adaptors_for_backbone
    if ANTTrainer is None:
        raise RuntimeError("ANTTrainer could not be imported. Ensure training.ant_trainer is available.")
    if build_adaptors_for_backbone is None:
        raise RuntimeError("build_adaptors_for_backbone is not available to construct adaptors.")


def main(seed: int = 42):
    """Run a tiny one-step ANT training demonstration."""
    _safe_imports()
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # Dummy backbone θ
    theta = DummyBackbone().to(device)

    # Build a small adaptor module for a 3-layer backbone as a demonstration
    in_channels_list = [3, 64, 128]
    try:
        adaptor = build_adaptors_for_backbone("ddpm", in_channels_list=in_channels_list, activation=torch.nn.ReLU).to(device)
    except Exception:
        # If adaptor builder is not available, create a tiny single-layer adaptor as fallback
        adaptor = torch.nn.Linear(3, 3).to(device)
        adaptor.weight.data.zero_()
        adaptor.bias.data.zero_()

    # Dummy diffusion schedule tensors
    T = 10
    alphas_cumprod = torch.linspace(0.1, 0.9, steps=T)
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    # Simple zero-gradient signals for eps_theta and grad_logp
    def eps_theta_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
        return torch.zeros_like(x_t)

    def grad_logp_fn(x_t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x_t)

    # Instantiate the trainer (with a tiny learning rate for demonstration)
    try:
        trainer = ANTTrainer(
            adaptor=adaptor,
            theta=theta,
            eps_theta_fn=eps_theta_fn,
            grad_logp_fn=grad_logp_fn,
            alphas_cumprod=alphas_cumprod,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            T=T,
            gamma=5.0,
            omega=0.02,
            J=10,
            lr_adaptor=5e-4,
            device=device,
        )
    except Exception:
        raise RuntimeError("Failed to instantiate ANTTrainer. Ensure all required components are available.")

    trainer.to(device)

    # Create a tiny target x0 batch: shape [B, C, H, W]
    B, C, H, W = 4, 3, 64, 64
    x0_target = torch.randn(B, C, H, W, device=device)

    # Single training step
    t = 5
    loss = trainer.train_step(x0_target, t=t)
    print(f"[train_loop] Single ANT training step completed. Loss: {loss.item():.6f}")


if __name__ == "__main__":
    # Optional seed control via env var for reproducibility in experiments
    seed_env = os.environ.get("ANT_SEED")
    seed = int(seed_env) if seed_env is not None else 42
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(seed=seed)
