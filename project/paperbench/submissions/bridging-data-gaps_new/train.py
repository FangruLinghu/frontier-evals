## train.py
"""
ANT training loop for DPMs-ANT: Adversarial Noise-Based Transfer Learning for Diffusion Models.

This module defines the adaptor-only training loop that tunes a light-weight adaptor ψ
attached to a frozen diffusion backbone θ. Two core strategies from the DPMs-ANT paper
are implemented:

  1) Similarity-guided training: a fixed binary classifier φ provides a gradient signal
     that estimates domain similarity between the source and target, injected into the
     diffusion denoising objective.

  2) Adversarial noise selection (AN): an inner loop performs a finite-step gradient ascent
     to identify the "worst-case" Gaussian noise ε for the current adaptor and input, and
     this adversarial noise is used in the training objective.

The implementation below follows the API design described in the repository plan and is
compatible with the provided utilities and data module. It is designed to be self-contained
and deterministic given seeds defined in the configuration or at runtime.

Notes and design decisions:
- The adaptor is attached to a subset of Conv2d modules in the base diffusion backbone via
  forward hooks. The adaptor uses a bottleneck design with down-projection to dimension d
  and up-projection back to the original channel dimension. All adaptor parameters are
  initialized to zero (adaptor_init_zero) to ensure the base model behavior is preserved
  before training.
- The training loop updates only the adaptor ψ parameters; the base diffusion backbone θ is
  frozen.
- The inner adversarial loop requires, for a given timesteps t, access to the diffusion schedule
  quantities α_t and ᾱ_t (cumulative product). This implementation builds a simple linear beta
  schedule if the base model does not expose a schedule, which provides deterministic behavior
  suitable for reproducibility. The exact schedule can be controlled via the config if needed.

Dependencies:
- from model import DiffusionWrapper
- from utils import set_seed, log_metrics, save_checkpoint, load_checkpoint

These utilities are used for deterministic seeding and lightweight logging/checkpointing.

Usage:
- An instance of ANTTrainer is created with a frozen DiffusionWrapper, a target data loader,
  a phi-classifier (fixed during adaptor training), and a config dictionary. The train_step
  method is invoked per training batch/epoch by a higher-level trainer (e.g., main pipeline).

"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DiffusionWrapper
from utils import log_metrics


class ANTTrainer:
    """
    Adaptor Neural Train (ANT) trainer for DPMs-ANT.

    This trainer performs adaptor-only updates using:
      - similarity-guided loss (via a fixed classifier φ)
      - an inner loop to identify worst-case Gaussian noise (adversarial noise selection)

    The adaptor is attached to a pre-trained diffusion backbone θ (frozen during training).
    """

    def __init__(
        self,
        diff_model: DiffusionWrapper,
        data_loader: torch.utils.data.DataLoader,
        phi_classifier: nn.Module,
        config: Dict[str, Any],
    ) -> None:
        """
        Initialize the ANT trainer.

        Args:
            diff_model: DiffusionWrapper instance with adaptor ψ attached. θ is frozen.
            data_loader: PyTorch DataLoader providing target-domain samples (x0).
            phi_classifier: Fixed (non-trainable) binary classifier φ used for similarity guidance.
            config: Dict-like configuration containing hyperparameters (training.gamma, training.J, etc.).
        """
        self.diff_model: DiffusionWrapper = diff_model
        self.data_loader: torch.utils.data.DataLoader = data_loader
        self.phi_classifier: nn.Module = phi_classifier
        self.config: Dict[str, Any] = config

        # Training hyperparameters (with safe defaults)
        training_cfg = config.get("training", {}) or {}
        self.gamma: float = float(training_cfg.get("gamma", 5.0))
        self.J: int = int(training_cfg.get("J", 10))
        self.omega: float = float(training_cfg.get("omega", 0.02))
        self.iterations_per_task: int = int(training_cfg.get("iterations_per_task", 300))
        self.batch_size: int = int(training_cfg.get("batch_size", 40))
        self.lr_adaptor: float = float(training_cfg.get("lr_ddpm", 5e-05))

        # Adaptor configuration (bottleneck size)
        adaptor_cfg = config.get("model", {}).get("adaptor_params_ddpm", {})
        self.adaptor_bottleneck: int = int(adaptor_cfg.get("d", 8))  # default d=8
        # Internal schedule info
        self.T: int = int(config.get("T", 1000))  # timesteps; default to 1000

        # Prepare diffusion schedule (α_t, ᾱ_t). We maintain a simple linear beta schedule
        # if the base model does not expose a schedule. This ensures reproducibility.
        self._alpha_bar_list: List[float] = []
        self._alpha_list: List[float] = []
        self._schedule_built: bool = False
        self._build_schedule(self.T)

        # Freeze base θ and log adaptor scaffolding
        self.diff_model.freeze_base()
        self._adaptor_params: List[torch.nn.Parameter] = []
        for hook in getattr(self.diff_model, "_adaptor_hooks", []):
            self._adaptor_params.extend(list(hook.adaptor.parameters()))

        # Optimizer updates only adaptor ψ
        self.optimizer = torch.optim.Adam(self._adaptor_params, lr=self.lr_adaptor)

        # Track adaptor usage for logging
        self.gamma_tensor = torch.tensor(self.gamma, dtype=torch.float32)

    # ----------------------------
    # Public API helpers
    # ----------------------------
    @property
    def adaptor_param_count(self) -> int:
        """Return total number of adaptor parameters being trained."""
        return sum(p.numel() for p in self._adaptor_params)

    # ----------------------------
    # Schedule utilities
    # ----------------------------
    def _build_schedule(self, T: int) -> None:
        """
        Build a simple linear schedule for α_t and ᾱ_t if not provided by the base model.

        Returns:
            None (populates self._alpha_bar_list and self._alpha_list)
        """
        if self._schedule_built:
            return

        # Linear beta schedule from small start to end
        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, steps=T + 1)[1:]  # t=1..T
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Move to Python floats for quick access in inner loops
        self._alpha_list = [float(a) for a in alphas.tolist()]
        self._alpha_bar_list = [float(ab) for ab in alpha_bar.tolist()]
        self._schedule_built = True

    def _get_schedule_values(self, t: int) -> Tuple[float, float, float, float]:
        """
        Retrieve sqrt(ᾱ_t), sqrt(1 - ᾱ_t), ᾱ_t, α_t for a given timestep t ∈ {1..T}.

        Returns:
            (sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t, alpha_bar_t, alpha_t)
        """
        t = max(1, min(int(t), self.T))
        alpha_bar_t = self._alpha_bar_list[t - 1]
        alpha_t = self._alpha_list[t - 1]
        sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(float(max(0.0, 1.0 - alpha_bar_t)))
        return sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t, alpha_bar_t, alpha_t

    # ----------------------------
    # Inner adversarial loop (Equation 7 approximation)
    # ----------------------------
    def inner_adversarial_loop(
        self, x0: torch.Tensor, t: int, epsilon_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a finite-step gradient ascent to approximate the worst-case Gaussian noise ε*.

        Args:
            x0: input target image, shape [1, C, H, W]
            t: timestep index in {1, ..., T}
            epsilon_init: optional initial epsilon, shape [1, C, H, W]

        Returns:
            (epsilon_star, x_t_star): the adversarial noise and corresponding noised input at timestep t
        """
        # Ensure shapes
        if x0.dim() != 4:
            raise ValueError(f"x0 must be 4D [B, C, H, W], got {x0.shape}")

        # Initialize epsilon
        if epsilon_init is None:
            epsilon = torch.randn_like(x0)
        else:
            epsilon = epsilon_init.clone()

        epsilon = epsilon.detach().requires_grad_(True)

        sqrt_ab, sqrt_1m_ab, alpha_bar_t, alpha_t = self._get_schedule_values(t)

        for _ in range(self.J):
            # Ensure gradient path
            epsilon.requires_grad_(True)

            # x_t = sqrt(ᾱ_t) x0 + sqrt(1 - ᾱ_t) ε
            x_t = sqrt_ab * x0 + sqrt_1m_ab * epsilon

            # ε_theta(x_t, t)
            eps_theta = self.diff_model(x_t, t)

            # Loss encouraging larger reconstruction error w.r.t ε
            loss_adv = ((epsilon - eps_theta) ** 2).sum()

            # Backprop to ε
            if epsilon.grad is not None:
                epsilon.grad.detach_()
                epsilon.grad.zero_()
            loss_adv.backward(retain_graph=True)

            with torch.no_grad():
                grad_eps = epsilon.grad
                # Gradient ascent step
                epsilon = epsilon + self.omega * grad_eps

                # Normalize: mean 0, unit std per sample
                mean = epsilon.mean(dim=[1, 2], keepdim=True)
                epsilon = epsilon - mean
                std = epsilon.std(dim=[1, 2], keepdim=True) + 1e-6
                epsilon = epsilon / std

                # Prepare for next iteration
                epsilon = epsilon.detach().requires_grad_(True)

        epsilon_star = epsilon.detach()
        x_t_star = sqrt_ab * x0 + sqrt_1m_ab * epsilon_star
        return epsilon_star, x_t_star

    # ----------------------------
    # Similarity-guided loss (Equation 5)
    # ----------------------------
    def compute_similarity_loss(
        self,
        x_t_star: torch.Tensor,
        t: int,
        epsilon_star: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the similarity-guided loss term for a given (x_t^*, t, ε^*).

        This function computes ε_θ(x_t^*, t) and ∇_{x_t^*} log p_φ(y=T | x_t^*),
        then constructs the loss term:
            L_sim = || ε^* − ε_θ(x_t^*, t) − σ̂_t^2 γ ∇_{x_t^*} log p_φ(y=T | x_t^*) ||^2

        Args:
            x_t_star: the adversarially noised input at timestep t, requires_grad enabled.
            t: timestep index.
            epsilon_star: the adversarial noise ε^*.

        Returns:
            L_sim (scalar tensor)
        """
        # Ensure x_t_star requires grad for potential second-order paths
        x_t_star = x_t_star.clone().detach().requires_grad_(True)

        # ε_theta
        eps_theta = self.diff_model(x_t_star, t)

        # Gradient of log probability with respect to x_t^*
        # φ should be a 2-class classifier; compute log p(y=T|x)
        logits = self.phi_classifier(x_t_star)
        # Softmax over last dimension; assume logits [N, 2]
        p_T = torch.softmax(logits, dim=-1)[:, 1:2]
        log_p_T = torch.log(p_T + 1e-7)

        # grad of log p_T w.r.t x_t^*
        grad_log_p_T = torch.autograd.grad(
            log_p_T, x_t_star, grad_outputs=torch.ones_like(log_p_T), retain_graph=True, create_graph=True
        )[0]

        # σ̂_t
        t_int = int(t)
        if t_int <= 1:
            sigma_hat = 0.0
        else:
            alpha_bar_t = self._alpha_bar_list[t_int - 1]
            alpha_bar_prev = self._alpha_bar_list[t_int - 2]
            alpha_t = self._alpha_list[t_int - 1]
            # guard against division by zero
            denom = max(1e-12, (1.0 - alpha_bar_t))
            sigma_hat = (1.0 - alpha_bar_prev) * math.sqrt(float(alpha_t) / denom)

        sigma_hat_tensor = torch.tensor(sigma_hat, dtype=x_t_star.dtype, device=x_t_star.device)
        gamma_scalar = torch.tensor(self.gamma, dtype=x_t_star.dtype, device=x_t_star.device)
        # L_sim scalar
        residual = epsilon_star - eps_theta - (sigma_hat_tensor ** 2) * gamma_scalar * grad_log_p_T
        L_sim = (residual * residual).sum()

        return L_sim

    # ----------------------------
    # Per-batch training step (API)
    # ----------------------------
    def train_step(self, batch: Any, epoch: int) -> Dict[str, float]:
        """
        Execute one training step over a batch. Updates adaptor ψ only.

        This method assumes batch is a standard PyTorch batch from data_loader
        returning: (images, labels) or just images. We only use images (x0).

        Args:
            batch: batch from the target data loader.
            epoch: current training epoch or iteration index (for logging).

        Returns:
            A dictionary with scalar metrics for logging (e.g., loss).
        """
        self.diff_model.train()
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        if images.dim() == 3:
            images = images.unsqueeze(0)

        B = images.size(0)
        total_loss = 0.0
        log_entries = {}

        # Iterate through the batch (one sample at a time for stability)
        for i in range(B):
            x0_i = images[i : i + 1]  # [1, C, H, W]

            # Sample t uniformly in [1, T]
            t = random.randint(1, self.T)

            # Inner adversarial loop to obtain worst-case noise ε^*, x_t^*
            epsilon_init = torch.randn_like(x0_i)
            epsilon_star, x_t_star = self.inner_adversarial_loop(x0_i, t, epsilon_init)

            # Compute similarity-guided loss
            L_sim = self.compute_similarity_loss(x_t_star, t, epsilon_star)

            # Backprop through adaptor ψ only
            self.optimizer.zero_grad()
            L_sim.backward()
            self.optimizer.step()

            total_loss += float(L_sim.item())

        avg_loss = total_loss / max(1, B)

        # Simple metrics for logging
        adaptor_norm = sum(p.data.norm().item() for p in self._adaptor_params)
        log_entries = {
            "train/ANT_loss": float(avg_loss),
            "train/adaptor_norm": float(adaptor_norm),
            "train/epoch": int(epoch),
        }

        log_metrics(log_entries)
        return log_entries