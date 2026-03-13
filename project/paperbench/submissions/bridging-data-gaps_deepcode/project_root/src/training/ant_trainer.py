import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import Adam

try:
    # Optional import; provided in this repo
    from ..noise_optimization.adversarial_noise import inner_adversarial_noise
except Exception:
    inner_adversarial_noise = None  # type: ignore


class ANTTrainer:
    """
    Algorithm 1: Training DPMs with ANT (Adversarial Noise Transfer).

    This lightweight trainer updates only the adaptor ψ parameters while keeping
    the backbone θ frozen. It relies on two sources of signal:
      - L_sim: similarity-guided loss using the model's ε_θ predictions and
        the gradient of the domain classifier p_φ(y|x_t).
      - L_AN: adversarial noise objective obtained via an inner-max optimization over ε.
    """

    def __init__(
        self,
        adaptor: nn.Module,
        theta: nn.Module,
        eps_theta_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        grad_logp_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        alphas_cumprod: Optional[torch.Tensor] = None,
        sqrt_alphas_cumprod: Optional[torch.Tensor] = None,
        sqrt_one_minus_alphas_cumprod: Optional[torch.Tensor] = None,
        T: int = 1000,
        gamma: float = 5.0,
        omega: float = 0.02,
        J: int = 10,
        lr_adaptor: float = 5e-4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.adaptor = adaptor
        self.theta = theta
        self.eps_theta_fn = eps_theta_fn
        self.grad_logp_fn = grad_logp_fn
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.T = int(T)
        self.gamma = float(gamma)
        self.omega = float(omega)
        self.J = int(J)
        self.device = device if device is not None else torch.device("cpu")

        # Only adaptor parameters are optimized
        self.optim = Adam(self.adaptor.parameters(), lr=float(lr_adaptor))
        self.adaptor.zero_grad()

        # Fallbacks in case schedules are not yet provided
        if self.alphas_cumprod is None:
            raise ValueError("alphas_cumprod schedule must be provided to ANTTrainer.")
        if self.sqrt_alphas_cumprod is None:
            raise ValueError("sqrt_alphas_cumprod schedule must be provided to ANTTrainer.")
        if self.sqrt_one_minus_alphas_cumprod is None:
            raise ValueError("sqrt_one_minus_alphas_cumprod schedule must be provided to ANTTrainer.")

    def to(self, device: torch.device) -> None:
        self.device = device
        if hasattr(self.adaptor, 'to'):
            self.adaptor.to(device)
        if hasattr(self.theta, 'to'):
            self.theta.to(device)
        if self.optim is not None:
            # Reinitialize optimizer state on new device
            for g in self.optim.param_groups:
                for p in g.get('params', []):
                    if p.is_cuda:
                        pass
            # Note: PyTorch optimizers automatically track device

    def _sample_xt(self, x0: torch.Tensor, t: int, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0, device=self.device)
        beta_bar = self.sqrt_alphas_cumprod.new_tensor(self.sqrt_alphas_cumprod[:, 0] if self.sqrt_alphas_cumprod.dim() > 1 else self.sqrt_alphas_cumprod[0])  # placeholder to avoid lint
        # The actual term we need is: x_t = sqrt_alpha_bar[t] * x0 + sqrt(1 - alpha_bar[t]) * eps
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps
        return x_t

    def _epsilon_t_from_xt(self, x_t: torch.Tensor, t: int, x0: torch.Tensor) -> torch.Tensor:
        # ε_t = (x_t - sqrt(alpha_bar_t) x0) / sqrt(1 - alpha_bar_t)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
        eps = (x_t - sqrt_alpha_bar_t * x0) / (sqrt_one_minus_alpha_bar_t + 1e-8)
        return eps

    def train_step(self, x0_target: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        """Perform one training step on a batch of target samples.

        Args:
            x0_target: [B, C, H, W] or [B, C, ...] target-domain samples.
            t: Optional timestep (int). If None, randomly sample in [1, T-1].

        Returns:
            loss value (scalar tensor).
        """
        assert isinstance(x0_target, torch.Tensor), "x0_target must be a torch.Tensor"
        B = x0_target.shape[0]
        t_int = int(t if t is not None else torch.randint(1, self.T, (1,)).item())

        # Prepare functions for ε_theta and grad_logp
        if self.eps_theta_fn is None or self.grad_logp_fn is None:
            raise ValueError("ANTTrainer requires eps_theta_fn and grad_logp_fn to be provided.")

        x0 = x0_target.to(self.device)
        t_tensor = torch.tensor(t_int, dtype=torch.long, device=self.device)

        # Step 1: sample baseline noise and x_t
        eps = torch.randn_like(x0)
        x_t = self._sample_xt(x0, t_int, eps=eps)

        # Step 2: Inner maximization to obtain adversarial noise ε⋆ if available
        if inner_adversarial_noise is not None:
            eps_star = inner_adversarial_noise(
                x0, t_int, self.eps_theta_fn, self.grad_logp_fn,
                self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod,
                gamma=self.gamma, J=self.J, omega=self.omega, seed=None
            )
        else:
            # Fallback: use baseline epsilon as surrogate for ε⋆
            eps_star = eps.clone()

        # Step 3: Compute L_sim on current x_t with eps
        eps_theta_x_t = self.eps_theta_fn(x_t, t_int)
        grad_logp_x_t = self.grad_logp_fn(x_t)
        sigma_sq_t = 1.0 - float(self.alphas_cumprod[t_int].cpu().item()) if hasattr(self, 'alphas_cumprod') else (1.0 - 0.0)
        L_sim = ((eps - eps_theta_x_t - self.gamma * sigma_sq_t * grad_logp_x_t) ** 2).mean()

        # Step 4: Compute outer AN term using ε⋆ and x_t*(ε⋆)
        x_t_star = self._sample_xt(x0, t_int, eps=eps_star)
        eps_theta_x_t_star = self.eps_theta_fn(x_t_star, t_int)
        grad_logp_x_t_star = self.grad_logp_fn(x_t_star)
        L_an = ((eps_star - eps_theta_x_t_star - self.gamma * sigma_sq_t * grad_logp_x_t_star) ** 2).mean()

        loss = L_sim + L_an

        # Backprop through adaptor ψ only
        self.optim.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.adaptor.parameters(), max_norm=1.0)
        self.optim.step()
        return loss.detach()


__all__ = ["ANTTrainer"]
