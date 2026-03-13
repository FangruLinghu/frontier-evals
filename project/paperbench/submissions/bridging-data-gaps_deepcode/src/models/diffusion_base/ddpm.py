import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Minimal DDPM backbone scaffold for the DPMs-ANT project.

- Provides a small CNN-based epsilon predictor ε_theta(x_t, t).
- Exposes utilities to sample x_t from a given x_0 and timestep t using a simple
  linear (pseudo) schedule for alpha_bar_t.
- Note: This is a lightweight, educational placeholder intended to enable end-to-end
  integration with the adaptor and training loop. It is not a full-scale DDPM.
"""

class DDPMBackbone(nn.Module):
    def __init__(self, image_size: int = 32, channels: int = 3, num_timesteps: int = 10, hidden_dim: int = 64):
        super(DDPMBackbone, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_timesteps = int(num_timesteps)
        self.hidden_dim = hidden_dim
        # A compact, transformer-free U-Net-like scaffold (very small) for ε_theta
        # This is a lightweight encoder-decoder style block with residual-like path
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1),
        )
        # Optional small learnable scaling to stabilize gradients
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict epsilon given noisy image x_t and timestep t.
        Args:
            x_t: Noisy latent at time step t, shape (B, C, H, W)
            t: Timesteps, can be scalar tensor or plain tensor/list; will be coerced to float
        Returns:
            ε_theta(x_t, t): predicted noise, same shape as x_t
        """
        return self.predict_eps(x_t, t)

    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise component ε_theta for input x_t and timestep t.
        This is a lightweight predictor; in practice this would be a U-Net conditioned on t.
        """
        # Simple conditioning by broadcasting timestep into a channel-wise map could be added here.
        # For now, rely on the convolutional network to act as a denoiser conditioned implicitly by x_t.
        eps = self.net(x_t)
        return eps * self.scale

    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None) -> torch.Tensor:
        """Sample x_t from x0 using the forward diffusion equation with a simple schedule.
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * ε
        Args:
            x0: Original clean image, shape (B, C, H, W)
            t: Timestep(s). Can be scalar or tensor. Assumes 0 <= t <= num_timesteps.
            eps: Optional pre-sampled ε ~ N(0, I) with same shape as x0.
        Returns:
            x_t: Noisy image at time t
        """
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar = self.alpha_bar_t(t, device=x0.device)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * eps
        return xt

    def alpha_bar_t(self, t: torch.Tensor, device=None) -> torch.Tensor:
        """Compute alpha_bar_t for given t using a simple schedule.
        We use a lightweight linear schedule for demonstration:
            a_bar_t = 1.0 - t / max(1, num_timesteps)
        The function accepts both scalar and tensor t for batch processing.
        """
        if device is None:
            # infer device from t if possible
            device = t.device if isinstance(t, torch.Tensor) else torch.device("cpu")
        # Normalize t to [0, 1] range
        if isinstance(t, torch.Tensor):
            t_f = t.to(dtype=torch.float32, device=device)
        else:
            t_f = torch.tensor(float(t), dtype=torch.float32, device=device)
        # Ensure proper shape for broadcasting
        T = max(1.0, float(self.num_timesteps))
        a_bar = 1.0 - t_f / T
        a_bar = torch.clamp(a_bar, min=1e-5, max=0.999)
        return a_bar

# Minimal helper factory for external imports
def get_ddpm_backbone(**kwargs) -> DDPMBackbone:
    return DDPMBackbone(**kwargs)
