import math
from typing import Optional

import torch
import torch.nn as nn


class DenoiserBlock(nn.Module):
    """A lightweight per-channel denoiser block that conditions on a time embedding.

    It applies two 3x3 convolutions with a residual style shaping and adds a
    per-channel time conditioning after the second convolution.
    """

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], t_emb: [B, time_dim]
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        # Add time conditioning: broadcast t_emb across spatial dims
        t_proj = self.time_mlp(t_emb)  # [B, C]
        h = h + t_proj[:, :, None, None]
        return h


class LDMBackbone(nn.Module):
    """A lightweight Latent Diffusion Model (LDM) backbone.

    This module operates in a latent space, taking x_t (latent representation at time t)
    and predicting the noise ε_θ(x_t, t). The backbone is designed to be lightweight
    for demonstration purposes and to integrate with the Houlsby-style adaptor framework.

    - If input channels differ from latent_dim, a small input projection is applied.
    - Time conditioning is provided via a sinusoidal embedding fed through a small MLP.
    - θ (the backbone parameters) are intended to be frozen during adaptor training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 32,
        time_embedding_dim: int = 128,
        max_time_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embedding_dim = time_embedding_dim
        self.max_time_steps = max_time_steps

        # Optional projection if input latent has different dimensionality than the model's latent_dim
        if in_channels != latent_dim:
            self.input_proj = nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1)
        else:
            self.input_proj = nn.Identity()

        # Simple time embedding via sinusoidal functions
        self._time_embedding_cache: Optional[torch.Tensor] = None
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.denoiser = DenoiserBlock(latent_dim, time_embedding_dim)

    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] with integer timesteps in [0, max_time_steps)
        B = t.shape[0]
        device = t.device
        # Sinusoidal embedding with half_dim pairs of sin/cos
        half_dim = max(1, self.time_embedding_dim // 2)
        i = torch.arange(half_dim, device=device).float()
        # frequencies as in the original Transformer-style positional encodings
        freqs = torch.exp(-i * (math.log(10000.0) / half_dim))
        t = t.float().unsqueeze(1)  # [B, 1]
        args = t * freqs.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, time_embedding_dim]
        if emb.shape[1] < self.time_embedding_dim:
            # pad if needed
            pad = self.time_embedding_dim - emb.shape[1]
            emb = torch.cat([emb, torch.zeros(B, pad, device=device, dtype=emb.dtype)], dim=1)
        return emb

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: [B, in_channels, H, W] latent representation at time t
        z = self.input_proj(x_t)  # [B, latent_dim, H, W]
        t_emb = self._get_time_embedding(t)  # [B, time_embedding_dim]
        t_cond = self.time_mlp(t_emb)  # [B, latent_dim]
        eps = self.denoiser(z, t_cond)
        return eps


__all__ = ["LDMBackbone"]
