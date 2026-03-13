"""
Adaptor module for DPMs-ANT.

The adaptor layer is added to each layer of the pre-trained U-Net to learn
the shift gap between source and target domains. During training, only the
adaptor parameters are updated while the pre-trained U-Net remains frozen.

From Section 4.3:
    x_t^l = θ^l(x_t^{l-1}) + ψ^l(x_t^{l-1})

where θ^l is the frozen pre-trained layer and ψ^l is the adaptor.

The adaptor architecture follows Houlsby et al. (2019):
    ψ^l(x) = f(x @ W_down) @ W_up

Parameters:
    - c = 4, d = 8 for DDPMs (spatial downscale factor and bottleneck dim)
    - c = 2, d = 8 for LDMs
    - All adaptor parameters initialized to zero so initial output is zero
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdaptorLayer(nn.Module):
    """
    Single adaptor layer that projects down, applies nonlinearity, and projects back up.

    Architecture:
        input (w, h, r) -> downsample spatial by c -> W_down to bottleneck d
        -> nonlinearity -> W_up back to r -> upsample spatial by c -> output

    Args:
        channels: Number of input/output channels (r)
        spatial_downscale: Spatial downscale factor (c), default 4 for DDPM
        bottleneck_dim: Bottleneck dimension (d), default 8
    """

    def __init__(
        self,
        channels: int,
        spatial_downscale: int = 4,
        bottleneck_dim: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.spatial_downscale = spatial_downscale
        self.bottleneck_dim = bottleneck_dim

        # Down projection: channels -> bottleneck_dim
        self.w_down = nn.Conv2d(channels, bottleneck_dim, kernel_size=1, bias=True)
        # Nonlinearity
        self.act = nn.GELU()
        # Up projection: bottleneck_dim -> channels
        self.w_up = nn.Conv2d(bottleneck_dim, channels, kernel_size=1, bias=True)

        # Initialize to zero so initial output is zero
        self._zero_init()

    def _zero_init(self):
        """Initialize all parameters to zero."""
        nn.init.zeros_(self.w_down.weight)
        nn.init.zeros_(self.w_down.bias)
        nn.init.zeros_(self.w_up.weight)
        nn.init.zeros_(self.w_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        b, c, h, w = x.shape

        # Spatial downscale
        if self.spatial_downscale > 1 and h >= self.spatial_downscale and w >= self.spatial_downscale:
            h_down = F.avg_pool2d(x, kernel_size=self.spatial_downscale)
        else:
            h_down = x

        # Down project
        h_down = self.w_down(h_down)

        # Nonlinearity
        h_down = self.act(h_down)

        # Up project
        h_up = self.w_up(h_down)

        # Spatial upscale back to original size
        if h_up.shape[2] != h or h_up.shape[3] != w:
            h_up = F.interpolate(h_up, size=(h, w), mode="nearest")

        return h_up


class UNetWithAdaptor(nn.Module):
    """
    Wraps a pre-trained UNet with adaptor layers.

    The pre-trained UNet parameters are frozen, and adaptor layers are
    injected at each ResBlock output. The forward pass computes:
        x_t^l = θ^l(x_t^{l-1}) + ψ^l(x_t^{l-1})

    Args:
        unet: Pre-trained UNet model
        spatial_downscale: Spatial downscale factor (c=4 for DDPM, c=2 for LDM)
        bottleneck_dim: Bottleneck dimension (d=8)
    """

    def __init__(
        self,
        unet: nn.Module,
        spatial_downscale: int = 4,
        bottleneck_dim: int = 8,
    ):
        super().__init__()
        self.unet = unet

        # Freeze UNet parameters
        for param in self.unet.parameters():
            param.requires_grad = False

        # Create adaptor layers for input blocks
        self.input_adaptors = nn.ModuleList()
        for module in self.unet.input_blocks:
            ch = self._get_block_out_channels(module)
            if ch is not None:
                self.input_adaptors.append(
                    AdaptorLayer(ch, spatial_downscale, bottleneck_dim)
                )
            else:
                self.input_adaptors.append(None)

        # Middle block adaptor
        mid_ch = self._get_block_out_channels(self.unet.middle_block)
        self.middle_adaptor = AdaptorLayer(mid_ch, spatial_downscale, bottleneck_dim) if mid_ch else None

        # Output block adaptors
        self.output_adaptors = nn.ModuleList()
        for module in self.unet.output_blocks:
            ch = self._get_block_out_channels(module)
            if ch is not None:
                self.output_adaptors.append(
                    AdaptorLayer(ch, spatial_downscale, bottleneck_dim)
                )
            else:
                self.output_adaptors.append(None)

    def _get_block_out_channels(self, block: nn.Module) -> Optional[int]:
        """Infer the output channels of a block."""
        for module in reversed(list(block.modules())):
            if isinstance(module, nn.Conv2d):
                return module.out_channels
            if isinstance(module, nn.Conv1d):
                return module.out_channels
        return None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptor injection.

        Args:
            x: Input tensor of shape (N, C, H, W)
            timesteps: 1D tensor of N timestep indices

        Returns:
            Output tensor of shape (N, out_channels, H, W)
        """
        from dpms_ant.models.unet import timestep_embedding

        emb = self.unet.time_embed(timestep_embedding(timesteps, self.unet.model_channels))

        hs = []
        h = x

        for i, module in enumerate(self.unet.input_blocks):
            h = module(h, emb)
            # Add adaptor output
            if i < len(self.input_adaptors) and self.input_adaptors[i] is not None:
                h = h + self.input_adaptors[i](h)
            hs.append(h)

        h = self.unet.middle_block(h, emb)
        if self.middle_adaptor is not None:
            h = h + self.middle_adaptor(h)

        for i, module in enumerate(self.unet.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            # Add adaptor output
            if i < len(self.output_adaptors) and self.output_adaptors[i] is not None:
                h = h + self.output_adaptors[i](h)

        return self.unet.out(h)

    def get_adaptor_parameters(self):
        """Get only the adaptor parameters for optimization."""
        params = []
        for adaptor in self.input_adaptors:
            if adaptor is not None:
                params.extend(adaptor.parameters())
        if self.middle_adaptor is not None:
            params.extend(self.middle_adaptor.parameters())
        for adaptor in self.output_adaptors:
            if adaptor is not None:
                params.extend(adaptor.parameters())
        return params

    def count_adaptor_parameters(self) -> int:
        """Count the number of adaptor parameters."""
        return sum(p.numel() for p in self.get_adaptor_parameters())

    def count_total_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.parameters())

    def parameter_rate(self) -> float:
        """Compute the ratio of adaptor parameters to total parameters."""
        return self.count_adaptor_parameters() / self.count_total_parameters()
