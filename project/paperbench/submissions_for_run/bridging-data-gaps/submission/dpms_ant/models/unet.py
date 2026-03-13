"""
U-Net architecture for DDPM, following Dhariwal & Nichol (2021).

This implements the U-Net denoising network used in improved DDPM / guided diffusion.
The architecture includes:
- Timestep embeddings
- ResBlocks with group normalization
- Self-attention at specified resolutions
- Up/down sampling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import List, Optional, Tuple


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D tensor of N indices
        dim: Dimension of the output
        max_period: Controls the minimum frequency of the embeddings

    Returns:
        Tensor of shape (N, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that casts to float32 for stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential module that passes timestep embeddings to children that need it."""

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """2x upsampling with optional convolution."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """2x downsampling with optional convolution."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.conv = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock(TimestepBlock):
    """
    Residual block with timestep conditioning.

    Args:
        channels: Input channels
        emb_channels: Timestep embedding channels
        dropout: Dropout rate
        out_channels: Output channels (defaults to input channels)
        use_scale_shift_norm: Whether to use scale-shift normalization
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            GroupNorm32(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block.

    Args:
        channels: Number of input channels
        num_heads: Number of attention heads
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)

        qkv = self.qkv(self.norm(x_flat))
        q, k, v = qkv.reshape(b * self.num_heads, -1, qkv.shape[2]).chunk(3, dim=1)

        scale = 1 / math.sqrt(math.sqrt(q.shape[1]))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = torch.einsum("bts,bcs->bct", weight, v)
        a = a.reshape(b, -1, a.shape[-1])

        return x_flat.reshape(b, c, *spatial) + self.proj_out(a).reshape(b, c, *spatial)


class UNetModel(nn.Module):
    """
    Full U-Net model with timestep conditioning.

    This follows the architecture from Dhariwal & Nichol (2021) used in
    guided diffusion and DDPM-PA.

    Args:
        image_size: Input image resolution
        in_channels: Number of input channels (3 for RGB)
        model_channels: Base channel count
        out_channels: Number of output channels
        num_res_blocks: Number of ResBlocks per resolution
        attention_resolutions: Resolutions at which to use attention
        dropout: Dropout rate
        channel_mult: Channel multiplier per resolution level
        num_heads: Number of attention heads
        use_scale_shift_norm: Whether to use scale-shift normalization
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 6,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (32, 16, 8),
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4, 4),
        num_heads: int = 4,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = model_channels
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, use_conv=True))
                )
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to an input batch.

        Args:
            x: Input tensor of shape (N, C, H, W)
            timesteps: 1D tensor of N timestep indices

        Returns:
            Output tensor of shape (N, out_channels, H, W)
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)

    def get_feature_channels(self) -> List[int]:
        """Get the channel dimensions at each layer for adaptor injection."""
        channels = []
        ch = self.model_channels
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                ch = mult * self.model_channels
                channels.append(ch)
            if level != len(self.channel_mult) - 1:
                channels.append(ch)
        # Middle block
        channels.append(ch)
        # Output blocks
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for _ in range(self.num_res_blocks + 1):
                ch = self.model_channels * mult
                channels.append(ch)
        return channels


def create_ffhq256_model() -> UNetModel:
    """Create a U-Net model configured for FFHQ 256x256."""
    return UNetModel(
        image_size=256,
        in_channels=3,
        model_channels=128,
        out_channels=6,  # 3 for mean, 3 for variance (learned)
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_heads=4,
        use_scale_shift_norm=True,
    )


def create_church256_model() -> UNetModel:
    """Create a U-Net model configured for LSUN Church 256x256."""
    return UNetModel(
        image_size=256,
        in_channels=3,
        model_channels=128,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_heads=4,
        use_scale_shift_norm=True,
    )
