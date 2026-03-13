## model/unet.py
"""U-Net backbone for diffusion models.

This module implements the pre-trained U-Net backbone θ^l used in the similarity-guided
diffusion model adaptation. The backbone is kept frozen during training while adaptor
layers ψ^l are added to each layer for few-shot adaptation.

Architecture follows Ho et al. (2020) DDPM design with:
- Encoder (down_blocks): progressively downsampling feature maps
- Bottleneck (mid_block): processing at lowest resolution
- Decoder (up_blocks): progressively upsampling with skip connections
- Time embedding: sinusoidal positional encoding for timestep t

The forward pass predicts noise ε_θ(x_t, t) from noisy input x_t at timestep t.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, List
from torch import Tensor


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for timesteps.
    
    Implements sinusoidal positional encoding to embed timestep t into a
    high-dimensional vector, following the original DDPM design.
    
    Attributes:
        dim: Dimension of the embedding vector
    """
    
    def __init__(self, dim: int) -> None:
        """Initialize time embedding layer.
        
        Args:
            dim: Dimension of the embedding vector (typically model_channels * 4)
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, t: Tensor) -> Tensor:
        """Compute sinusoidal time embedding.
        
        Args:
            t: Timestep tensor of shape [B] with integer timesteps
        
        Returns:
            Embedded timestep tensor of shape [B, dim]
        """
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with group normalization.
    
    Standard residual block used in DDPM U-Net with:
    - GroupNorm normalization
    - SiLU activation
    - Time embedding conditioning via linear projection
    - Skip connection
    
    Attributes:
        channels: Number of input/output channels
        time_emb_dim: Dimension of time embedding
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        out_channels: Optional[int] = None
    ) -> None:
        """Initialize residual block.
        
        Args:
            channels: Number of input channels
            time_emb_dim: Dimension of time embedding
            dropout: Dropout probability
            out_channels: Number of output channels (if None, equals channels)
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = channels
        
        self.channels = channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        
        # First normalization and conv
        self.norm1 = nn.GroupNorm(32, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Second normalization and conv
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection projection if channels differ
        if channels != out_channels:
            self.skip = nn.Conv2d(channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        """Apply residual block transformation.
        
        Args:
            x: Input tensor [B, C, H, W]
            time_emb: Time embedding [B, time_emb_dim]
        
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        h = self.conv1(self.act1(self.norm1(x)))
        
        # Add time embedding conditioning
        h = h + self.time_emb_proj(self.act2(time_emb))[:, :, None, None]
        
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Spatial attention block for U-Net.
    
    Applies self-attention across spatial dimensions for feature refinement.
    
    Attributes:
        channels: Number of channels
        num_heads: Number of attention heads
    """
    
    def __init__(self, channels: int, num_heads: int = 1) -> None:
        """Initialize attention block.
        
        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply attention mechanism.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H * W)
        
        # Project and add residual
        out = self.proj(out)
        out = out.reshape(B, C, H, W)
        
        return x + out


class Downsample(nn.Module):
    """Downsampling layer for encoder path."""
    
    def __init__(self, channels: int) -> None:
        """Initialize downsampling.
        
        Args:
            channels: Number of channels
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply downsampling.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor [B, C, H/2, W/2]
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer for decoder path."""
    
    def __init__(self, channels: int) -> None:
        """Initialize upsampling.
        
        Args:
            channels: Number of channels
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply upsampling with nearest neighbor interpolation.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor [B, C, H*2, W*2]
        """
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNetBackbone(nn.Module):
    """U-Net backbone for noise prediction in diffusion models.
    
    This is the pre-trained frozen backbone θ^l used in the similarity-guided
    adaptation framework. The architecture follows Ho et al. (2020) DDPM with:
    - Encoder path: progressively downsampling with residual blocks
    - Bottleneck: processing at lowest resolution
    - Decoder path: progressively upsampling with skip connections
    
    The model predicts noise ε_θ(x_t, t) from noisy input x_t at timestep t.
    
    Attributes:
        time_embed: Time embedding MLP for timestep conditioning
        down_blocks: List of downsampling blocks for encoder
        mid_block: Middle/bottleneck block
        up_blocks: List of upsampling blocks for decoder
        layer_output_dims: List of output dimensions for each layer (for adapters)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple = (8, 16),
        dropout: float = 0.0,
        channel_mult: Tuple = (1, 2, 4, 4),
        num_heads: int = 4,
        use_spatial_transformer: bool = False,
        context_dim: Optional[int] = None
    ) -> None:
        """Initialize U-Net backbone with standard diffusion model architecture.
        
        Args:
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            out_channels: Number of output channels (same as in_channels for noise prediction)
            model_channels: Base number of channels (128 or 256)
            num_res_blocks: Number of residual blocks per layer
            attention_resolutions: Tuple of resolutions to apply attention
            dropout: Dropout probability (0.0 for no dropout)
            channel_mult: Channel multiplier for each resolution level
            num_heads: Number of attention heads
            use_spatial_transformer: Whether to use spatial transformer (False for standard DDPM)
            context_dim: Dimension for conditioning context (None by default)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        
        # Time embedding dimension is 4x model channels
        time_emb_dim = model_channels * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input conv
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Track layer dimensions for adapter insertion
        self.layer_output_dims: List[int] = []
        
        # Build encoder (down blocks)
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        ds = 1  # Downsampling factor
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(ch, time_emb_dim, dropout, out_ch)
                )
                self.layer_output_dims.append(out_ch)
                ch = out_ch
            
            # Downsample at all but last level
            if level < len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
                ds *= 2
                self.layer_output_dims.append(ch)  # Record after downsample too
        
        # Build middle block
        self.mid_block = nn.Sequential(
            ResBlock(ch, time_emb_dim, dropout, ch),
            AttentionBlock(ch, num_heads) if ch in [
                model_channels * channel_mult[i] 
                for i, res in enumerate([4, 8, 16, 32])
                if model_channels * mult >= 64
            ] or ch >= 64 else nn.Identity(),
            ResBlock(ch, time_emb_dim, dropout, ch)
        )
        self.layer_output_dims.append(ch)
        
        # Build decoder (up blocks)
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):  # +1 for skip connection concat
                self.up_blocks.append(
                    ResBlock(ch + ch if i > 0 else ch, time_emb_dim, dropout, out_ch)
                )
                self.layer_output_dims.append(out_ch)
                ch = out_ch
            
            # Upsample at all but last level
            if level > 0:
                self.up_blocks.append(Upsample(ch))
        
        # Output conv
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
        
        # Store for feature extraction
        self.time_emb_dim = time_emb_dim
    
    def forward(self, x: Tensor, t: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """Forward pass through U-Net: predict noise ε_θ(x_t, t).
        
        Args:
            x: Noisy input tensor [B, C, H, W] (x_t in paper notation)
            t: Timestep indices [B] (t in paper notation)
            context: Optional conditioning context [B, context_dim] (unused in standard DDPM)
        
        Returns:
            Predicted noise tensor [B, C, H, W] (ε_θ(x_t, t) in paper notation)
        """
        # Time embedding
        time_emb = self.time_embed(t)
        
        # Input convolution
        h = self.input_conv(x)
        
        # Encoder path with skip connections
        hs = [h]
        for module in self.down_blocks:
            if isinstance(module, ResBlock):
                h = module(h, time_emb)
            elif isinstance(module, Downsample):
                h = module(h)
            hs.append(h)
        
        # Middle block
        h = self.mid_block(h)
        
        # Decoder path
        for module in self.up_blocks:
            if isinstance(module, ResBlock):
                # Concatenate skip connection
                if isinstance(hs[-1], Tensor):
                    h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)
            elif isinstance(module, Upsample):
                h = module(h)
        
        # Output convolution
        h = self.output_conv(h)
        
        return h
    
    def get_layer_features(self, x: Tensor, t: Tensor) -> List[Tensor]:
        """Extract intermediate features from each layer l for adaptor insertion.
        
        This method processes the input through each layer and collects the output
        features, which are used for computing:
            x_t^l = θ^l(x_t^{l-1}) + ψ^l(x_t^{l-1})
        
        Args:
            x: Input tensor [B, C, H, W] (x_t^{l-1})
            t: Timestep indices [B]
        
        Returns:
            List of feature tensors at each layer output dimension.
            Each tensor can be used to compute ψ^l(x_t^{l-1}).
        """
        # Time embedding
        time_emb = self.time_embed(t)
        
        # Input convolution
        h = self.input_conv(x)
        
        # Collect features from each layer
        features = []
        
        # Encoder features
        for module in self.down_blocks:
            if isinstance(module, ResBlock):
                h = module(h, time_emb)
                features.append(h)
            elif isinstance(module, Downsample):
                h = module(h)
        
        # Middle features
        h = self.mid_block(h)
        features.append(h)
        
        # Decoder features (before upsampling)
        for module in self.up_blocks:
            if isinstance(module, ResBlock):
                if features and len(features) > 0:  # skip connection concat happens in forward
                    pass  # We only collect pre-concat features
                h = module(h, time_emb)
                features.append(h)
            elif isinstance(module, Upsample):
                h = module(h)
        
        return features
    
    def freeze(self) -> None:
        """Freeze all parameters of the U-Net backbone.
        
        Sets requires_grad=False for all parameters. This is called during training
        to keep θ frozen while only updating the adaptor parameters ψ.
        """
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze parameters if needed for debugging or fine-tuning.
        
        Sets requires_grad=True for all parameters. Allows the backbone to be
        fine-tuned or inspected.
        """
        for param in self.parameters():
            param.requires_grad = True


def create_unet_backbone(config: dict) -> UNetBackbone:
    """Factory function to create U-Net backbone from configuration.
    
    Args:
        config: Dictionary containing model configuration parameters
    
    Returns:
        UNetBackbone instance configured according to config
    """
    return UNetBackbone(
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('out_channels', 3),
        model_channels=config.get('model_channels', 128),
        num_res_blocks=config.get('num_res_blocks', 2),
        attention_resolutions=config.get('attention_resolutions', (8, 16)),
        dropout=config.get('dropout', 0.0),
        channel_mult=config.get('channel_mult', (1, 2, 4, 4)),
        num_heads=config.get('num_heads', 4),
        use_spatial_transformer=config.get('use_spatial_transformer', False),
        context_dim=config.get('context_dim', None)
    )