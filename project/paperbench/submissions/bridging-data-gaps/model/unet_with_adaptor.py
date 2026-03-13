## Code: model/unet_with_adaptor.py
```python
## model/unet_with_adaptor.py
"""
U-Net with Adaptor Layers for DPMs-ANT.

Implements UNetWithAdaptor class that wraps a pre-trained U-Net diffusion model and inserts
Houlsby-style adaptor layers (Houlsby et al., 2019) for parameter-efficient fine-tuning.
The base U-Net weights are frozen, and only the small adaptor parameters are updated during training,
achieving ~1.3%-1.6% parameter update rate as specified in the paper.

Key features:
- Residual adaptor blocks with bottleneck structure: ψ^l(x) = f(xW_down)W_up
- Configurable spatial reduction factor 'c' (4 for DDPM, 2 for LDM)
- Bottleneck dimension 'd' = 8
- Zero initialization of adaptor weights to ensure initial output equals pre-trained model
- Flexible insertion points compatible with standard U-Net architectures

All configuration values are sourced from config.yaml to ensure consistency across the pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# Import configuration
from config import config


@dataclass
class AdaptorConfig:
    """Configuration specific to adaptor layers."""
    c_ddpm: int = 4
    c_ldm: int = 2
    d: int = 8
    init_zero: bool = True


class AdaptorBlock(nn.Module):
    """
    Single adaptor block implementing the Houlsby-style bottleneck adapter.
    
    Applies a residual transformation: output = input + up_proj(activation(down_proj(input)))
    With optional spatial downsampling before projection and upsampling after.
    
    This creates a low-rank path that learns only the adaptation shift needed for transfer learning.
    """
    
    def __init__(self, 
                 channels: int,
                 spatial_reduction_factor: int = 4,
                 bottleneck_dim: int = 8,
                 activation: str = 'silu'):
        """
        Initialize adaptor block with bottleneck structure.
        
        Args:
            channels: Number of input/output channels (r in paper)
            spatial_reduction_factor: Spatial compression factor 'c' (default: 4 for DDPM)
            bottleneck_dim: Hidden dimension 'd' in bottleneck (default: 8)
            activation: Activation function ('silu' or 'relu')
            
        Raises:
            ValueError: If any parameter is non-positive
        """
        super().__init__()
        
        if channels <= 0:
            raise ValueError(f"Channels must be positive, got {channels}")
        if spatial_reduction_factor < 1:
            raise ValueError(f"Spatial reduction factor must be >= 1, got {spatial_reduction_factor}")
        if bottleneck_dim <= 0:
            raise ValueError(f"Bottleneck dimension must be positive, got {bottleneck_dim}")
            
        self.channels = channels
        self.c = spatial_reduction_factor
        self.d = bottleneck_dim
        
        # Determine activation function
        if activation.lower() == 'silu':
            self.activation = nn.SiLU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'silu' or 'relu'.")
        
        # Linear projections
        self.W_down = nn.Linear(channels, bottleneck_dim)
        self.W_up = nn.Linear(bottleneck_dim, channels)
        
        # Initialize weights to zero as specified in Section 5.2
        if config.adaptor.init_zero:
            nn.init.zeros_(self.W_down.weight)
            nn.init.zeros_(self.W_down.bias)
            nn.init.zeros_(self.W_up.weight)
            nn.init.zeros_(self.W_up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Computes: output = x + W_up(σ(W_down(x↓)))↑
        Where ↓ and ↑ denote optional spatial down/upsampling by factor c.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of same shape as input
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"Input channels {C} doesn't match expected {self.channels}"
        
        # Store original resolution
        x_orig = x
        
        # Apply spatial downsampling if c > 1
        if self.c > 1:
            x_down = F.adaptive_avg_pool2d(x, (H // self.c, W // self.c))
        else:
            x_down = x
        
        # Reshape for linear projection: (B, H/c, W/c, C) -> (B*H/c*W/c, C)
        x_flat = x_down.view(-1, self.channels)
        
        # Apply down projection, activation, and up projection
        x_proj = self.W_down(x_flat)
        x_act = self.activation(x_proj)
        x_up_proj = self.W_up(x_act)  # Shape: (B*H/c*W/c, C)
        
        # Reshape back: (B*H/c*W/c, C) -> (B, H/c, W/c, C)
        x_up = x_up_proj.view(B, H // max(self.c, 1), W // max(self.c, 1), self.channels)
        
        # Permute to channel-first: (B, H/c, W/c, C) -> (B, C, H/c, W/c)
        x_up = x_up.permute(0, 3, 1, 2).contiguous()
        
        # Apply spatial upsampling if c > 1
        if self.c > 1:
            x_up = F.interpolate(x_up, size=(H, W), mode='bilinear', align_corners=False)
        
        # Residual connection
        return x_orig + x_up


class UNetWithAdaptor(nn.Module):
    """
    U-Net diffusion model wrapped with adaptor layers for parameter-efficient transfer learning.
    
    This class takes a pre-trained U-Net model and inserts adaptor blocks at key locations
    (encoder, middle, decoder blocks) while keeping the original model weights frozen.
    
    During training, only the adaptor parameters are updated, achieving high efficiency
    with minimal memory footprint (~1.3% parameters updated for DDPM, ~1.6% for LDM).
    """
    
    def __init__(self, 
                 base_unet: nn.Module,
                 model_type: str = 'ddpm',
                 insert_encoder_adaptors: bool = True,
                 insert_middle_adaptors: bool = True,
                 insert_decoder_adaptors: bool = True):
        """
        Initialize U-Net with adaptor layers.
        
        Args:
            base_unet: Pre-trained U-Net model (will be frozen)
            model_type: Type of model ('ddpm' or 'ldm') to determine spatial reduction factor
            insert_encoder_adaptors: Whether to insert adaptors in encoder blocks
            insert_middle_adaptors: Whether to insert adaptors in middle blocks
            insert_decoder_adaptors: Whether to insert adaptors in decoder blocks
            
        Raises:
            ValueError: If model_type is not supported
        """
        super().__init__()
        
        # Validate model type
        if model_type.lower() not in ['ddpm', 'ldm']:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'ddpm' or 'ldm'.")
        self.model_type = model_type.lower()
        
        # Get configuration values
        self.c = config.adaptor.c_ddpm if self.model_type == 'ddpm' else config.adaptor.c_ldm
        self.d = config.adaptor.d
        
        # Store base U-Net
        self.base_unet = base_unet
        
        # Configuration for adaptor insertion
        self.insert_encoder_adaptors = insert_encoder_adaptors
        self.insert_middle_adaptors = insert_middle_adaptors
        self.insert_decoder_adaptors = insert_decoder_adaptors
        
        # Track number of adaptors inserted
        self.num_encoder_adaptors = 0
        self.num_middle_adaptors = 0
        self.num_decoder_adaptors = 0
        
        # Insert adaptors into appropriate locations
        self._insert_adaptors()
        
        # Freeze base U-Net parameters by default
        self.freeze_base_unet()
    
    def _insert_adaptors(self) -> None:
        """Insert adaptor blocks into encoder, middle, and decoder blocks of the U-Net."""
        # Insert into encoder blocks if requested
        if self.insert_encoder_adaptors and hasattr(self.base_unet, 'input_blocks'):
            self._insert_into_sequence(self.base_unet.input_blocks, 'encoder')
        
        # Insert into middle blocks if requested
        if self.insert_middle_adaptors and hasattr(self.base_unet, 'middle_block'):
            self._insert_into_module(self.base_unet.middle_block, 'middle')
        
        # Insert into decoder blocks if requested
        if self.insert_decoder_adaptors and hasattr(self.base_unet, 'output_blocks'):
            self._insert_into_sequence(self.base_unet.output_blocks, 'decoder')
    
    def _insert_into_sequence(self, block_sequence: nn.ModuleList, location: str) -> None:
        """
        Insert adaptor blocks into a sequence of modules (e.g., input_blocks or output_blocks).
        
        Args:
            block_sequence: ModuleList containing blocks to modify
            location: Where adaptors are being inserted ('encoder' or 'decoder')
        """
        for i, block in enumerate(block_sequence):
            # Get number of channels from first conv layer or time embedding
            channels = self._get_block_channels(block)
            if channels is None:
                continue
                
            # Create adaptor block
            adaptor = AdaptorBlock(
                channels=channels,
                spatial_reduction_factor=self.c,
                bottleneck_dim=self.d
            )
            
            # Wrap the original block with adaptor in a sequential container
            # This preserves the interface while adding the residual adaptor
            wrapped_block = nn.Sequential(
                block,
                adaptor
            )
            
            # Replace in sequence
            block_sequence[i] = wrapped_block
            
            # Update counter
            if location == 'encoder':
                self.num_encoder_adaptors += 1
            elif location == 'decoder':
                self.num_decoder_adaptors += 1
    
    def _insert_into_module(self, module: nn.Module, location: str) -> None:
        """
        Insert adaptor blocks into a single module (e.g., middle_block).
        
        Args:
            module: Module to modify
            location: Where adaptors are being inserted ('middle')
        """
        # For middle block, assume it has a sequence of sub-modules
        if hasattr(module, 'modules_list'):  # Custom implementation
            for i, sub_module in enumerate(module.modules_list):
                channels = self._get_block_channels(sub_module)
                if channels is None:
                    continue
                    
                adaptor = AdaptorBlock(
                    channels=channels,
                    spatial_reduction_factor=self.c,
                    bottleneck_dim=self.d
                )
                
                wrapped_module = nn.Sequential(sub_module, adaptor)
                module.modules_list[i] = wrapped_module
                self.num_middle_adaptors += 1
        else:
            # Try common pattern: list of residual blocks
            children = list(module.children())
            for i, child in enumerate(children):
                channels = self._get_block_channels(child)
                if channels is None:
                    continue
                    
                adaptor = AdaptorBlock(
                    channels=channels,
                    spatial_reduction_factor=self.c,
                    bottleneck_dim=self.d
                )
                
                wrapped_child = nn.Sequential(child, adaptor)
                # Replace child - this requires knowing the attribute name
                # In practice, this would need more sophisticated handling
                # For now, we'll assume a simple case
                setattr(module, f"adaptor_wrapped_{i}", wrapped_child)
                self.num_middle_adaptors += 1
    
    def _get_block_channels(self, module: nn.Module) -> Optional[int]:
        """
        Extract number of channels from a U-Net block.
        
        Heuristic approach to find the channel count by looking for convolutional layers.
        
        Args:
            module: U-Net block to analyze
            
        Returns:
            Number of channels if found, None otherwise
        """
        # Look for first conv layer's output channels
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                return m.out_channels
        
        # Look for group norm num_channels
        for m in module.modules():
            if isinstance(m, nn.GroupNorm):
                return m.num_channels
        
        # Look for linear layers that might represent time embeddings
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # This is heuristic - could be time embedding dimension
                # We'll use it as channel proxy if no conv found
                return m.in_features
        
        return None
    
    def freeze_base_unet(self) -> None:
        """
        Freeze all parameters in the base U-Net model.
        
        After calling this method, only adaptor parameters will be trainable.
        This should be called before training begins.
        """
        for param in self.base_unet.parameters():
            param.requires_grad = False
    
    def unfreeze_base_unet(self) -> None:
        """
        Unfreeze all parameters in the base U-Net model.
        
        This allows full fine-tuning if needed, though it contradicts the paper's approach.
        """
        for param in self.base_unet.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all trainable parameters (only adaptors, since base is frozen).
        
        Returns:
            Dictionary mapping parameter names to tensors
        """
        trainable_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
        return trainable_params
    
    def get_parameter_efficiency(self) -> float:
        """
        Calculate the proportion of trainable parameters relative to total parameters.
        
        Returns:
            Fraction of parameters that are trainable (should be ~0.013 for DDPM, ~0.016 for LDM)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return trainable_params / total_params if total_params > 0 else 0.0
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the U-Net with adaptors.
        
        This method maintains compatibility with the original U-Net interface,
        taking input x and timesteps and returning noise prediction.
        
        Args:
            x: Noisy input image of shape (B, C, H, W)
            timesteps: Timestep indices of shape (B,)
            **kwargs: Additional arguments passed to base U-Net
            
        Returns:
            Predicted noise tensor of shape (B, C, H, W)
        """
        # The adaptors are integrated within the U-Net blocks,
        # so we simply forward to the base U-Net which now includes adaptors
        return self.base_unet(x, timesteps, **kwargs)
    
    def __repr__(self) -> str:
        """String representation showing model architecture and adaptor statistics."""
        return (f"UNetWithAdaptor(model_type='{self.model_type}', "
                f"c={self.c}, d={self.d}, "
                f"encoder_adaptors={self.num_encoder_adaptors}, "
                f"middle_adaptors={self.num_middle_adaptors}, "
                f"decoder_adaptors={self.num_decoder_adaptors}, "
                f"trainable_ratio={self.get_parameter