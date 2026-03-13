## model/adapter.py
```python
"""Adapter module for similarity-guided diffusion model adaptation.

This module implements the adaptor layer ψ^l for each U-Net layer as described
in the paper. The adaptor is applied as:
    x_t^l = θ^l(x_t^{l-1}) + ψ^l(x_t^{l-1})

Where ψ^l(x^{l-1}) = f(x^{l-1}W_down)W_up
- W_down: projects from R^{input_dim} to R^{d*c}
- f: ReLU non-linear activation
- W_up: projects from R^{d*c} back to R^{input_dim}

For DDPMs: c=4, d=8 (intermediate_dim = 32)
For LDMs: c=2, d=8 (intermediate_dim = 16)
"""

import torch
import torch.nn as nn
from torch import Tensor


class Adapter(nn.Module):
    """Adapter module for layer-wise adaptation in diffusion models.
    
    Implements the adaptor transformation ψ^l(x) = f(xW_down)W_up as described
    in Section 4.2 of the paper. This allows for efficient few-shot adaptation
    of pre-trained diffusion models by adding small trainable components to
    each layer while keeping the backbone frozen.
    
    Attributes:
        W_down: Downsampling projection matrix (nn.Linear)
            Projects from input_dim to intermediate_dim = d_dim * c_factor
        activation: ReLU non-linear activation function
        W_up: Upsampling projection matrix (nn.Linear)
            Projects from intermediate_dim back to input_dim
    """
    
    def __init__(
        self,
        input_dim: int,
        c_factor: int,
        d_dim: int
    ) -> None:
        """Initialize adapter with adaptor formula ψ^l(x^{l-1}) = f(x^{l-1}W_down)W_up.
        
        Projects input from R^{input_dim} to R^{d*c} then back to R^{input_dim}.
        
        Args:
            input_dim: Dimension of input features (D in paper notation).
                      For U-Net layers, this is the number of channels.
            c_factor: Downsampling factor c (4 for DDPMs, 2 for LDMs).
                     Controls the reduction ratio in intermediate dimension.
            d_dim: Projection dimension d (typically 8).
                   Used as d*c for intermediate dimension.
        
        Example:
            >>> # For DDPM: c=4, d=8, input_dim=128
            >>> adapter = Adapter(input_dim=128, c_factor=4, d_dim=8)
            >>> # intermediate_dim = 4 * 8 = 32
            >>>
            >>> # For LDM: c=2, d=8, input_dim=128
            >>> adapter = Adapter(input_dim=128, c_factor=2, d_dim=8)
            >>> # intermediate_dim = 2 * 8 = 16
        """
        super().__init__()
        
        # Compute intermediate dimension: d * c
        # This is the bottleneck dimension after downsampling
        intermediate_dim = d_dim * c_factor
        
        # W_down: projects from input_dim to intermediate_dim
        # Shape: [input_dim, intermediate_dim]
        self.W_down = nn.Linear(input_dim, intermediate_dim, bias=False)
        
        # Non-linear activation function f(·) = ReLU
        self.activation = nn.ReLU(inplace=True)
        
        # W_up: projects from intermediate_dim back to input_dim
        # Shape: [intermediate_dim, input_dim]
        self.W_up = nn.Linear(intermediate_dim, input_dim, bias=False)
        
        # Initialize weights using Xavier uniform for better gradient flow
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize linear layer weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.W_down.weight)
        nn.init.xavier_uniform_(self.W_up.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply adaptor transformation ψ(x) = f(xW_down)W_up.
        
        Args:
            x: Input tensor of shape [B, input_dim] or [B, H, W, input_dim]
               For U-Net layers, typically [B, channels] or [B, C, H, W]
        
        Returns:
            Output tensor of same shape as input [B, input_dim]
        
        Example:
            >>> x = torch.randn(32, 128)  # batch=32, channels=128
            >>> adapter = Adapter(input_dim=128, c_factor=4, d_dim=8)
            >>> output = adapter(x)  # [32, 128]
        """
        # Apply downprojection: x -> W_down
        x_down = self.W_down(x)
        
        # Apply non-linear activation: f(·)
        x_activated = self.activation(x_down)
        
        # Apply upprojection: -> W_up
        x_up = self.W_up(x_activated)
        
        return x_up