"""
Transformer architecture for Simformer.

Implements the transformer encoder with support for attention masks
that can encode dependency structures between variables.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with support for attention masks.

    As described in the paper:
    attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

    The attention mask can be used to enforce dependency structures.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional mask of shape (seq_len, seq_len) or
                           (batch_size, seq_len, seq_len) or
                           (batch_size, num_heads, seq_len, seq_len)
                           1 = attend, 0 = mask out

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (batch_size, num_heads, seq_len, seq_len)

        # Apply attention mask
        if attention_mask is not None:
            # Expand mask dimensions if necessary
            if attention_mask.dim() == 2:
                # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(1)

            # Convert mask: 1 = attend, 0 = mask out
            # We use large negative values for masked positions
            attn_scores = attn_scores.masked_fill(
                attention_mask == 0, float("-inf")
            )

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Handle NaN from all-masked rows
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)
        # (batch_size, num_heads, seq_len, head_dim)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.

    Args:
        embed_dim: Input/output dimension
        hidden_dim: Hidden layer dimension (typically 4x embed_dim)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with pre-norm architecture.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Pre-norm attention block
        x = x + self.attention(self.norm1(x), attention_mask)

        # Pre-norm feed-forward block
        x = x + self.ff(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for Simformer.

    Based on the paper specifications:
    - Token dimension: 50
    - 6 layers (8 for complex tasks)
    - 4 heads
    - Widening factor: 3

    Args:
        embed_dim: Token embedding dimension (default: 50)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 4)
        widening_factor: Factor for feed-forward hidden dim (default: 3)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        embed_dim: int = 50,
        num_layers: int = 6,
        num_heads: int = 4,
        widening_factor: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        ff_dim = embed_dim * widening_factor

        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all transformer layers.

        Args:
            x: Input tokens of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask encoding dependencies

        Returns:
            Output of shape (batch_size, seq_len, embed_dim)
        """
        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.final_norm(x)

        return x


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization for conditioning on time.

    Projects time embedding to scale and shift parameters for layer norm.

    Args:
        embed_dim: Embedding dimension
        time_embed_dim: Time embedding dimension
    """

    def __init__(self, embed_dim: int, time_embed_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * embed_dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer norm.

        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
            t_emb: Time embedding of shape (batch_size, time_embed_dim)

        Returns:
            Normalized output of shape (batch_size, seq_len, embed_dim)
        """
        # Project time embedding to scale and shift
        params = self.proj(t_emb)  # (batch_size, 2 * embed_dim)
        scale, shift = params.chunk(2, dim=-1)  # Each (batch_size, embed_dim)

        # Expand for broadcasting
        scale = scale.unsqueeze(1)  # (batch_size, 1, embed_dim)
        shift = shift.unsqueeze(1)  # (batch_size, 1, embed_dim)

        # Apply adaptive norm
        x = self.norm(x)
        x = x * (1 + scale) + shift

        return x


class TransformerEncoderLayerWithTime(nn.Module):
    """
    Transformer encoder layer with time conditioning via AdaLN.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        time_embed_dim: Time embedding dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.adaln1 = AdaLN(embed_dim, time_embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.adaln2 = AdaLN(embed_dim, time_embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with time-conditioned adaptive layer norm.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            t_emb: Time embedding of shape (batch_size, time_embed_dim)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # AdaLN attention block
        x = x + self.attention(self.adaln1(x, t_emb), attention_mask)

        # AdaLN feed-forward block
        x = x + self.ff(self.adaln2(x, t_emb))

        return x


class TransformerEncoderWithTime(nn.Module):
    """
    Transformer encoder with time conditioning for diffusion models.

    Args:
        embed_dim: Token embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        time_embed_dim: Time embedding dimension
        widening_factor: Factor for feed-forward hidden dim
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 50,
        num_layers: int = 6,
        num_heads: int = 4,
        time_embed_dim: int = 128,
        widening_factor: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        ff_dim = embed_dim * widening_factor

        # Stack of transformer layers with time conditioning
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithTime(
                embed_dim, num_heads, ff_dim, time_embed_dim, dropout
            )
            for _ in range(num_layers)
        ])

        # Final adaptive layer norm
        self.final_adaln = AdaLN(embed_dim, time_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all transformer layers with time conditioning.

        Args:
            x: Input tokens of shape (batch_size, seq_len, embed_dim)
            t_emb: Time embedding of shape (batch_size, time_embed_dim)
            attention_mask: Optional attention mask encoding dependencies

        Returns:
            Output of shape (batch_size, seq_len, embed_dim)
        """
        for layer in self.layers:
            x = layer(x, t_emb, attention_mask)

        x = self.final_adaln(x, t_emb)

        return x
