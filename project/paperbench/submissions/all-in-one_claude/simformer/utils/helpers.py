"""
Utility functions and helper classes for Simformer.
"""

import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Get the best available device (MPS for Mac M3, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier projection for time embedding.

    Projects scalar time values to a high-dimensional embedding using
    random Fourier features, as used in score-based diffusion models.

    Args:
        embed_dim: Dimension of the output embedding (will be 2 * embed_dim due to sin/cos)
        scale: Scale factor for the random frequencies
    """

    def __init__(self, embed_dim: int = 64, scale: float = 30.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Randomly sample frequencies from N(0, scale^2)
        self.register_buffer(
            "W", torch.randn(embed_dim) * scale, persistent=True
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values of shape (batch_size,) or (batch_size, 1)

        Returns:
            Embedding of shape (batch_size, 2 * embed_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)

        # Compute 2 * pi * W * t
        t_proj = 2 * math.pi * t * self.W  # (batch_size, embed_dim)

        # Concatenate sin and cos features
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for continuous-valued positions.

    Used for encoding time points in function-valued parameters.

    Args:
        embed_dim: Dimension of the output embedding
        max_positions: Maximum number of positions (for normalization)
    """

    def __init__(self, embed_dim: int, max_positions: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions

        # Compute frequency bands
        half_dim = embed_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        self.register_buffer(
            "freq_bands", torch.exp(torch.arange(half_dim) * -emb), persistent=True
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Position values of shape (batch_size,) or (batch_size, num_positions)

        Returns:
            Embedding of shape (batch_size, embed_dim) or (batch_size, num_positions, embed_dim)
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(-1)

        # Expand dimensions for broadcasting
        pos_expanded = positions.unsqueeze(-1)  # (..., 1)
        freq_expanded = self.freq_bands  # (half_dim,)

        # Compute embeddings
        embeddings = pos_expanded * freq_expanded  # (..., half_dim)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        activation: Activation function
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def batch_eye(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create a batch of identity matrices."""
    return torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)


def unsqueeze_like(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unsqueeze tensor to match the number of dimensions of target."""
    while tensor.dim() < target.dim():
        tensor = tensor.unsqueeze(-1)
    return tensor


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from tensor a at indices t and reshape for broadcasting.

    Args:
        a: Tensor of values to extract from
        t: Tensor of indices
        x_shape: Shape to broadcast to

    Returns:
        Extracted values reshaped for broadcasting
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def exists(val) -> bool:
    """Check if a value exists (is not None)."""
    return val is not None


def default(val, default_val):
    """Return val if it exists, otherwise return default_val."""
    return val if exists(val) else default_val


def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [0, 1] to [-1, 1]."""
    return x * 2 - 1


def unnormalize_to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    """Unnormalize tensor from [-1, 1] to [0, 1]."""
    return (x + 1) / 2


class EMA:
    """
    Exponential Moving Average for model parameters.

    Args:
        model: Model whose parameters to track
        decay: Decay rate for EMA
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
