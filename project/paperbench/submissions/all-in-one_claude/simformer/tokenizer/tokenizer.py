"""
Tokenizer for Simulation-Based Inference (SBI).

As described in the paper, the tokenizer represents each variable as:
1. An identifier that uniquely identifies the variable
2. A representation of the value of the variable
3. A condition state (whether the variable is conditioned on or not)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import torch.nn as nn

from simformer.utils.helpers import GaussianFourierProjection, SinusoidalPositionalEncoding


@dataclass
class Token:
    """
    A token representing a single variable in the SBI setting.

    Attributes:
        id_embedding: Embedding representing the variable's identity
        value_embedding: Embedding representing the variable's value
        condition_embedding: Embedding representing the condition state
        combined: Combined token embedding
    """

    id_embedding: torch.Tensor
    value_embedding: torch.Tensor
    condition_embedding: torch.Tensor
    combined: torch.Tensor


class SBITokenizer(nn.Module):
    """
    Tokenizer for SBI that converts (θ, x) pairs into token sequences.

    The tokenizer creates embeddings for:
    - Variable identifiers (learned embeddings)
    - Variable values (linear projection + optional Fourier features)
    - Condition states (learned embeddings for conditioned/latent)

    For function-valued parameters, the identifier includes a positional encoding
    of the index set (e.g., time points).

    Args:
        n_params: Number of parameter variables
        n_data: Number of data variables
        token_dim: Dimension of the output token embedding
        value_embed_dim: Dimension for value embedding before projection
        use_fourier_values: Whether to use Fourier features for value embedding
        fourier_scale: Scale for Fourier features
        max_positions: Maximum positions for function-valued parameters
    """

    def __init__(
        self,
        n_params: int,
        n_data: int,
        token_dim: int = 50,
        value_embed_dim: int = 32,
        use_fourier_values: bool = True,
        fourier_scale: float = 10.0,
        max_positions: float = 100.0,
    ):
        super().__init__()

        self.n_params = n_params
        self.n_data = n_data
        self.n_total = n_params + n_data
        self.token_dim = token_dim
        self.value_embed_dim = value_embed_dim
        self.use_fourier_values = use_fourier_values

        # Learnable identifier embeddings for each variable
        self.id_embeddings = nn.Embedding(self.n_total, token_dim)

        # Learnable condition state embeddings (0 = latent, 1 = conditioned)
        self.condition_embeddings = nn.Embedding(2, token_dim)

        # Value embedding network
        if use_fourier_values:
            self.value_fourier = GaussianFourierProjection(
                embed_dim=value_embed_dim // 2, scale=fourier_scale
            )
            value_input_dim = value_embed_dim
        else:
            value_input_dim = 1

        self.value_projection = nn.Sequential(
            nn.Linear(value_input_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # Positional encoding for function-valued parameters
        self.positional_encoding = SinusoidalPositionalEncoding(
            embed_dim=token_dim, max_positions=max_positions
        )

        # Layer norm for final token
        self.layer_norm = nn.LayerNorm(token_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.id_embeddings.weight, std=0.02)
        nn.init.normal_(self.condition_embeddings.weight, std=0.02)

    def embed_values(self, values: torch.Tensor) -> torch.Tensor:
        """
        Embed variable values.

        Args:
            values: Values of shape (batch_size, n_variables)

        Returns:
            Value embeddings of shape (batch_size, n_variables, token_dim)
        """
        batch_size, n_vars = values.shape

        if self.use_fourier_values:
            # Apply Fourier features to each value
            # Reshape for processing
            values_flat = values.reshape(-1, 1)  # (batch_size * n_vars, 1)
            fourier_features = self.value_fourier(values_flat.squeeze(-1))  # (batch_size * n_vars, value_embed_dim)
            value_embeds = self.value_projection(fourier_features)
            value_embeds = value_embeds.reshape(batch_size, n_vars, self.token_dim)
        else:
            # Simple linear projection
            values_expanded = values.unsqueeze(-1)  # (batch_size, n_vars, 1)
            value_embeds = self.value_projection(values_expanded)

        return value_embeds

    def embed_identifiers(
        self,
        batch_size: int,
        positions: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Embed variable identifiers.

        Args:
            batch_size: Batch size
            positions: Optional positions for function-valued parameters
                       Shape: (batch_size, n_variables) or None
            device: Device to create embeddings on

        Returns:
            Identifier embeddings of shape (batch_size, n_variables, token_dim)
        """
        if device is None:
            device = self.id_embeddings.weight.device

        # Get base identifier embeddings
        indices = torch.arange(self.n_total, device=device)
        id_embeds = self.id_embeddings(indices)  # (n_total, token_dim)
        id_embeds = id_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_total, token_dim)

        # Add positional encoding for function-valued parameters if provided
        if positions is not None:
            pos_embeds = self.positional_encoding(positions)
            id_embeds = id_embeds + pos_embeds

        return id_embeds

    def embed_conditions(
        self,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed condition states.

        Args:
            condition_mask: Binary mask of shape (batch_size, n_variables)
                           1 = conditioned, 0 = latent

        Returns:
            Condition embeddings of shape (batch_size, n_variables, token_dim)
        """
        # Convert to long indices for embedding lookup
        condition_indices = condition_mask.long()  # (batch_size, n_variables)

        # Get embeddings
        return self.condition_embeddings(condition_indices)

    def forward(
        self,
        values: torch.Tensor,
        condition_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create token embeddings from values and condition masks.

        Args:
            values: Variable values of shape (batch_size, n_variables)
            condition_mask: Binary condition mask of shape (batch_size, n_variables)
            positions: Optional positions for function-valued parameters

        Returns:
            Token embeddings of shape (batch_size, n_variables, token_dim)
        """
        batch_size = values.shape[0]
        device = values.device

        # Get individual embeddings
        id_embeds = self.embed_identifiers(batch_size, positions, device)
        value_embeds = self.embed_values(values)
        condition_embeds = self.embed_conditions(condition_mask)

        # Combine embeddings (sum as in the paper)
        tokens = id_embeds + value_embeds + condition_embeds

        # Apply layer norm
        tokens = self.layer_norm(tokens)

        return tokens

    def create_tokens(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        condition_mask: torch.Tensor,
        theta_positions: Optional[torch.Tensor] = None,
        x_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create tokens from separate theta and x tensors.

        Args:
            theta: Parameter values of shape (batch_size, n_params)
            x: Data values of shape (batch_size, n_data)
            condition_mask: Condition mask of shape (batch_size, n_params + n_data)
            theta_positions: Optional positions for function-valued parameters
            x_positions: Optional positions for function-valued data

        Returns:
            Token embeddings of shape (batch_size, n_params + n_data, token_dim)
        """
        # Concatenate theta and x
        values = torch.cat([theta, x], dim=-1)

        # Concatenate positions if provided
        if theta_positions is not None or x_positions is not None:
            if theta_positions is None:
                theta_positions = torch.zeros_like(theta)
            if x_positions is None:
                x_positions = torch.zeros_like(x)
            positions = torch.cat([theta_positions, x_positions], dim=-1)
        else:
            positions = None

        return self.forward(values, condition_mask, positions)


class EmbeddingNetwork(nn.Module):
    """
    Embedding network for high-dimensional data (e.g., time series, images).

    This can be used to compress complex data into a single token, as mentioned
    in the paper for tasks like gravitational waves.

    Args:
        input_dim: Input dimension
        output_dim: Output token dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed high-dimensional input into a single token.

        Args:
            x: Input of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim)

        Returns:
            Embedding of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        return self.network(x)


class ConvEmbeddingNetwork(nn.Module):
    """
    Convolutional embedding network for sequence data.

    Useful for tasks like gravitational waves where the data is a long time series.

    Args:
        input_channels: Number of input channels
        output_dim: Output token dimension
        hidden_channels: List of hidden channel dimensions
        kernel_size: Convolution kernel size
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden_channels: List[int] = [32, 64, 128],
        kernel_size: int = 5,
    ):
        super().__init__()

        layers = []
        in_channels = input_channels

        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Final projection
        self.projection = nn.Linear(hidden_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed sequence data into a single token.

        Args:
            x: Input of shape (batch_size, seq_len) or (batch_size, channels, seq_len)

        Returns:
            Embedding of shape (batch_size, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        x = self.conv_layers(x)
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.projection(x)

        return x
