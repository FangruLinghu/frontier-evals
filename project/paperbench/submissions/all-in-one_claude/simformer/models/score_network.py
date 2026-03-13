"""
Score Network for Simformer.

The score network takes tokens and time as input and outputs the score
(gradient of log probability) for each variable.
"""

from typing import Optional
import torch
import torch.nn as nn

from simformer.models.transformer import TransformerEncoderWithTime
from simformer.utils.helpers import GaussianFourierProjection


class ScoreNetwork(nn.Module):
    """
    Score network that combines tokenizer, transformer, and output projection.

    The network estimates the score function s(x_t, t) = ∇_{x_t} log p_t(x_t)
    for the diffusion model.

    Architecture:
    1. Time embedding via Gaussian Fourier features
    2. Transformer encoder with time conditioning
    3. Linear projection to score output

    Args:
        n_variables: Total number of variables (parameters + data)
        token_dim: Dimension of token embeddings
        time_embed_dim: Dimension of time embedding
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        widening_factor: Feed-forward widening factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_variables: int,
        token_dim: int = 50,
        time_embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        widening_factor: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_variables = n_variables
        self.token_dim = token_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding network
        # As per paper: 128-dimensional random Gaussian Fourier embedding
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_embed_dim // 2, scale=30.0),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Transformer encoder with time conditioning
        self.transformer = TransformerEncoderWithTime(
            embed_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            time_embed_dim=time_embed_dim,
            widening_factor=widening_factor,
            dropout=dropout,
        )

        # Output projection: from token_dim to 1 (score for each variable)
        self.output_proj = nn.Linear(token_dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute score estimates for all variables.

        Args:
            tokens: Token embeddings of shape (batch_size, n_variables, token_dim)
            t: Diffusion time of shape (batch_size,)
            attention_mask: Optional attention mask of shape (n_variables, n_variables)
                           or (batch_size, n_variables, n_variables)

        Returns:
            Score estimates of shape (batch_size, n_variables)
        """
        # Embed time
        t_emb = self.time_embed(t)  # (batch_size, time_embed_dim)

        # Pass through transformer
        hidden = self.transformer(tokens, t_emb, attention_mask)
        # (batch_size, n_variables, token_dim)

        # Project to scores
        scores = self.output_proj(hidden).squeeze(-1)  # (batch_size, n_variables)

        return scores

    def get_score(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        condition_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get score for latent (unconditioned) variables only.

        Args:
            tokens: Token embeddings
            t: Diffusion time
            condition_mask: Binary mask (1 = conditioned, 0 = latent)
            attention_mask: Optional attention mask

        Returns:
            Score estimates for latent variables (conditioned variables get 0)
        """
        scores = self.forward(tokens, t, attention_mask)

        # Zero out scores for conditioned variables
        latent_mask = 1 - condition_mask
        scores = scores * latent_mask

        return scores


class ConditionalScoreNetwork(nn.Module):
    """
    Conditional score network that can handle different conditioning scenarios.

    This extends the basic score network to support:
    - Different attention masks based on conditioning
    - Optional embedding networks for high-dimensional data

    Args:
        n_params: Number of parameter variables
        n_data: Number of data variables
        token_dim: Dimension of token embeddings
        time_embed_dim: Dimension of time embedding
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        widening_factor: Feed-forward widening factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_params: int,
        n_data: int,
        token_dim: int = 50,
        time_embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        widening_factor: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_params = n_params
        self.n_data = n_data
        self.n_variables = n_params + n_data

        # Base score network
        self.score_network = ScoreNetwork(
            n_variables=self.n_variables,
            token_dim=token_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            widening_factor=widening_factor,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        condition_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute conditional score estimates.

        The score is only computed for latent (unconditioned) variables.
        Conditioned variables receive zero score.

        Args:
            tokens: Token embeddings of shape (batch_size, n_variables, token_dim)
            t: Diffusion time of shape (batch_size,)
            condition_mask: Binary mask of shape (batch_size, n_variables)
                           1 = conditioned, 0 = latent
            attention_mask: Optional attention mask

        Returns:
            Score estimates of shape (batch_size, n_variables)
        """
        return self.score_network.get_score(
            tokens, t, condition_mask, attention_mask
        )

    def posterior_score(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute score for posterior estimation (data conditioned, parameters latent).

        Args:
            tokens: Token embeddings
            t: Diffusion time
            attention_mask: Optional attention mask

        Returns:
            Score estimates for parameters (data scores are zero)
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Create posterior condition mask
        condition_mask = torch.zeros(batch_size, self.n_variables, device=device)
        condition_mask[:, self.n_params:] = 1.0  # Condition on data

        return self.forward(tokens, t, condition_mask, attention_mask)

    def likelihood_score(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute score for likelihood estimation (parameters conditioned, data latent).

        Args:
            tokens: Token embeddings
            t: Diffusion time
            attention_mask: Optional attention mask

        Returns:
            Score estimates for data (parameter scores are zero)
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Create likelihood condition mask
        condition_mask = torch.zeros(batch_size, self.n_variables, device=device)
        condition_mask[:, :self.n_params] = 1.0  # Condition on parameters

        return self.forward(tokens, t, condition_mask, attention_mask)

    def joint_score(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute score for joint distribution (no conditioning).

        Args:
            tokens: Token embeddings
            t: Diffusion time
            attention_mask: Optional attention mask

        Returns:
            Score estimates for all variables
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Create joint condition mask (all zeros = no conditioning)
        condition_mask = torch.zeros(batch_size, self.n_variables, device=device)

        return self.forward(tokens, t, condition_mask, attention_mask)
