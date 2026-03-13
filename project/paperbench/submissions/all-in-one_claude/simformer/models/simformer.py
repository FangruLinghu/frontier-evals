"""
Simformer: All-in-one simulation-based inference model.

This is the main model class that combines all components:
- Tokenizer for SBI
- Score network (transformer-based)
- Diffusion model (VESDE/VPSDE)
- Mask generator for dependencies
"""

from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn

from simformer.tokenizer.tokenizer import SBITokenizer
from simformer.models.score_network import ConditionalScoreNetwork
from simformer.masks.attention_masks import MaskGenerator


class Simformer(nn.Module):
    """
    Simformer model for all-in-one simulation-based inference.

    The Simformer can:
    - Sample arbitrary conditionals of the joint p(θ, x)
    - Estimate posterior p(θ|x), likelihood p(x|θ), and any other conditional
    - Handle function-valued (∞-dimensional) parameters
    - Incorporate dependency structures via attention masks

    Args:
        n_params: Number of parameter variables
        n_data: Number of data variables
        token_dim: Dimension of token embeddings (default: 50)
        time_embed_dim: Dimension of time embedding (default: 128)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 4)
        widening_factor: Feed-forward widening factor (default: 3)
        dropout: Dropout probability (default: 0.0)
        mask_type: Type of attention mask ("dense", "directed", "undirected")
        use_fourier_values: Whether to use Fourier features for values
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
        mask_type: str = "dense",
        use_fourier_values: bool = True,
        param_structure: Optional[torch.Tensor] = None,
        data_structure: Optional[torch.Tensor] = None,
        param_to_data: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.n_params = n_params
        self.n_data = n_data
        self.n_variables = n_params + n_data
        self.token_dim = token_dim
        self.mask_type = mask_type

        # Tokenizer
        self.tokenizer = SBITokenizer(
            n_params=n_params,
            n_data=n_data,
            token_dim=token_dim,
            use_fourier_values=use_fourier_values,
        )

        # Score network
        self.score_network = ConditionalScoreNetwork(
            n_params=n_params,
            n_data=n_data,
            token_dim=token_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            widening_factor=widening_factor,
            dropout=dropout,
        )

        # Mask generator
        self.mask_generator = MaskGenerator(
            n_params=n_params,
            n_data=n_data,
            mask_type=mask_type,
            param_structure=param_structure,
            data_structure=data_structure,
            param_to_data=param_to_data,
        )

    def forward(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        condition_mask: torch.Tensor,
        theta_positions: Optional[torch.Tensor] = None,
        x_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute score estimates.

        Args:
            theta: Parameter values of shape (batch_size, n_params)
            x: Data values of shape (batch_size, n_data)
            t: Diffusion time of shape (batch_size,)
            condition_mask: Binary mask of shape (batch_size, n_params + n_data)
            theta_positions: Optional positions for function-valued parameters
            x_positions: Optional positions for function-valued data

        Returns:
            Score estimates of shape (batch_size, n_params + n_data)
        """
        # Create tokens
        tokens = self.tokenizer.create_tokens(
            theta, x, condition_mask, theta_positions, x_positions
        )

        # Get attention mask
        attention_mask = self.mask_generator.get_edge_mask(condition_mask)

        # Compute score
        scores = self.score_network(tokens, t, condition_mask, attention_mask)

        return scores

    def compute_score(
        self,
        values: torch.Tensor,
        t: torch.Tensor,
        condition_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute score from concatenated values.

        Args:
            values: Concatenated values of shape (batch_size, n_params + n_data)
            t: Diffusion time of shape (batch_size,)
            condition_mask: Binary mask of shape (batch_size, n_params + n_data)
            positions: Optional positions for function-valued variables

        Returns:
            Score estimates of shape (batch_size, n_params + n_data)
        """
        theta = values[:, :self.n_params]
        x = values[:, self.n_params:]

        if positions is not None:
            theta_positions = positions[:, :self.n_params]
            x_positions = positions[:, self.n_params:]
        else:
            theta_positions = None
            x_positions = None

        return self.forward(theta, x, t, condition_mask, theta_positions, x_positions)

    def posterior_score(
        self,
        theta_t: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score for posterior estimation p(θ|x).

        Args:
            theta_t: Noisy parameter values at time t
            x: Observed data values (clean)
            t: Diffusion time

        Returns:
            Score estimates for parameters
        """
        batch_size = theta_t.shape[0]
        device = theta_t.device

        # Create posterior condition mask (condition on x)
        condition_mask = torch.zeros(batch_size, self.n_variables, device=device)
        condition_mask[:, self.n_params:] = 1.0

        scores = self.forward(theta_t, x, t, condition_mask)

        # Return only parameter scores
        return scores[:, :self.n_params]

    def likelihood_score(
        self,
        theta: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score for likelihood estimation p(x|θ).

        Args:
            theta: Observed parameter values (clean)
            x_t: Noisy data values at time t
            t: Diffusion time

        Returns:
            Score estimates for data
        """
        batch_size = theta.shape[0]
        device = theta.device

        # Create likelihood condition mask (condition on theta)
        condition_mask = torch.zeros(batch_size, self.n_variables, device=device)
        condition_mask[:, :self.n_params] = 1.0

        scores = self.forward(theta, x_t, t, condition_mask)

        # Return only data scores
        return scores[:, self.n_params:]

    def joint_score(
        self,
        theta_t: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score for joint distribution p(θ, x).

        Args:
            theta_t: Noisy parameter values at time t
            x_t: Noisy data values at time t
            t: Diffusion time

        Returns:
            Score estimates for all variables
        """
        batch_size = theta_t.shape[0]
        device = theta_t.device

        # Create joint condition mask (no conditioning)
        condition_mask = torch.zeros(batch_size, self.n_variables, device=device)

        return self.forward(theta_t, x_t, t, condition_mask)

    def arbitrary_conditional_score(
        self,
        values_t: torch.Tensor,
        t: torch.Tensor,
        condition_mask: torch.Tensor,
        clean_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute score for arbitrary conditional distribution.

        This allows sampling any conditional of the joint p(θ, x).

        Args:
            values_t: Noisy values at time t (batch_size, n_variables)
            t: Diffusion time
            condition_mask: Binary mask indicating which variables are conditioned
            clean_values: Clean values for conditioned variables (optional)

        Returns:
            Score estimates for latent variables
        """
        # If clean values provided, replace conditioned variables
        if clean_values is not None:
            values_t = values_t * (1 - condition_mask) + clean_values * condition_mask

        theta_t = values_t[:, :self.n_params]
        x_t = values_t[:, self.n_params:]

        return self.forward(theta_t, x_t, t, condition_mask)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "n_params": self.n_params,
            "n_data": self.n_data,
            "token_dim": self.token_dim,
            "mask_type": self.mask_type,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Simformer":
        """Create model from configuration."""
        return cls(**config)


def create_simformer(
    n_params: int,
    n_data: int,
    task_complexity: str = "standard",
    mask_type: str = "dense",
    **kwargs,
) -> Simformer:
    """
    Factory function to create Simformer with task-appropriate settings.

    Args:
        n_params: Number of parameter variables
        n_data: Number of data variables
        task_complexity: "standard" (6 layers) or "complex" (8 layers)
        mask_type: Type of attention mask
        **kwargs: Additional arguments to pass to Simformer

    Returns:
        Configured Simformer model
    """
    # Default settings from paper
    config = {
        "token_dim": 50,
        "time_embed_dim": 128,
        "num_heads": 4,
        "widening_factor": 3,
        "dropout": 0.0,
    }

    # Adjust layers based on task complexity
    if task_complexity == "standard":
        config["num_layers"] = 6
    elif task_complexity == "complex":
        config["num_layers"] = 8
    else:
        raise ValueError(f"Unknown task complexity: {task_complexity}")

    # Override with any provided kwargs
    config.update(kwargs)

    return Simformer(
        n_params=n_params,
        n_data=n_data,
        mask_type=mask_type,
        **config,
    )
