"""
Attention masks for Simformer.

This module implements condition masks (MC) and edge masks (ME) as described in the paper.
- Condition mask (MC): Binary mask indicating which variables are conditioned on
- Edge mask (ME): Attention mask representing dependency structure between variables
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn


def create_dense_mask(n: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a dense attention mask where all variables can attend to each other.

    Args:
        n: Number of variables
        device: Device to create mask on

    Returns:
        Dense mask of shape (n, n) with all ones
    """
    return torch.ones(n, n, device=device)


def create_diagonal_mask(n: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a diagonal (identity) attention mask.

    This corresponds to independent variables that cannot attend to each other.

    Args:
        n: Number of variables
        device: Device to create mask on

    Returns:
        Identity mask of shape (n, n)
    """
    return torch.eye(n, device=device)


def create_directed_mask(
    n_params: int,
    n_data: int,
    param_structure: Optional[torch.Tensor] = None,
    data_structure: Optional[torch.Tensor] = None,
    param_to_data: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a directed attention mask for the simulator's dependency structure.

    In a directed mask:
    - Parameters can attend to other parameters based on param_structure
    - Data can attend to parameters that generated them
    - Data can attend to other data based on data_structure

    Args:
        n_params: Number of parameter variables
        n_data: Number of data variables
        param_structure: (n_params, n_params) adjacency matrix for parameter dependencies
        data_structure: (n_data, n_data) adjacency matrix for data dependencies
        param_to_data: (n_data, n_params) matrix indicating which parameters generate which data
        device: Device to create mask on

    Returns:
        Directed mask of shape (n_params + n_data, n_params + n_data)
    """
    n_total = n_params + n_data

    # Initialize with zeros
    mask = torch.zeros(n_total, n_total, device=device)

    # Parameter block: parameters can attend to themselves and other parameters
    if param_structure is not None:
        mask[:n_params, :n_params] = param_structure.to(device)
    else:
        # Default: fully connected parameters
        mask[:n_params, :n_params] = torch.ones(n_params, n_params, device=device)

    # Data can attend to parameters
    if param_to_data is not None:
        mask[n_params:, :n_params] = param_to_data.to(device)
    else:
        # Default: all data can attend to all parameters
        mask[n_params:, :n_params] = torch.ones(n_data, n_params, device=device)

    # Data block: data can attend to other data
    if data_structure is not None:
        mask[n_params:, n_params:] = data_structure.to(device)
    else:
        # Default: identity (each data point attends only to itself)
        mask[n_params:, n_params:] = torch.eye(n_data, device=device)

    # Self-attention: all variables attend to themselves
    mask.fill_diagonal_(1.0)

    return mask


def create_undirected_mask(
    n_params: int,
    n_data: int,
    param_structure: Optional[torch.Tensor] = None,
    data_structure: Optional[torch.Tensor] = None,
    param_to_data: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create an undirected attention mask by symmetrizing a directed mask.

    Args:
        n_params: Number of parameter variables
        n_data: Number of data variables
        param_structure: (n_params, n_params) adjacency matrix for parameter dependencies
        data_structure: (n_data, n_data) adjacency matrix for data dependencies
        param_to_data: (n_data, n_params) matrix indicating which parameters generate which data
        device: Device to create mask on

    Returns:
        Undirected (symmetric) mask of shape (n_params + n_data, n_params + n_data)
    """
    directed = create_directed_mask(
        n_params, n_data, param_structure, data_structure, param_to_data, device
    )

    # Symmetrize: if either direction has an edge, both directions have an edge
    undirected = torch.maximum(directed, directed.T)

    return undirected


def sample_condition_mask(
    batch_size: int,
    n_params: int,
    n_data: int,
    mask_type: str = "random",
    p_bernoulli: float = 0.5,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample condition masks for training.

    The condition mask indicates which variables are observed (conditioned on).

    Args:
        batch_size: Number of masks to sample
        n_params: Number of parameter variables
        n_data: Number of data variables
        mask_type: Type of mask to sample:
            - "joint": All zeros (unconditional)
            - "posterior": Data conditioned, parameters latent
            - "likelihood": Parameters conditioned, data latent
            - "random": Random Bernoulli mask
            - "mixed": Mixture of the above
        p_bernoulli: Probability for Bernoulli sampling
        device: Device to create mask on

    Returns:
        Condition mask of shape (batch_size, n_params + n_data)
    """
    n_total = n_params + n_data

    if mask_type == "joint":
        # Unconditional: all variables are latent
        return torch.zeros(batch_size, n_total, device=device)

    elif mask_type == "posterior":
        # Posterior: data is observed, parameters are latent
        mask = torch.zeros(batch_size, n_total, device=device)
        mask[:, n_params:] = 1.0  # Condition on data
        return mask

    elif mask_type == "likelihood":
        # Likelihood: parameters are observed, data is latent
        mask = torch.zeros(batch_size, n_total, device=device)
        mask[:, :n_params] = 1.0  # Condition on parameters
        return mask

    elif mask_type == "random":
        # Random Bernoulli mask
        return (torch.rand(batch_size, n_total, device=device) < p_bernoulli).float()

    elif mask_type == "mixed":
        # Mixture strategy as described in the paper:
        # Uniformly sample joint, posterior, likelihood, or two random masks
        masks = torch.zeros(batch_size, n_total, device=device)

        for i in range(batch_size):
            choice = torch.randint(0, 4, (1,)).item()
            if choice == 0:
                # Joint
                pass  # Already zeros
            elif choice == 1:
                # Posterior
                masks[i, n_params:] = 1.0
            elif choice == 2:
                # Likelihood
                masks[i, :n_params] = 1.0
            else:
                # Random with different probabilities
                p = 0.3 if torch.rand(1).item() < 0.5 else 0.7
                masks[i] = (torch.rand(n_total, device=device) < p).float()

        return masks

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def update_mask_for_conditioning(
    edge_mask: torch.Tensor,
    condition_mask: torch.Tensor,
    n_params: int,
) -> torch.Tensor:
    """
    Update the edge mask based on the condition mask for directed graphs.

    When conditioning on certain variables, we need to add edges to ensure
    all dependencies are captured (Webb et al., 2018).

    Args:
        edge_mask: (n, n) base edge mask
        condition_mask: (batch_size, n) or (n,) condition mask
        n_params: Number of parameter variables

    Returns:
        Updated edge mask of shape (batch_size, n, n) or (n, n)
    """
    if condition_mask.dim() == 1:
        condition_mask = condition_mask.unsqueeze(0)

    batch_size = condition_mask.shape[0]
    n = edge_mask.shape[0]
    device = edge_mask.device

    # Start with the base mask
    updated_masks = edge_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

    # Find which variables are conditioned on
    conditioned_data = condition_mask[:, n_params:].bool()

    # For posterior estimation: add edges from data to parameters
    # This allows information flow from observed data to latent parameters
    for b in range(batch_size):
        if conditioned_data[b].any():
            # Add edges: parameters can attend to conditioned data
            for i in range(n_params):
                for j in range(n_params, n):
                    if condition_mask[b, j] > 0:
                        # The parameter can attend to this conditioned data
                        updated_masks[b, i, j] = 1.0

    if batch_size == 1:
        return updated_masks.squeeze(0)

    return updated_masks


class MaskGenerator(nn.Module):
    """
    Generator for condition and edge masks during training and inference.

    Args:
        n_params: Number of parameter variables
        n_data: Number of data variables
        mask_type: Type of edge mask ("dense", "directed", "undirected")
        param_structure: Optional parameter dependency structure
        data_structure: Optional data dependency structure
        param_to_data: Optional parameter-to-data dependency structure
    """

    def __init__(
        self,
        n_params: int,
        n_data: int,
        mask_type: str = "dense",
        param_structure: Optional[torch.Tensor] = None,
        data_structure: Optional[torch.Tensor] = None,
        param_to_data: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_params = n_params
        self.n_data = n_data
        self.n_total = n_params + n_data
        self.mask_type = mask_type

        # Store dependency structures as buffers
        if param_structure is not None:
            self.register_buffer("param_structure", param_structure)
        else:
            self.param_structure = None

        if data_structure is not None:
            self.register_buffer("data_structure", data_structure)
        else:
            self.data_structure = None

        if param_to_data is not None:
            self.register_buffer("param_to_data", param_to_data)
        else:
            self.param_to_data = None

        # Pre-compute base edge mask
        self._create_base_mask()

    def _create_base_mask(self):
        """Create the base edge mask based on mask type."""
        if self.mask_type == "dense":
            base_mask = create_dense_mask(self.n_total)
        elif self.mask_type == "directed":
            base_mask = create_directed_mask(
                self.n_params,
                self.n_data,
                self.param_structure,
                self.data_structure,
                self.param_to_data,
            )
        elif self.mask_type == "undirected":
            base_mask = create_undirected_mask(
                self.n_params,
                self.n_data,
                self.param_structure,
                self.data_structure,
                self.param_to_data,
            )
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        self.register_buffer("base_edge_mask", base_mask)

    def sample_condition_mask(
        self,
        batch_size: int,
        mask_type: str = "mixed",
        device: torch.device = None,
    ) -> torch.Tensor:
        """Sample condition masks for training."""
        if device is None:
            device = self.base_edge_mask.device

        return sample_condition_mask(
            batch_size, self.n_params, self.n_data, mask_type, device=device
        )

    def get_edge_mask(
        self,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the edge mask, optionally updated for conditioning.

        Args:
            condition_mask: Optional condition mask to update edges for

        Returns:
            Edge mask of appropriate shape
        """
        if condition_mask is None or self.mask_type != "directed":
            return self.base_edge_mask

        return update_mask_for_conditioning(
            self.base_edge_mask, condition_mask, self.n_params
        )

    def get_posterior_mask(self, batch_size: int = 1) -> torch.Tensor:
        """Get condition mask for posterior estimation."""
        device = self.base_edge_mask.device
        mask = torch.zeros(batch_size, self.n_total, device=device)
        mask[:, self.n_params:] = 1.0
        return mask

    def get_likelihood_mask(self, batch_size: int = 1) -> torch.Tensor:
        """Get condition mask for likelihood estimation."""
        device = self.base_edge_mask.device
        mask = torch.zeros(batch_size, self.n_total, device=device)
        mask[:, :self.n_params] = 1.0
        return mask

    def get_joint_mask(self, batch_size: int = 1) -> torch.Tensor:
        """Get condition mask for joint distribution (no conditioning)."""
        device = self.base_edge_mask.device
        return torch.zeros(batch_size, self.n_total, device=device)
