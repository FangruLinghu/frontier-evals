## classifier/binary_classifier.py

```python
"""Binary classifier for source vs target domain discrimination.

This module implements the binary classifier p_φ used in the similarity-guided
diffusion model adaptation. The classifier distinguishes between source and target
domain images at noisy timestep t, providing gradients used in Equation 5:

    ||ε - ε_θ(x_t,t) - σ̂_t² γ ∇_{x_t} log p_φ(y=T|x_t)||²

The classifier is trained separately (not end-to-end with diffusion model) to
predict whether a noisy image x_t comes from source (y=0) or target (y=1)
distribution at timestep t.

Classes:
    SinusoidalPositionalEmbedding: Sinusoidal encoding for diffusion timesteps
    BinaryClassifier: Binary classifier p_φ for domain discrimination
"""

import torch
import torch.nn as nn
import math
from torch import Tensor


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep encoding.
    
    Implements sinusoidal positional encoding as described in the paper for
    encoding diffusion timesteps. The embedding uses sin and cos at different
    frequencies:
        pos_encoding(dim, pos) = sin(pos / 10000^(2i/dim)) for even dims
        pos_encoding(dim, pos) = cos(pos / 10000^(2i/dim)) for odd dims
    
    Attributes:
        embedding_dim: Dimension of timestep embeddings
        scale: Scaling factor for embeddings
    """
    
    def __init__(self, embedding_dim: int, scale: float = 1.0) -> None:
        """Initialize sinusoidal embedding layer for timestep encoding.
        
        Args:
            embedding_dim: Dimension of timestep embeddings
            scale: Scaling factor for embeddings (default: 1.0)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale
    
    def forward(self, timesteps: Tensor) -> Tensor:
        """Convert timesteps to sinusoidal embeddings.
        
        Uses sin and cos at different frequencies to encode positional information.
        Even dimensions use sin, odd dimensions use cos.
        
        Args:
            timesteps: [B] tensor of integer timesteps
        
        Returns:
            [B, embedding_dim] tensor with positional encodings using sin and cos
            at different frequencies
        """
        half_dim = self.embedding_dim // 2
        
        # Compute frequency components: 10000^(-2i/dim) for i in [0, half_dim-1]
        # Using log(10000) / half_dim for numerical stability
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -embeddings)
        
        # Apply scale factor
        embeddings = embeddings * self.scale
        
        # Compute positional encoding: [B, half_dim]
        # timesteps[:, None] broadcasts to [B, 1], embeddings[None, :] broadcasts to [1, half_dim]
        embeddings = timesteps.float()[:, None] * embeddings[None, :]
        
        # Concatenate sin and cos: [B, embedding_dim]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class BinaryClassifier(nn.Module):
    """Binary classifier for source vs target domain discrimination.
    
    This classifier p_φ predicts whether a noisy image x_t comes from the source
    domain (y=0) or target domain (y=1) at timestep t. It is used in the similarity-guided
    training loss (Equation 5) to compute gradients:
        ∇_{x_t} log p_φ(y=T|x_t)
    
    Architecture:
        Input (flattened x_t + timestep embed) -> FC -> SiLU -> FC -> SiLU -> FC -> Output (2 classes)
    
    The network is trained separately from the diffusion model to distinguish
    source vs target domain images at each noisy timestep t.
    
    Attributes:
        input_dim: Input feature dimension (flattened image size or 2D coordinates)
        hidden_dim: Hidden layer dimension
        embedding_dim: Timestep embedding dimension
        timestep_embed: Sinusoidal embedding for timesteps
        fc1: First fully connected layer: input_dim + embedding_dim -> hidden_dim
        fc2: Second fully connected layer: hidden_dim -> hidden_dim
        fc3: Output layer: hidden_dim -> 2 (binary classification: source vs target)
        act: SiLU activation function
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 64
    ) -> None:
        """Initialize the binary classifier network.
        
        Architecture: Input (flattened x_t + timestep embed) -> FC -> SiLU -> FC -> SiLU -> FC -> Output (2 classes)
        
        Args:
            input_dim: Input feature dimension (flattened image size or 2D coordinates)
            hidden_dim: Hidden layer dimension (default: 256)
            embedding_dim: Timestep embedding dimension (default: 64)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Sinusoidal embedding for timesteps
        self.timestep_embed = SinusoidalPositionalEmbedding(embedding_dim, scale=1.0)
        
        # First FC layer: input_dim + embedding_dim -> hidden_dim
        self.fc1 = nn.Linear(input_dim + embedding_dim, hidden_dim)
        
        # Second FC layer: hidden_dim -> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer: hidden_dim -> 2 (source vs target)
        self.fc3 = nn.Linear(hidden_dim, 2)
        
        # SiLU activation
        self.act = nn.SiLU()
    
    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Forward pass through classifier.
        
        Steps:
            1. Flatten x_t if not 1D
            2. Get sinusoidal timestep embeddings
            3. Concatenate flattened x_t and timestep embed
            4. Pass through FC layers with SiLU activations
            5. Return raw logits
        
        Args:
            x_t: Noisy image tensor [B, C, H, W] or [B, D]
            t: Timestep tensor [B]
        
        Returns:
            logits [B, 2] for binary classification (source=0, target=1)
        """
        # Step 1: Flatten x_t if not 1D
        if x_t.dim() > 2:
            x_t_flat = x_t.view(x_t.size(0), -1)
        else:
            x_t_flat = x_t
        
        # Ensure input_dim matches flattened dimension
        assert x_t_flat.size(1) == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got x_t_flat.size(1)={x_t_flat.size(1)}"
        )
        
        # Step 2: Get sinusoidal timestep embeddings [B, embedding_dim]
        t_emb = self.timestep_embed(t)
        
        # Step 3: Concatenate flattened x_t and timestep embed [B, input_dim + embedding_dim]
        combined = torch.cat([x_t_flat, t_emb], dim=-1)
        
        # Step 4: Pass through FC layers with SiLU activations
        h = self.act(self.fc1(combined))
        h = self.act(self.fc2(h))
        
        # Step 5: Return raw logits [B, 2]
        logits = self.fc3(h)
        
        return logits
    
    def compute_gradient(
        self,
        x_t: Tensor,
        t: Tensor,
        target_class: int = 1
    ) -> Tensor:
        """Compute gradient of log probability w.r.t. input x_t.
        
        Computes ∇_{x_t} log p_φ(y=target|x_t) for use in similarity-guided
        training loss (Equation 5):
            ||ε - ε_θ(x_t,t) - σ̂_t² γ ∇_{x_t} log p_φ(y=T|x_t)||²
        
        This gradient guides the diffusion model to generate samples that match
        the target domain distribution.
        
        Args:
            x_t: Noisy image tensor [B, C, H, W] or [B, D]
            t: Timestep tensor [B]
            target_class: Target class for gradient computation (1 for target, 0 for source)
                         Default: 1 (target domain)
        
        Returns:
            Gradient tensor same shape as x_t: ∇_{x_t} log p_φ(y=target|x_t)
        """
        # Save original shape for later
        original_shape = x_t.shape
        
        # Step 1: Clone x_t and set requires_grad for gradient computation
        x_t_grad = x_t.clone().detach().requires_grad_(True)
        
        # Step 2: Forward pass to get logits
        logits = self.forward(x_t_grad, t)
        
        # Step 3: Compute log_softmax and select target class log probability
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[:, target_class]
        
        # Step 4: Backward to get grad_x_t
        # Sum over batch dimension since we want gradient for each sample
        target_log_prob.sum().backward()
        
        # Step 5: Return gradient (same shape as original x_t)
        grad_x_t = x_t_grad.grad
        
        # Handle case where gradient is None (should not happen with requires_grad=True)
        if grad_x_t is None:
            raise RuntimeError(
                "Gradient computation failed. Ensure x_t requires grad and "
                "target_log_prob is not detached from computation graph."
            )
        
        return grad_x_t
    
    def get_probabilities(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Get softmax probabilities for source and target classes.
        
        Args:
            x_t: Noisy image tensor [B, C, H, W] or [B, D]
            t: Timestep tensor [B]
        
        Returns:
            probabilities [B, 2] where:
                [:, 0] = P(source|x_t)
                [:, 1] = P(target|x_t)
        """
        logits = self.forward(x_t, t)
        return torch.softmax(logits, dim=-1)