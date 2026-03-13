## model/transformer.py
"""
Transformer-based score estimator for Simformer.

Implements the core neural architecture: a transformer that processes sequences
of structured tokens (representing simulator parameters and data) to estimate
the score function of a diffusion model. The model incorporates diffusion time
via Fourier features and respects dependency structures through attention masks.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple, Any
import numpy as np

# Import Flax components
try:
    from flax import linen as nn
except ImportError:
    raise ImportError("Please install flax: pip install flax")

# Configuration defaults (will be overridden by config.yaml)
DEFAULT_CONFIG = {
    "model": {
        "token_dim": 50,
        "time_embedding_dim": 128,
        "num_layers": 6,
        "num_heads": 4,
        "attention_size": 10,  # head dimension
        "widening_factor": 3,
        "use_fourier_features": True,
        "fourier_scale": 1.0
    },
    "task_specific": {
        "Lotka_Volterra": {"num_layers": 8},
        "SIRD": {"num_layers": 8},
        "Hodgkin_Huxley": {"num_layers": 8}
    }
}


def load_config() -> dict:
    """
    Load configuration from global context or return defaults.
    In practice, this would integrate with Hydra.
    """
    return DEFAULT_CONFIG


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer with optional masking."""
    
    num_heads: int = 4
    head_dim: int = 10
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Apply multi-head attention to input sequence.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Attention mask of shape (seq_len, seq_len) or (batch, seq_len, seq_len)
                 Values should be 0 or -inf (or boolean)
        
        Returns:
            Output tensor of same shape as input
        """
        batch, seq_len, embed_dim = x.shape
        
        # Project to query, key, value
        qkv_proj = nn.Dense(
            features=3 * self.num_heads * self.head_dim,
            use_bias=self.use_bias
        )
        qkv = qkv_proj(x)  # (batch, seq_len, 3 * num_heads * head_dim)
        
        # Reshape to separate heads
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.transpose(2, 0, 3, 1, 4)  # Each: (batch, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scale = jnp.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) / scale  # (batch, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match number of heads
            if mask.ndim == 2:
                mask = mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]   # (batch, 1, seq_len, seq_len)
            # Convert boolean mask to float (-inf for masked positions)
            mask_float = jnp.where(mask, 0.0, -jnp.inf)
            attn_weights = attn_weights + mask_float
            
        # Apply softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Weighted sum using values
        attended = attn_weights @ v  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads and project back
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        output = nn.Dense(
            features=embed_dim,
            use_bias=self.use_bias
        )(attended)
        
        return output


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""
    
    dim: int
    num_heads: int
    mlp_ratio: float = 3.0
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Apply one transformer block with residual connections.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attn_mask: Attention mask passed to MultiHeadAttention
        
        Returns:
            Output tensor of same shape
        """
        # Self-attention sublayer with LayerNorm and residual connection
        h = nn.LayerNorm()(x)
        h = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.dim // self.num_heads
        )(h, mask=attn_mask)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=True)
        x = x + h
        
        # Feed-forward sublayer with LayerNorm and residual connection
        h = nn.LayerNorm()(x)
        hidden_dim = int(self.dim * self.mlp_ratio)
        h = nn.Dense(features=hidden_dim)(h)
        h = nn.gelu(h)
        h = nn.Dense(features=self.dim)(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=True)
        x = x + h
        
        return x


class Simformer(nn.Module):
    """Main Simformer model combining tokenizer, transformer, and time embedding."""
    
    config: Dict[str, Any]
    
    def setup(self):
        """Initialize all model components from config."""
        cfg = self.config["model"]
        task_cfg = self.config.get("task_specific", {})
        current_task = self.config.get("task_name", "default")
        
        # Get token dimensionality
        self.token_dim = cfg["token_dim"]  # Default 50
        
        # Determine number of layers (can be task-specific)
        if current_task in task_cfg and "num_layers" in task_cfg[current_task]:
            self.num_layers = task_cfg[current_task]["num_layers"]
        else:
            self.num_layers = cfg["num_layers"]  # Default 6
        
        # Time embedding settings
        self.time_embedding_dim = cfg["time_embedding_dim"]  # 128
        self.fourier_scale = cfg.get("fourier_scale", 1.0)
        
        # Build Gaussian Fourier Features matrix (fixed at init)
        fourier_key = jax.random.PRNGKey(42)
        self.B = jax.random.normal(fourier_key, (self.time_embedding_dim // 2, 1)) * self.fourier_scale
        
        # Projection layer for time embedding
        self.time_proj = nn.Dense(self.token_dim)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                dim=self.token_dim,
                num_heads=cfg["num_heads"],
                mlp_ratio=cfg["widening_factor"],
                name=f"block_{i}"
            ) for i in range(self.num_layers)
        ]
        
        # Final output head (identity or small projection)
        # Score output should have same dimensionality as input
        self.output_proj = nn.Dense(self.token_dim)
    
    def _fourier_features(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Gaussian Fourier Features for diffusion time t.
        
        Args:
            t: Scalar or batch of diffusion times, shape (B,) or ()
        
        Returns:
            Fourier feature embedding of shape (..., time_embedding_dim)
        """
        # Reshape t for broadcasting
        t = jnp.asarray(t)
        if t.ndim == 0:
            t = t[None]  # Add batch dim temporarily
        
        # Project t through B matrix
        proj = 2 * jnp.pi * jnp.dot(t.reshape(-1, 1), self.B.T)  # (B, time_embedding_dim//2)
        
        # Create [cos, sin] features
        features = jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)  # (B, time_embedding_dim)
        
        return features
    
    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cond_mask: Optional[jnp.ndarray] = None,
        attn_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Forward pass of Simformer model.
        
        Args:
            x: Input token sequence of shape (batch, seq_len, token_dim)
            t: Diffusion time scalar or vector of shape (batch,) or ()
            cond_mask: Binary mask indicating observed variables, shape (seq_len,) or (batch, seq_len)
                       Currently unused in forward pass but kept for interface consistency
            attn_mask: Attention mask for structural dependencies, shape (seq_len, seq_len)
        
        Returns:
            Estimated score of same shape as input (batch, seq_len, token_dim)
        """
        # Handle scalar t
        if isinstance(t, float) or t.ndim == 0:
            t = jnp.full((x.shape[0],), t)
        
        # Compute time embedding
        time_emb_fourier = self._fourier_features(t)  # (batch, time_embedding_dim)
        time_emb = self.time_proj(time_emb_fourier)  # (batch, token_dim)
        
        # Add time embedding to all tokens
        # Shape broadcasting: (batch, 1, token_dim)
        time_emb = time_emb.reshape(-1, 1, self.token_dim)
        h = x + time_emb
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            h = block(h, attn_mask=attn_mask)
        
        # Final projection
        score_pred = self.output_proj(h)
        
        return score_pred


# Example usage and testing
if __name__ == "__main__":
    import jax.random as random
    
    # Test basic functionality
    key = random.PRNGKey(0)
    keys = random.split(key, 2)
    
    # Create dummy config
    config = {
        "model": {
            "token_dim": 50,
            "time_embedding_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "attention_size": 10,
            "widening_factor": 3,
            "use_fourier_features": True,
            "fourier_scale": 1.0
        },
        "task_name": "test"
    }
    
    # Initialize model
    model = Simformer(config=config)
    params = model.init(keys[0],
                       x=random.normal(keys[1], (2, 10, 50)),
                       t=jnp.array([0.1, 0.9]))
    
    # Run forward pass
    x_test = random.normal(keys[1], (2, 10, 50))
    t_test = jnp.array([0.1, 0.9])
    attn_mask = jnp.ones((10, 10))  # Full attention
    score = model.apply(params, x_test, t_test, attn_mask=attn_mask)
    
    print(f"Input shape: {x_test.shape}")
    print(f"Score output shape: {score.shape}")
    assert score.shape == x_test.shape, "Output shape mismatch"
