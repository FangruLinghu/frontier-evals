## model/tokenizer.py
"""
Tokenizer for Simformer that encodes variables into structured tokens.

Implements the VariableTokenizer class that transforms simulator variables
(parameters or data points) into fixed-size token vectors suitable for input
to the transformer. Each token combines identifier, value, and condition state
information using learnable embeddings.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import required Flax components
try:
    from flax import linen as nn
except ImportError:
    raise ImportError("Please install flax: pip install flax")

# Configuration defaults (will be overridden by config.yaml)
DEFAULT_CONFIG = {
    "model": {
        "token_dim": 50,
        "time_embedding_dim": 128,
        "use_fourier_features": True,
        "fourier_scale": 1.0
    }
}


def load_config() -> dict:
    """
    Load configuration from global context or return defaults.
    In practice, this would integrate with Hydra.
    """
    return DEFAULT_CONFIG


class VariableTokenizer(nn.Module):
    """
    Tokenizer that converts simulator variables into embedded token vectors.
    
    Each variable is encoded as a combination of:
    - Identifier embedding (learnable vector for var_id)
    - Value projection (linear transformation of scalar value)
    - Condition state embedding (learnable vector for observed/latent status)
    
    For function-valued parameters (e.g., time-dependent), includes random Fourier
    features of the index (e.g., time stamp).
    """
    
    # Configuration parameters with defaults
    vocab_size: int = 1000
    token_dim: int = 50
    use_fourier_features: bool = True
    fourier_scale: float = 1.0
    fourier_features_dim: int = 32
    
    def setup(self):
        """Initialize all embedding layers."""
        # Identifier embedding table
        self.identifier_embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.token_dim
        )
        
        # Value projection layer
        self.value_projection = nn.Dense(self.token_dim)
        
        # Condition state embedding (0: latent, 1: conditioned)
        self.condition_embedding = nn.Embed(
            num_embeddings=2,
            features=self.token_dim
        )
        
        # Random Fourier Features projection matrix (fixed or learnable)
        if self.use_fourier_features:
            # Fixed random matrix for RFF
            B_key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
            self.B = jax.random.normal(B_key, (self.fourier_features_dim // 2, 1)) * self.fourier_scale
        else:
            self.B = None
    
    def _random_fourier_features(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Random Fourier Features for time/space index.
        
        Uses the formula:
        phi(t) = [cos(2πBt), sin(2πBt)]
        
        Args:
            t: Scalar time/space coordinate
            
        Returns:
            RFF vector of shape (fourier_features_dim,)
        """
        if not self.use_fourier_features:
            return jnp.zeros(self.fourier_features_dim)
            
        # Project time to higher dimension
        proj = 2 * jnp.pi * jnp.dot(self.B, t.reshape(-1))
        
        # Compute cos and sin components
        features = jnp.concatenate([
            jnp.cos(proj).squeeze(),
            jnp.sin(proj).squeeze()
        ])
        
        # Project back to token_dim via linear layer
        rff_proj = nn.Dense(self.token_dim)(features)
        return rff_proj
    
    def encode(
        self,
        var_id: str,
        value: float,
        time_idx: Optional[float] = None,
        is_conditioned: bool = False
    ) -> jnp.ndarray:
        """
        Encode a single variable into a token vector.
        
        Args:
            var_id: Unique identifier string for the variable
            value: Scalar value of the variable
            time_idx: Time/space index for functional parameters (optional)
            is_conditioned: Whether this variable is observed/fixed
            
        Returns:
            Token vector of shape (token_dim,)
        """
        # Convert inputs to JAX arrays
        value = jnp.array(value)
        is_cond = jnp.array(int(is_conditioned))
        
        # Get identifier embedding
        # In practice, var_id should be mapped to integer index via vocabulary
        # Here we use hash-based mapping for demonstration
        var_id_hash = hash(var_id) % self.vocab_size
        id_emb = self.identifier_embedding(jnp.array(var_id_hash))
        
        # Get value projection
        val_emb = self.value_projection(value.reshape(1, 1)).squeeze()
        
        # Get condition state embedding
        cond_emb = self.condition_embedding(is_cond)
        
        # Initialize base token as sum of components
        token = id_emb + val_emb + cond_emb
        
        # Add RFF contribution if time index provided
        if time_idx is not None:
            rff_emb = self._random_fourier_features(jnp.array(time_idx))
            token += rff_emb
            
        return token
    
    def batch_encode(
        self,
        variables: List[Dict]
    ) -> jnp.ndarray:
        """
        Encode a batch of variables into token sequence.
        
        Args:
            variables: List of dictionaries containing variable information
                     Each dict should have keys:
                     - 'var_id': str
                     - 'value': float
                     - 'time_idx': Optional[float]
                     - 'is_conditioned': bool
        
        Returns:
            Token sequence of shape (seq_len, token_dim)
        """
        # Process each variable through encode method
        tokens = []
        for var in variables:
            token = self.encode(
                var_id=var["var_id"],
                value=var["value"],
                time_idx=var.get("time_idx"),
                is_conditioned=var.get("is_conditioned", False)
            )
            tokens.append(token)
            
        # Stack into sequence
        return jnp.stack(tokens)
    
    @classmethod
    def create_from_config(cls, config: dict) -> 'VariableTokenizer':
        """
        Create tokenizer instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing model settings
            
        Returns:
            Initialized VariableTokenizer instance
        """
        model_cfg = config.get("model", {})
        
        return cls(
            vocab_size=model_cfg.get("vocab_size", 1000),
            token_dim=model_cfg.get("token_dim", 50),
            use_fourier_features=model_cfg.get("use_fourier_features", True),
            fourier_scale=model_cfg.get("fourier_scale", 1.0),
            fourier_features_dim=model_cfg.get("time_embedding_dim", 128) // 4 * 2  # Use portion for RFF
        )


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    key = jax.random.PRNGKey(0)
    
    # Create config
    config = load_config()
    
    # Initialize tokenizer
    tokenizer = VariableTokenizer.create_from_config(config)
    params = tokenizer.init(key, 
                           var_id="theta_0", 
                           value=1.0, 
                           time_idx=None, 
                           is_conditioned=False)
    
    # Test single encoding
    token = tokenizer.apply(params, 
                           var_id="theta_0", 
                           value=1.5, 
                           time_idx=None, 
                           is_conditioned=True)
    print(f"Single token shape: {token.shape}")
    
    # Test batch encoding
    variables = [
        {"var_id": "theta_0", "value": 1.5, "is_conditioned": True},
        {"var_id": "x_obs", "value": 42.0, "time_idx": 12.3, "is_conditioned": False},
        {"var_id": "theta_1", "value": 2.0, "is_conditioned": True}
    ]
    
    seq = tokenizer.apply(params, variables, method=VariableTokenizer.batch_encode)
    print(f"Batch token shape: {seq.shape}")
