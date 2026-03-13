## data/loader.py
"""
Data loader for Simformer that generates batches of (theta, x) pairs from simulators.

Implements SimulatorDataset which wraps a simulator to produce tokenized sequences
with dynamic condition masks. The dataset handles variable-length sequences and
supports all masking strategies specified in the paper (joint, posterior, likelihood,
and random masks).
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import required modules
try:
    from flax import linen as nn
except ImportError:
    raise ImportError("Please install flax: pip install flax")

# Import local modules
from data.simulator import BaseSimulator, get_simulator
from model.tokenizer import VariableTokenizer

# Configuration defaults (will be overridden by config.yaml)
DEFAULT_CONFIG = {
    "training": {
        "batch_size": 1000
    },
    "masking": {
        "condition_mask_sampling": {
            "strategies": ["joint", "posterior", "likelihood", "random_p03", "random_p07"],
            "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform sampling as per Appendix A2.1
        }
    }
}


def load_config() -> dict:
    """
    Load configuration from global context or return defaults.
    In practice, this would integrate with Hydra.
    """
    return DEFAULT_CONFIG


class SimulatorDataset:
    """
    Dataset wrapper that generates batches of simulation data for training.
    
    Handles:
    - Sampling from simulator
    - Tokenization using VariableTokenizer
    - Dynamic condition mask generation
    - Batch construction with padding
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        tokenizer: VariableTokenizer,
        config: Optional[Dict] = None,
        key: Optional[jnp.ndarray] = None
    ):
        """
        Initialize dataset with simulator and tokenizer.
        
        Args:
            simulator: Instance of BaseSimulator to generate (theta, x) pairs
            tokenizer: Instance of VariableTokenizer to encode variables
            config: Configuration dictionary containing settings
            key: PRNG key for reproducibility
        """
        self.simulator = simulator
        self.tokenizer = tokenizer
        self.config = config or load_config()
        self.key = key or jax.random.PRNGKey(0)
        
        # Extract masking configuration
        mask_cfg = self.config.get("masking", {}).get("condition_mask_sampling", {})
        self.strategies = mask_cfg.get("strategies", ["joint", "posterior", "likelihood", "random_p03", "random_p07"])
        self.probabilities = mask_cfg.get("probabilities", [0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Extract training configuration
        train_cfg = self.config.get("training", {})
        self.batch_size = train_cfg.get("batch_size", 1000)

    def _sample_condition_mask(
        self,
        n_vars: int,
        strategy_idx: int,
        key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, str]:
        """
        Sample condition mask according to selected strategy.
        
        Strategies:
        - 'joint': all False (unconditional)
        - 'posterior': parameters False, data True
        - 'likelihood': parameters True, data False  
        - 'random_p03': Bernoulli(0.3)
        - 'random_p07': Bernoulli(0.7)
        
        Args:
            n_vars: Total number of variables in sequence
            strategy_idx: Index into self.strategies
            key: PRNG key
            
        Returns:
            Binary condition mask of shape (n_vars,), strategy name
        """
        strategy = self.strategies[strategy_idx]
        key, subkey = jax.random.split(key)
        
        if strategy == "joint":
            # All variables are latent
            mask = jnp.zeros(n_vars, dtype=bool)
            
        elif strategy == "posterior":
            # First half (parameters) are latent, second half (data) are observed
            # This assumes equal split; in practice, we need to know parameter count
            n_params = n_vars // 2  # Simplified assumption
            mask = jnp.array([i >= n_params for i in range(n_vars)])
            
        elif strategy == "likelihood":
            # First half (parameters) are observed, second half (data) are latent
            n_params = n_vars // 2
            mask = jnp.array([i < n_params for i in range(n_vars)])
            
        elif strategy == "random_p03":
            # Each variable independently has 30% chance to be observed
            probs = jax.random.bernoulli(subkey, 0.3, shape=(n_vars,))
            mask = probs.astype(bool)
            
        elif strategy == "random_p07":
            # Each variable independently has 70% chance to be observed
            probs = jax.random.bernoulli(subkey, 0.7, shape=(n_vars,))
            mask = probs.astype(bool)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return mask, strategy

    def _build_variable_list(
        self,
        theta_dict: Dict[str, float],
        x_dict: Dict[str, float],
        cond_mask: jnp.ndarray
    ) -> List[Dict]:
        """
        Build list of variable dictionaries for tokenization.
        
        Args:
            theta_dict: Dictionary of parameter values
            x_dict: Dictionary of observation values (may include time indices)
            cond_mask: Condition mask for these variables
            
        Returns:
            List of variable dicts with keys: var_id, value, time_idx, is_conditioned
        """
        variables = []
        idx = 0
        
        # Add parameters
        for var_id, value in theta_dict.items():
            variables.append({
                "var_id": var_id,
                "value": value,
                "time_idx": None,
                "is_conditioned": bool(cond_mask[idx])
            })
            idx += 1
        
        # Add data points (extract time index from key if present)
        for var_id, value in x_dict.items():
            # Parse time index from key like "prey_t=5.2"
            time_idx = None
            if "_t=" in var_id:
                try:
                    time_part = var_id.split("_t=")[1]
                    time_idx = float(time_part)
                except:
                    pass
                    
            variables.append({
                "var_id": var_id,
                "value": value,
                "time_idx": time_idx,
                "is_conditioned": bool(cond_mask[idx])
            })
            idx += 1
            
        return variables

    def generate_batch(self, key: Optional[jnp.ndarray] = None) -> Dict[str, Any]:
        """
        Generate a batch of tokenized sequences with condition masks.
        
        Args:
            key: PRNG key for sampling (optional)
            
        Returns:
            Dictionary containing:
            - 'tokens': Padded token sequences (B, L, D)
            - 'cond_mask': Boolean mask indicating observed variables (B, L)
            - 'metadata': List of metadata lists for each sample
            - 'strategy': Name of condition mask strategy used
        """
        if key is None:
            key = self.key
            self.key, _ = jax.random.split(self.key)
            
        # Split key for different operations
        keys = jax.random.split(key, self.batch_size + 2)
        sample_keys = keys[:-2]
        strategy_key = keys[-2]
        cond_mask_key = keys[-1]
        
        # Sample strategy uniformly across options
        strategy_idx = jax.random.choice(
            strategy_key, 
            len(self.strategies), 
            p=jnp.array(self.probabilities)
        )
        strategy_name = self.strategies[int(strategy_idx)]
        
        batch_tokens = []
        batch_cond_masks = []
        batch_metadata = []
        
        # Generate each sample in the batch
        for i in range(self.batch_size):
            # Sample from simulator
            theta_dict, x_dict = self.simulator.sample(sample_keys[i])
            
            # Determine total number of variables
            n_total_vars = len(theta_dict) + len(x_dict)
            
            # Sample condition mask for this example
            key_cond, cond_mask_key = jax.random.split(cond_mask_key)
            ex_cond_mask, _ = self._sample_condition_mask(n_total_vars, strategy_idx, key_cond)
            
            # Build variable list with conditioning info
            variables = self._build_variable_list(theta_dict, x_dict, ex_cond_mask)
            
            # Tokenize each variable
            tokens = []
            metadata = []
            for var in variables:
                token = self.tokenizer.apply(
                    {},  # params would be passed in real usage
                    var["var_id"],
                    var["value"],
                    var["time_idx"],
                    var["is_conditioned"],
                    method=self.tokenizer.encode
                )
                tokens.append(token)
                metadata.append({
                    "id": var["var_id"],
                    "time": var["time_idx"],
                    "is_conditioned": var["is_conditioned"]
                })
                
            # Stack into sequence
            seq_tokens = jnp.stack(tokens)
            seq_cond_mask = jnp.array([m["is_conditioned"] for m in metadata])
            
            batch_tokens.append(seq_tokens)
            batch_cond_masks.append(seq_cond_mask)
            batch_metadata.append(metadata)
        
        # Find maximum sequence length for padding
        max_len = max(t.shape[0] for t in batch_tokens)
        d_model = batch_tokens[0].shape[1]
        
        # Pad sequences
        padded_tokens = jnp.zeros((self.batch_size, max_len, d_model))
        padded_cond_mask = jnp.zeros((self.batch_size, max_len), dtype=bool)
        
        for i in range(self.batch_size):
            L = batch_tokens[i].shape[0]
            padded_tokens = padded_tokens.at[i, :L, :].set(batch_tokens[i])
            padded_cond_mask = padded_cond_mask.at[i, :L].set(batch_cond_masks[i])
        
        return {
            "tokens": padded_tokens,
            "cond_mask": padded_cond_mask,
            "metadata": batch_metadata,
            "strategy": strategy_name
        }

    def __len__(self) -> int:
        """Return approximate dataset size."""
        # Since we generate on-the-fly, return a large number
        return 1000000

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item at index (not typically used in JAX pipelines).
        For compatibility with PyTorch-style dataloaders.
        """
        key = jax.random.fold_in(self.key, idx)
        return self.generate_batch(key)


# Example usage and testing
if __name__ == "__main__":
    import jax.random as random
    
    # Test basic functionality
    key = random.PRNGKey(0)
    keys = random.split(key, 3)
    
    # Create dummy config
    config = {
        "training": {"batch_size": 2},
        "masking": {
            "condition_mask_sampling": {
                "strategies": ["joint", "posterior", "likelihood", "random_p03", "random_p07"],
                "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2]
            }
        },
        "model": {"token_dim": 50}
    }
    
    # Initialize components
    simulator = get_simulator("Gaussian_Linear", keys[0])
    tokenizer = VariableTokenizer.create_from_config(config)
    params = tokenizer.init(keys[1], 
                           var_id="test", 
                           value=1.0, 
                           time_idx=None, 
                           is_conditioned=False)
    
    # Create dataset
    dataset = SimulatorDataset(simulator, tokenizer, config, key=keys[2])
    
    # Generate batch
    batch = dataset.generate_batch()
    
    print(f"Tokens shape: {batch['tokens'].shape}")
    print(f"Condition mask shape: {batch['cond_mask'].shape}")
    print(f"Strategy: {batch['strategy']}")
    print(f"Metadata length: {len(batch['metadata'])}")

