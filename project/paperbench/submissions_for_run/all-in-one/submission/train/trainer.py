```python
## train/trainer.py
"""
Simformer training loop implementation.

Implements the SimformerTrainer class that orchestrates the full training pipeline:
- Loads configuration from Hydra
- Initializes model, optimizer, and data loader
- Executes denoising score matching with dynamic masking
- Handles early stopping and validation
- Logs metrics for reproducibility

The trainer supports both VESDE and VPSDE SDE types and dynamically samples condition masks
(joint, posterior, likelihood, random) as specified in the paper.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional
import optax
from flax.training import train_state
import numpy as np
from functools import partial

# Import required modules
try:
    from flax import linen as nn
except ImportError:
    raise ImportError("Please install flax: pip install flax")

# Import local modules
from model.transformer import Simformer
from model.diffusion import VESDE, VPSDE
from data.loader import SimulatorDataset
from utils.masks import AttentionMaskBuilder

# Configuration defaults (will be overridden by config.yaml)
DEFAULT_CONFIG = {
    "training": {
        "batch_size": 1000,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "early_stopping": True,
        "validation_split": 0.1,
        "max_epochs": None,
        "patience": 20,
        "min_delta": 1e-4
    },
    "model": {
        "token_dim": 50
    },
    "sde": {
        "type": "VESDE",
        "vesde": {
            "sigma_min": 0.0001,
            "sigma_max": 15.0,
            "t_min": 1e-5,
            "t_max": 1.0
        },
        "vpsde": {
            "beta_min": 0.01,
            "beta_max": 10.0,
            "t_min": 1e-5,
            "t_max": 1.0
        }
    },
    "masking": {
        "condition_mask_sampling": {
            "strategies": ["joint", "posterior", "likelihood", "random_p03", "random_p07"],
            "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2]
        },
        "attention_mask_type": "undirected",
        "dynamic_mask_adaptation": True
    },
    "task_name": "default"
}


def load_config() -> dict:
    """
    Load configuration from global context or return defaults.
    In practice, this would integrate with Hydra.
    """
    return DEFAULT_CONFIG


class SimformerTrainer:
    """
    Trainer class for Simformer model using denoising score matching.
    
    Implements the full training loop with dynamic masking of observed variables,
    supporting all conditional distributions (joint, posterior, likelihood).
    """

    def __init__(
        self,
        model: Simformer,
        dataset: SimulatorDataset,
        config: Dict[str, Any],
        key: Optional[jnp.ndarray] = None
    ):
        """
        Initialize the trainer with model, dataset, and configuration.
        
        Args:
            model: Simformer instance to be trained
            dataset: SimulatorDataset providing (theta, x) pairs
            config: Configuration dictionary containing hyperparameters
            key: PRNG key for initialization (optional)
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.key = key or jax.random.PRNGKey(0)
        
        # Extract training configuration
        train_cfg = self.config.get("training", {})
        self.batch_size = train_cfg.get("batch_size", 1000)
        self.learning_rate = train_cfg.get("learning_rate", 1e-3)
        self.early_stopping = train_cfg.get("early_stopping", True)
        self.validation_split = train_cfg.get("validation_split", 0.1)
        self.max_epochs = train_cfg.get("max_epochs", None)
        self.patience = train_cfg.get("patience", 20)
        self.min_delta = train_cfg.get("min_delta", 1e-4)
        
        # Extract SDE configuration
        sde_cfg = self.config.get("sde", {})
        self.sde_type = sde_cfg.get("type", "VESDE")
        
        if self.sde_type == "VESDE":
            vesde_cfg = sde_cfg.get("vesde", {})
            self.sde = VESDE(
                sigma_min=vesde_cfg.get("sigma_min", 0.0001),
                sigma_max=vesde_cfg.get("sigma_max", 15.0),
                t_min=vesde_cfg.get("t_min", 1e-5),
                t_max=vesde_cfg.get("t_max", 1.0)
            )
        else:  # VPSDE
            vpsde_cfg = sde_cfg.get("vpsde", {})
            self.sde = VPSDE(
                beta_min=vpsde_cfg.get("beta_min", 0.01),
                beta_max=vpsde_cfg.get("beta_max", 10.0),
                t_min=vpsde_cfg.get("t_min", 1e-5),
                t_max=vpsde_cfg.get("t_max", 1.0)
            )
        
        # Extract masking configuration
        mask_cfg = self.config.get("masking", {})
        self.attention_mask_type = mask_cfg.get("attention_mask_type", "undirected")
        self.dynamic_mask_adaptation = mask_cfg.get("dynamic_mask_adaptation", True)
        
        # Initialize model parameters
        self.key, init_key = jax.random.split(self.key)
        dummy_tokens = jnp.zeros((1, 10, self.config["model"].get("token_dim", 50)))
        dummy_t = jnp.array([0.5])
        self.params = self.model.init(init_key, dummy_tokens, dummy_t)
        
        # Initialize optimizer
        self.optimizer = optax.adam(self.learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=self.optimizer
        )
        
        # For early stopping
        self.best_loss = float('inf')
        self.wait = 0
        self.best_params = None
        
        # Task name for attention mask builder
        self.task_name = self.config.get("task_name", "default")

    def _sample_diffusion_time(self, key: jnp.ndarray) -> float:
        """
        Sample diffusion time t uniformly from [t_min, t_max].
        
        Args:
            key: PRNG key
            
        Returns:
            Diffusion time scalar
        """
        t_min = self.sde.t_min
        t_max = self.sde.t_max
        return jax.random.uniform(key, minval=t_min, maxval=t_max)

    def _build_attention_mask(
        self,
        seq_len: int,
        cond_mask: jnp.ndarray,
        key: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Build attention mask based on task structure and conditioning.
        
        Args:
            seq_len: Length of token sequence
            cond_mask: Binary mask indicating observed variables
            key: PRNG key
            
        Returns:
            Attention mask of shape (seq_len, seq_len)
        """
        # Create mask builder
        mask_builder = AttentionMaskBuilder(self.task_name, self.config)
        base_mask = mask_builder.get_base_mask()
        
        # Resize mask if necessary
        if base_mask.shape[0] != seq_len:
            # Simple resize by repeating/cropping - in practice should use proper indexing
            if base_mask.shape[0] > seq_len:
                base_mask = base_mask[:seq_len, :seq_len]
            else:
                # Pad with zeros
                pad_width = seq_len - base_mask.shape[0]
                base_mask = jnp.pad(base_mask, ((0, pad_width), (0, pad_width)), mode='constant')
        
        # Adapt for conditioning if enabled
        if self.dynamic_mask_adaptation:
            adapted_mask = mask_builder.adapt_for_conditioning(base_mask, cond_mask)
            return adapted_mask
        else:
            return base_mask

    @partial(jax.jit, static_argnums=(0,))
    def _compute_loss(
        self,
        params: Any,
        tokens: jnp.ndarray,
        x0: jnp.ndarray,
        xt: jnp.ndarray,
        t: float,
        cond_mask: jnp.ndarray,
        attn_mask: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute denoising score matching loss with masking.
        
        Args:
            params: Model parameters
            tokens: Input token sequence
            x0: Original data point
            xt: Noisy version at time t
            t: Diffusion time
            cond_mask: Binary mask indicating observed variables
            attn_mask: Attention mask for structural dependencies
            
        Returns:
            Loss value and auxiliary metrics
        """
        # Predict score using model
        score_pred = self.model.apply(params, tokens, t, cond_mask, attn_mask)
        
        # Compute true analytical score
        true_score = self.sde.score(x0, xt, t)
        
        # Apply masking: only unobserved variables contribute to loss
        mask = 1.0 - cond_mask.astype(jnp.float32)  # 1 for latent, 0 for conditioned
        masked_error = mask * (score_pred - true_score)
        
        # Compute MSE loss
        loss = jnp.mean(jnp.sum(masked_error ** 2, axis=-1))
        
        # Compute auxiliary metrics
        metrics = {
            'loss': loss,
            'mse_score': jnp.mean((score_pred - true_score) ** 2),
            'grad_norm': jnp.linalg.norm(jax.grad(lambda p: self._compute_loss(p, tokens, x0, xt, t, cond_mask, attn_mask)[0])(params))
        }
        
        return loss, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _update_step(
        self,
        state: train_state.TrainState,
        tokens: jnp.ndarray,
        x0: jnp.ndarray,
        xt: jnp.ndarray,
        t: float,
        cond_mask: jnp.ndarray,
        attn_mask: jnp.ndarray
    ) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
        """
        Perform one optimization step.
        
        Args:
            state: Current training state
            tokens: Input token sequence
            x0: Original data point
            xt: Noisy version at time t
            t: Diffusion time
            cond_mask: Binary mask indicating observed variables
            attn_mask: Attention mask for structural dependencies
            
        Returns:
            Updated state and metrics
        """
        grad_fn = jax.value_and_grad(self._compute_loss, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params, tokens, x0, xt, t, cond_mask, attn_mask)
        new_state = state.apply_gradients(grads=grads)
        return new_state, {**metrics, 'loss': loss}

    def step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one training step on a batch.
        
        Args:
            batch: Dictionary containing 'tokens', 'cond_mask', 'metadata'
            
        Returns:
            Dictionary of metrics (loss, etc.)
        """
        # Split PRNG key
        self.key, *step_keys = jax.random.split(self.key, 4)
        
        # Extract batch data
        tokens = batch['tokens']  # (B, L, D)
        metadata_list = batch['metadata']
        strategy = batch['strategy']
        
        # Reconstruct x0 from metadata (parameter and data values)
        B, L, D = tokens.shape
        x0_vals = []
        
        for i in range(B):
            sample_vals = []
            for var_info in metadata_list[i]:
                sample_vals.append(var_info['value'])
            x0_vals.append(sample_vals)
            
        x0 = jnp.array(x0_vals)  # (B, L)
        
        # Sample diffusion time for each example in batch
        t_vals = jnp.array([
            self._sample_diffusion_time(jax.random.fold_in(step_keys[0], i)) 
            for i in range(B)
        ])
        
        # Generate noisy versions xt for each t
        xt_vals = []
        for i in range(B):
            mean, std = self.sde.marginal_prob(x0[i], t_vals[i])
            noise = jax.random.normal(step_keys[1], x0[i].shape)
            xt_vals.append(mean + std * noise)
            
        xt = jnp.stack(xt_vals)