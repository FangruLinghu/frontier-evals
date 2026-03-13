## utils/masks.py
"""
Attention mask builder for Simformer.

Implements task-specific attention masks that encode dependency structures
between parameters and data variables. Supports both undirected (symmetric)
and directed (causal) graphs, with optional dynamic adaptation during conditioning
using the algorithm from Webb et al. (2018).

The masks control which variables can attend to each other in the transformer,
enabling the model to exploit known independencies in the generative process.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple
import numpy as np

# Configuration defaults (will be overridden by config.yaml)
DEFAULT_CONFIG = {
    "masking": {
        "attention_mask_type": "undirected",  # or 'directed'
        "dynamic_mask_adaptation": True
    }
}


def load_config() -> Dict:
    """
    Load configuration from global context or return defaults.
    In practice, this would integrate with Hydra.
    """
    return DEFAULT_CONFIG


class AttentionMaskBuilder:
    """
    Builds and adapts attention masks for Simformer based on task structure.
    
    The attention mask encodes which variables can attend to each other
    in the transformer, reflecting the conditional independence structure
    of the simulator's generative model.
    """

    def __init__(self, task: str, config: Optional[Dict] = None):
        """
        Initialize mask builder for a specific task.
        
        Args:
            task: Name of the task (e.g., 'HMM', 'SLCP')
            config: Configuration dictionary containing masking settings
        """
        self.task = task
        self.config = config or load_config()
        self.mask_type = self.config["masking"].get("attention_mask_type", "undirected")
        self.dynamic_adaptation = self.config["masking"].get("dynamic_mask_adaptation", True)
        self.base_mask = self._build_base_mask()

    def get_base_mask(self) -> jnp.ndarray:
        """
        Get the base structural attention mask for this task.
        
        This mask represents the assumed dependency structure of the generative model.
        For undirected mode, it returns a symmetric version.
        
        Returns:
            Base attention mask of shape (seq_len, seq_len)
        """
        if self.mask_type == "undirected":
            # Symmetrize the base mask
            return jnp.logical_or(self.base_mask, self.base_mask.T).astype(jnp.float32)
        else:
            return self.base_mask

    def adapt_for_conditioning(
        self, 
        base_mask: jnp.ndarray, 
        cond_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Adapt the attention mask to remain faithful under conditioning.
        
        Implements the algorithm from Webb et al. (2018): when conditioning on a variable,
        add edges between all its parents to account for explaining-away effects.
        
        This ensures that the transformer can represent dependencies induced by observation.
        
        Args:
            base_mask: Original adjacency matrix of shape (N, N)
            cond_mask: Binary vector indicating observed variables, shape (N,)
            
        Returns:
            Adapted attention mask with additional edges added
        """
        if not self.dynamic_adaptation:
            return base_mask
            
        # Convert to boolean for logical operations
        adj = base_mask.astype(bool)
        cond_vec = cond_mask.astype(bool)
        n_vars = len(adj)
        
        # Build parent sets: parents[j] = {i | i -> j}
        parents = [set() for _ in range(n_vars)]
        for i in range(n_vars):
            for j in range(n_vars):
                if adj[i, j]:  # i -> j
                    parents[j].add(i)
        
        # Create new adjacency matrix
        new_adj = adj.copy()
        
        # For each conditioned node, fully connect its parents
        for node in range(n_vars):
            if cond_vec[node]:
                parent_list = list(parents[node])
                # Fully connect all pairs of parents
                for i_idx in range(len(parent_list)):
                    for j_idx in range(i_idx + 1, len(parent_list)):
                        pa_i = parent_list[i_idx]
                        pa_j = parent_list[j_idx]
                        new_adj = new_adj.at[pa_i, pa_j].set(True)
                        new_adj = new_adj.at[pa_j, pa_i].set(True)
        
        return new_adj.astype(jnp.float32)

    def _build_base_mask(self) -> jnp.ndarray:
        """Build the base structural mask based on task."""
        if self.task == "Gaussian_Linear":
            return self._gaussian_linear_mask()
        elif self.task == "Gaussian_Mixture":
            return self._gaussian_mixture_mask()
        elif self.task == "Two_Moons":
            return self._two_moons_mask()
        elif self.task == "SLCP":
            return self._slcp_mask()
        elif self.task == "Tree":
            return self._tree_mask()
        elif self.task == "HMM":
            return self._hmm_mask()
        elif self.task == "Lotka_Volterra":
            return self._lotka_volterra_mask()
        elif self.task == "SIRD":
            return self._sird_mask()
        elif self.task == "Hodgkin_Huxley":
            return self._hh_mask()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _gaussian_linear_mask(self) -> jnp.ndarray:
        """Base mask for Gaussian Linear task: θ_i → x_i only."""
        # Each parameter directly influences corresponding data point
        # No intra-θ or intra-x connections
        n_params = 10
        n_data = 10
        total_len = n_params + n_data
        
        mask = jnp.zeros((total_len, total_len))
        
        # θ_i → x_i (bipartite connections)
        for i in range(n_params):
            mask = mask.at[i, n_params + i].set(1.0)  # θ_i → x_i
            
        return mask

    def _gaussian_mixture_mask(self) -> jnp.ndarray:
        """Base mask for Gaussian Mixture: θ → x_0, x_1."""
        # Single parameter influences both data points
        mask = jnp.zeros((3, 3))  # θ, x0, x1
        
        # θ → x0, θ → x1
        mask = mask.at[0, 1].set(1.0)
        mask = mask.at[0, 2].set(1.0)
        
        return mask

    def _two_moons_mask(self) -> jnp.ndarray:
        """Base mask for Two Moons: θ_0,θ_1 → x_0,x_1."""
        # Parameters influence data, no internal connections
        mask = jnp.zeros((4, 4))  # θ0,θ1,x0,x1
        
        # θ → x connections
        for i in range(2):
            for j in range(2):
                mask = mask.at[i, 2 + j].set(1.0)
                
        return mask

    def _slcp_mask(self) -> jnp.ndarray:
        """Base mask for SLCP: all θ → all x_i."""
        # Five parameters influence eight data points (4 observations × 2 dims)
        n_params = 5
        n_data = 8
        total_len = n_params + n_data
        
        mask = jnp.zeros((total_len, total_len))
        
        # All parameters → all data points
        for i in range(n_params):
            for j in range(n_data):
                mask = mask.at[i, n_params + j].set(1.0)
                
        return mask

    def _tree_mask(self) -> jnp.ndarray:
        """Base mask for Tree task: hierarchical dependencies."""
        # Structure: θ0 → θ1, θ0 → θ2
        #           θ1 → x1, x2; θ2 → x3, x4
        total_len = 7  # θ0,θ1,θ2,x1,x2,x3,x4
        mask = jnp.zeros((total_len, total_len))
        
        # θ0 → θ1, θ2
        mask = mask.at[0, 1].set(1.0)
        mask = mask.at[0, 2].set(1.0)
        
        # θ1 → x1, x2
        mask = mask.at[1, 3].set(1.0)
        mask = mask.at[1, 4].set(1.0)
        
        # θ2 → x3, x4
        mask = mask.at[2, 5].set(1.0)
        mask = mask.at[2, 6].set(1.0)
        
        return mask

    def _hmm_mask(self) -> jnp.ndarray:
        """Base mask for HMM: Markov chain + emissions."""
        n_vars = 10  # θ0..θ9, x0..x9
        total_len = 2 * n_vars
        mask = jnp.zeros((total_len, total_len))
        
        # Transition: θ_i → θ_{i+1}
        for i in range(n_vars - 1):
            mask = mask.at[i, i + 1].set(1.0)
            
        # Emission: θ_i → x_i
        for i in range(n_vars):
            mask = mask.at[i, n_vars + i].set(1.0)
            
        return mask

    def _lotka_volterra_mask(self) -> jnp.ndarray:
        """Base mask for Lotka-Volterra: global params → time-series."""
        # 4 parameters → multiple prey/predator observations at different times
        # We assume maximum connectivity from params to data
        # Data points may have temporal dependencies
        n_params = 4
        # Variable length - we'll create a template for max expected sequence
        max_data_points = 20  # 10 prey + 10 predator observations
        total_len = n_params + max_data_points
        
        mask = jnp.zeros((total_len, total_len))
        
        # All parameters → all data points
        for i in range(n_params):
            for j in range(max_data_points):
                mask = mask.at[i, n_params + j].set(1.0)
                
        # Temporal dependencies among data points (optional)
        # For simplicity, assume no direct x_t → x_{t'} connections
        # These could be added if needed
        
        return mask

    def _sird_mask(self) -> jnp.ndarray:
        """Base mask for SIRD with time-dependent β(t)."""
        # Global parameters γ, μ → all states
        # β(t_i) → states at nearby times
        # Observations depend on local states
        n_global_params = 2  # γ, μ
        n_time_points = 10   # Discretized time points for β(t)
        n_states_per_time = 4  # S,I,R,D at each time
        n_obs_per_time = 4     # One observation per state per time
        
        total_params = n_global_params + n_time_points
        total_data = n_time_points * (n_states_per_time + n_obs_per_time)
        total_len = total_params + total_data
        
        mask = jnp.zeros((total_len, total_len))
        
        # Global parameters → all states and observations
        for p_idx in range(n_global_params):
            for t in range(n_time_points):
                # States at time t
                for s in range(n_states_per_time):
                    state_idx = total_params + t * (n_states_per_time + n_obs_per_time) + s
                    mask = mask.at[p_idx, state_idx].set(1.0)
                    
                # Observations at time t
                for o in range(n_obs_per_time):
                    obs_idx = total_params + t * (n_states_per_time + n_obs_per_time) + n_states_per_time + o
                    mask = mask.at[p_idx, obs_idx].set(1.0)
        
        # β(t_i) → states and observations at time t_j (local influence)
        for beta_idx in range(n_global_params, n_global_params + n_time_points):
            t_i = beta_idx - n_global_params
            for t_j in range(n_time_points):
                # Allow influence within reasonable temporal window
                if abs(t_i - t_j) <= 2:  # Local temporal support
                    # States at t_j
                    for s in range(n_states_per_time):
                        state_idx = total_params + t_j * (n_states_per_time + n_obs_per_time) + s
                        mask = mask.at[beta_idx, state_idx].set(1.0)
                        
                    # Observations at t_j
                    for o in range(n_obs_per_time):
                        obs_idx = total_params + t_j * (n_states_per_time + n_obs_per_time) + n_states_per_time + o
                        mask = mask.at[beta_idx, obs_idx].set(1.0)
        
        return mask

    def _hh_mask(self) -> jnp.ndarray:
        """Base mask for Hodgkin-Huxley: parameters → summary statistics."""
        # 7 parameters → 6 summary statistics + energy
        n_params = 7
        n_summary_stats = 6  # voltage_mean, voltage_std, spike_count, isi_mean, isi_cv, energy
        total_len = n_params + n_summary_stats
        
        mask = jnp.zeros((total_len, total_len))
        
        # All parameters → all summary statistics
        for i in range(n_params):
            for j in range(n_summary_stats):
                mask = mask.at[i, n_params + j].set(1.0)
                
        return mask
