## utils/visualization.py
"""
Visualization utilities for Simformer evaluation and analysis.

Implements plotting functions to reproduce key figures from the paper:
- Attention pattern evolution across transformer layers (Fig. A1a)
- Posterior marginals with ground truth overlay (Figs. 4b, 5a-b, 6a-b, 7b,e)
- Posterior predictive time series with uncertainty bands (Figs. 5a-b, 6a-b, 7c,f)

All plots follow the style of the original paper: clean, minimalistic, colorblind-friendly.
"""

import jax
import jax.numpy as jnp
from typing import List, Optional, Dict, Any, Sequence
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration defaults (will be overridden by config.yaml)
DEFAULT_CONFIG = {
    "evaluation": {
        "calibration": {
            "alpha_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    },
    "hydra": {
        "run": {
            "dir": "outputs/${now:%Y-%m-%d}/${task_name}_${simulation_budget}"
        }
    }
}


def load_config() -> dict:
    """
    Load configuration from global context or return defaults.
    In practice, this would integrate with Hydra.
    """
    return DEFAULT_CONFIG


def plot_attention_patterns(
    masks: List[jnp.ndarray],
    title: str = ""
) -> None:
    """
    Visualize the evolution of attention patterns across transformer layers.
    
    Reproduces Figure A1a from the paper, showing how the receptive field expands
    through multiple layers even when starting from a sparse base mask.
    
    Args:
        masks: List of attention matrices of shape (N, N), one per layer
        title: Optional title for the overall figure
        
    Example:
        plot_attention_patterns([mask_l1, mask_l2, mask_l3], "HMM Task - Layer-wise Attention")
    """
    num_layers = len(masks)
    if num_layers == 0:
        return
        
    # Determine grid layout (max 4 columns)
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_layers == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
        
    # Shared colormap and normalization
    vmax = max(jnp.max(mask) for mask in masks)
    
    for idx, (mask, ax) in enumerate(zip(masks, axes.flat)):
        # Convert to numpy for plotting
        mask_np = np.array(mask)
        
        # Create heatmap
        im = ax.imshow(mask_np, cmap='Blues', vmin=0, vmax=vmax, aspect='equal')
        
        # Add text annotations for clarity on small matrices
        if mask_np.shape[0] <= 10:
            for i in range(mask_np.shape[0]):
                for j in range(mask_np.shape[1]):
                    if mask_np[i, j] > 0:
                        ax.text(j, i, f'{mask_np[i,j]:.1f}', 
                               ha='center', va='center', color='white', fontsize=8)
        
        ax.set_title(f'Layer {idx + 1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        
        # Set ticks
        ax.set_xticks(range(mask_np.shape[1]))
        ax.set_yticks(range(mask_np.shape[0]))
        
    # Hide unused subplots
    for idx in range(num_layers, rows * cols):
        fig.delaxes(axes.flat[idx])
        
    # Add colorbar
    fig.colorbar(im, ax=axes, location='right', shrink=0.8, pad=0.05)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
        
    fig.tight_layout()
    plt.show()


def plot_posterior(
    samples: jnp.ndarray,
    true_param: Optional[jnp.ndarray] = None,
    task: str = "",
    var_names: Optional[List[str]] = None,
    alpha: float = 0.95
) -> None:
    """
    Plot pairwise marginal posteriors with optional true parameter overlay.
    
    Reproduces Figures 4b, 5a-b, 6a-b, 7b, 7e and Appendix A2 from the paper.
    Shows 2D kernel density estimates and 1D marginals in a pairplot-style grid.
    
    Args:
        samples: Array of shape (N_samples, D) containing posterior samples
        true_param: True parameter values of length D (optional)
        task: Name of the task (used for styling hints)
        var_names: Names for each parameter dimension (e.g., ['θ₁', 'θ₂'])
        alpha: Credible level for contour shading (default 95%)
        
    Example:
        plot_posterior(samples, true_param=[1.5, 2.0], task="Two_Moons", 
                      var_names=["θ₁", "θ₂"], alpha=0.95)
    """
    samples_np = np.array(samples)
    D = samples_np.shape[1]
    
    # Use provided variable names or generate defaults
    if var_names is None:
        var_names = [f"θ_{i}" for i in range(D)]
        
    # Create figure
    fig, axes = plt.subplots(D, D, figsize=(4 * D, 4 * D))
    if D == 1:
        axes = np.array([[axes]])
    elif D == 2:
        axes = axes.reshape(2, 2)
        
    # Global KDE for consistent contour levels
    kde = None
    try:
        kde = sns.utils.gaussian_kde(samples_np.T)
    except Exception:
        pass  # Fall back to histogram-only mode
        
    # Plot each subplot
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: marginal distribution
                sns.histplot(samples_np[:, i], ax=ax, kde=True, alpha=0.7, color='skyblue')
                ax.set_xlabel(var_names[i])
                ax.set_ylabel('Density' if j == 0 else '')
                
                # Mark true value if available
                if true_param is not None:
                    ax.axvline(true_param[i], color='red', linestyle='--', linewidth=2, label='True')
                    if j == 0:
                        ax.legend()
                        
            else:
                # Off-diagonal: 2D scatter or KDE
                if kde is not None:
                    # Create grid for KDE evaluation
                    x_min, x_max = samples_np[:, j].min(), samples_np[:, j].max()
                    y_min, y_max = samples_np[:, i].min(), samples_np[:, i].max()
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    
                    try:
                        # Evaluate KDE
                        kde_vals = kde(positions).reshape(xx.shape)
                        
                        # Find contour level corresponding to alpha
                        kde_flat = kde_vals.flatten()
                        kde_flat.sort()
                        cumsum = np.cumsum(kde_flat[::-1])
                        cumsum /= cumsum[-1]
                        threshold_idx = np.searchsorted(cumsum, 1 - alpha)
                        if threshold_idx < len(kde_flat):
                            threshold = kde_flat[::-1][threshold_idx]
                            
                            # Plot filled contours
                            ax.contourf(xx, yy, kde_vals, levels=[threshold, kde_vals.max()], 
                                      colors=['skyblue'], alpha=0.5)
                            ax.contour(xx, yy, kde_vals, levels=[threshold], 
                                     colors=['blue'], linewidths=2)
                    except:
                        # Fallback to scatter plot
                        ax.scatter(samples_np[:, j], samples_np[:, i], alpha=0.5, s=10, color='blue')
                else:
                    ax.scatter(samples_np[:, j], samples_np[:, i], alpha=0.5, s=10, color='blue')
                    
                # Mark true values if available
                if true_param is not None:
                    ax.plot(true_param[j], true_param[i], 'r*', markersize=15, label='True')
                    if i == D - 1 and j == 0:
                        ax.legend()
                        
                ax.set_xlabel(var_names[j])
                ax.set_ylabel(var_names[i])
                
    # Adjust spacing and labels
    fig.suptitle(f'Posterior Marginals - {task}' if task else 'Posterior Marginals', 
                 fontsize=16, y=0.98)
    fig.tight_layout()
    plt.show()


def plot_predictive(
    predictive_samples: jnp.ndarray,
    observed_data: Optional[Dict] = None,
    time_points: Optional[jnp.ndarray] = None,
    y_labels: Optional[List[str]] = None,
    title: str = ""
) -> None:
    """
    Plot posterior predictive distributions over time with uncertainty bands.
    
    Reproduces Figures 5a-b, 6a-b, 7c, 7f from the paper. Shows median prediction
    and credible intervals, overlaid with observed data points.
    
    Args:
        predictive_samples: Array of shape (N_samples, T, V) where T=time, V=variables
        observed_data: Dictionary with keys 'times', 'values', 'variables'
        time_points: Time indices for simulation output
        y_labels: Labels for each variable (e.g., ['Prey', 'Predator'])
        title: Plot title
        
    Example:
        plot_predictive(pred_samples, observed_data={'times': times, 'values': obs},
                       time_points=sim_times, y_labels=['S', 'I', 'R'], 
                       title='SIRD Model Predictions')
    """
    samples_np = np.array(predictive_samples)
    N_samples, T, V = samples_np.shape
    
    # Use default time points if not provided
    if time_points is None:
        time_points = np.arange(T)
    else:
        time_points = np.array(time_points)
        
    # Use default variable labels if not provided
    if y_labels is None:
        y_labels = [f"Var_{i}" for i in range(V)]
        
    # Create subplots for each variable
    fig, axes = plt.subplots(V, 1, figsize=(10, 3 * V), sharex=True)
    if V == 1:
        axes = [axes]
        
    # Compute statistics across samples
    median_pred = np.median(samples_np, axis=0)  # (T, V)
    lower_quantile = np.quantile(samples_np, 0.05, axis=0)  # 5th percentile
    upper_quantile = np.quantile(samples_np, 0.95, axis=0)  # 95th percentile
    
    # Plot each variable
    colors = sns.color_palette("husl", V)
    for v in range(V):
        ax = axes[v]
        
        # Plot uncertainty band
        ax.fill_between(time_points, lower_quantile[:, v], upper_quantile[:, v],
                       alpha=0.3, color=colors[v], label=f'{int((1-0.9)*100)}-{int(0.95*100)}% CI')
        
        # Plot median line
        ax.plot(time_points, median_pred[:, v], color=colors[v], linewidth=2, 
               label='Median Prediction')
        
        # Plot observed data if available
        if observed_data is not None:
            obs_times = np.array(observed_data.get('times', []))
            obs_vals = np.array(observed_data.get('values', []))
            
            # Filter observations for this variable if specified
            if 'variables' in observed_data:
                var_mask = np.array(observed_data['variables']) == v
                obs_times = obs_times[var_mask]
                obs_vals = obs_vals[var_mask]
                
            if len(obs_times) > 0:
                ax.scatter(obs_times, obs_vals, color='red', s=50, zorder=5, 
                          label='Observed Data', edgecolors='black')
        
        ax.set_ylabel(y_labels[v])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # Set common x-label
    axes[-1].set_xlabel('Time')
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
        
    fig.tight_layout()
    plt.show()

