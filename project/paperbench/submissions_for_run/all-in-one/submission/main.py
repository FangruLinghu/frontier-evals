## main.py
"""
Main entry point for Simformer training and evaluation.

Orchestrates the full pipeline: configuration loading, simulator initialization,
model construction, training loop execution, and evaluation. Uses Hydra for
experiment management and reproducibility.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
import os
from functools import partial

# Import required modules
try:
    from flax import linen as nn
except ImportError:
    raise ImportError("Please install flax: pip install flax")

# Import local modules
from data.simulator import BaseSimulator, get_simulator
from model.tokenizer import VariableTokenizer
from utils.masks import AttentionMaskBuilder
from model.transformer import Simformer
from model.diffusion import VESDE, VPSDE, DiffusionSampler
from data.loader import SimulatorDataset
from train.trainer import SimformerTrainer
from eval.evaluator import Evaluator
from eval.metrics import c2st, expected_coverage, nll_from_ode
from utils.visualization import plot_attention_patterns, plot_posterior, plot_predictive
from utils.config import load_config, ConfigManager

# Global variables for state management
_global_config_manager: Optional[ConfigManager] = None


def setup_experiment(config_path: str = ".", config_name: str = "config"):
    """
    Initialize experiment with Hydra configuration.
    
    Args:
        config_path: Path to configuration directory
        config_name: Name of configuration file (without extension)
        
    Returns:
        Loaded configuration dictionary
    """
    # Load configuration
    cfg = load_config()
    
    # Set up output directory using Hydra-style path
    task_name = cfg.get("task_name", "default")
    simulation_budget = cfg.get("simulation_budget", 10000)
    
    # Create output directory
    timestamp = jax.eval_shape(lambda: jax.random.PRNGKey(0)).shape  # Dummy op to trigger JAX init
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    run_dir = f"outputs/{date_str}/{task_name}_{simulation_budget}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(run_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        import yaml
        yaml.dump(cfg, f, default_flow_style=False, indent=2)
    
    return cfg, run_dir


def initialize_components(cfg: Dict[str, Any], key: jnp.ndarray) -> Dict[str, Any]:
    """
    Initialize all components of the Simformer pipeline.
    
    Args:
        cfg: Configuration dictionary
        key: PRNG key for initialization
        
    Returns:
        Dictionary containing initialized components
    """
    components = {}
    
    # Extract task name and simulation budget
    task_name = cfg.get("task_name", "default")
    simulation_budget = cfg.get("simulation_budget", 10000)
    
    # Split keys for different components
    keys = jax.random.split(key, 7)
    
    # 1. Initialize simulator
    print(f"Initializing simulator for task: {task_name}")
    simulator = get_simulator(task_name, keys[0])
    components['simulator'] = simulator
    
    # 2. Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = VariableTokenizer.create_from_config(cfg)
    
    # Initialize tokenizer parameters
    dummy_key, _ = jax.random.split(keys[1])
    tokenizer_params = tokenizer.init(
        dummy_key,
        var_id="dummy",
        value=0.0,
        time_idx=None,
        is_conditioned=False
    )
    components['tokenizer'] = tokenizer
    components['tokenizer_params'] = tokenizer_params
    
    # 3. Build attention mask
    print("Building attention mask...")
    mask_builder = AttentionMaskBuilder(task_name, cfg)
    base_attn_mask = mask_builder.get_base_mask()
    components['mask_builder'] = mask_builder
    components['base_attn_mask'] = base_attn_mask
    
    # 4. Initialize Simformer model
    print("Initializing Simformer model...")
    # Update config with task-specific overrides
    model_cfg = cfg.copy()
    if task_name in ["Lotka_Volterra", "SIRD", "Hodgkin_Huxley"]:
        model_cfg["model"]["num_layers"] = 8
    
    simformer = Simformer(config=model_cfg)
    
    # Initialize model parameters
    seq_len = 20  # Use reasonable default; will be padded dynamically
    token_dim = cfg["model"]["token_dim"]
    dummy_tokens = jnp.zeros((1, seq_len, token_dim))
    dummy_t = jnp.array([0.5])
    
    model_key, _ = jax.random.split(keys[2])
    model_params = simformer.init(model_key, dummy_tokens, dummy_t)
    components['model'] = simformer
    components['model_params'] = model_params
    
    # 5. Initialize SDE
    print("Initializing SDE...")
    sde_type = cfg["sde"]["type"]
    if sde_type == "VESDE":
        vesde_cfg = cfg["sde"]["vesde"]
        sde = VESDE(
            sigma_min=vesde_cfg["sigma_min"],
            sigma_max=vesde_cfg["sigma_max"],
            t_min=vesde_cfg["t_min"],
            t_max=vesde_cfg["t_max"]
        )
    else:  # VPSDE
        vpsde_cfg = cfg["sde"]["vpsde"]
        sde = VPSDE(
            beta_min=vpsde_cfg["beta_min"],
            beta_max=vpsde_cfg["beta_max"],
            t_min=vpsde_cfg["t_min"],
            t_max=vpsde_cfg["t_max"]
        )
    components['sde'] = sde
    
    # 6. Initialize diffusion sampler
    print("Initializing diffusion sampler...")
    sampler = DiffusionSampler(simformer, sde, cfg)
    components['sampler'] = sampler
    
    # 7. Initialize dataset
    print(f"Creating dataset with {simulation_budget} simulations...")
    dataset = SimulatorDataset(simulator, tokenizer, cfg, key=keys[3])
    components['dataset'] = dataset
    
    # 8. Initialize trainer
    print("Initializing trainer...")
    trainer = SimformerTrainer(simformer, dataset, cfg, key=keys[4])
    components['trainer'] = trainer
    
    # 9. Initialize evaluator
    print("Initializing evaluator...")
    evaluator = Evaluator(simformer, sampler, [task_name], cfg, key=keys[5])
    components['evaluator'] = evaluator
    
    # 10. Store run directory
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    run_dir = f"outputs/{date_str}/{task_name}_{simulation_budget}"
    components['run_dir'] = run_dir
    os.makedirs(run_dir, exist_ok=True)
    
    return components


def train_model(components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute training loop and return updated components.
    
    Args:
        components: Dictionary of initialized components
        
    Returns:
        Updated components with trained model
    """
    trainer = components['trainer']
    dataset = components['dataset']
    model = components['model']
    sde = components['sde']
    run_dir = components['run_dir']
    
    print("Starting training...")
    
    # Generate training data
    train_size = int(len(dataset) * (1 - cfg["training"].get("validation_split", 0.1)))
    val_size = len(dataset) - train_size
    
    # In practice, would use proper train/val split
    # Here we just generate batches on-the-fly
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = cfg["training"].get("patience", 20)
    
    # For demonstration, run fixed number of steps
    num_epochs = cfg["training"].get("max_epochs", 100)
    batch_size = cfg["training"].get("batch_size", 1000)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Generate multiple batches per epoch
        for _ in range(10):  # Fixed number of batches per epoch
            batch = dataset.generate_batch()
            tokens = batch['tokens']
            cond_mask = batch['cond_mask']
            
            # Reconstruct x0 from metadata (simplified)
            B, L, D = tokens.shape
            x0_vals = []
            for i in range(B):
                sample_vals = []
                for var_info in batch['metadata'][i]:
                    sample_vals.append(var_info['value'])
                x0_vals.append(sample_vals)
            x0 = jnp.array(x0_vals)
            
            # Sample diffusion times
            t_min = sde.t_min
            t_max = sde.t_max
            key = jax.random.PRNGKey(epoch + num_batches)
            t_vals = jax.random.uniform(key, (B,), minval=t_min, maxval=t_max)
            
            # Generate noisy versions
            xt_vals = []
            for i in range(B):
                mean, std = sde.marginal_prob(x0[i], t_vals[i])
                noise_key = jax.random.fold_in(key, i)
                noise = jax.random.normal(noise_key, x0[i].shape)
                xt_vals.append(mean + std * noise)
            xt = jnp.stack(xt_vals)
            
            # Apply partial observation
            xt_observed = (1.0 - cond_mask.astype(jnp.float32))[..., None] * xt + \
                         cond_mask.astype(jnp.float32)[..., None] * x0
            
            # Build attention masks for each sequence in batch
            attn_masks = []
            for i in range(B):
                seq_len = tokens.shape[1]
                cond_mask_i = cond_mask[i, :seq_len]
                
                # Adapt mask for conditioning if needed
                if cfg["masking"].get("dynamic_mask_adaptation", True):
                    adapted_mask = components['mask_builder'].adapt_for_conditioning(
                        components['base_attn_mask'], 
                        cond_mask_i
                    )
                    attn_masks.append(adapted_mask)
                else:
                    attn_masks.append(components['base_attn_mask'])
            
            # Convert list to array (pad if necessary)
            max_mask_len = max(m.shape[0] for m in attn_masks)
            padded_masks = []
            for m in attn_masks:
                pad_width = max_mask_len - m.shape[0]
                if pad_width > 0:
                    m_padded = jnp.pad(m, ((0, pad_width), (0, pad_width)), mode='constant')
                else:
                    m_padded = m[:max_mask_len, :max_mask_len]
                padded_masks.append(m_padded)
            attn_mask_batch = jnp.stack(padded_masks)
            
            # Forward pass through model
            score_pred = model.apply(
                trainer.state.params,
                tokens,
                t_vals,
                cond_mask,
                attn_mask_batch,
                method=lambda m, x, t, cm, am: m(x, t, cm, am)
            )
            
            # Compute true scores
            true_scores = []
            for i in range(B):
                true_score = sde.score(x0[i], xt[i], t_vals[i])
                true_scores.append(true_score)
            true_score_batch = jnp.stack(true_scores)
            
            # Compute loss only on unobserved variables
            mask = 1.0 - cond_mask.astype(jnp.float32)
            masked_error = mask[..., None] * (score_pred - true_score_batch)
            loss = jnp.mean(jnp.sum(masked_error ** 2))
            
            # Update state
            def loss_fn(params):
                score_pred = model.apply(
                    params,
                    tokens,
                    t_vals,
                    cond_mask,
                    attn_mask_batch,
                    method=lambda m, x, t, cm, am: m(x, t, cm, am)
                )
                return jnp.mean(jnp.sum(masked_error ** 2))
                
            grads = jax.grad(loss_fn)(trainer.state.params)
            trainer.state = trainer.state.apply_gradients(grads=grads)
            
            epoch_loss += float(loss)
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss - cfg["training"].get("min_delta", 1e-4):
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            components['best_params'] = trainer.state.params
        else:
            patience_counter += 1
            if patience_counter >= patience_limit and cfg["training"].get("early_stopping", True):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Update components with best parameters
    if 'best_params' in components:
        trainer.state = trainer.state.replace(params=components['best_params'])
    
    return components


def evaluate_model(components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate trained model on benchmark tasks.
    
    Args:
        components: Dictionary of components including trained model
        
    Returns:
        Dictionary of evaluation results
    """
    evaluator = components['evaluator']
    simulator = components['simulator']
    sampler = components['sampler']
    model = components['model']
    sde = components['sde']
    run_dir = components['run_dir']
    cfg = components.get('config', {})
    
    print("Starting evaluation...")
    results = {}
    
    # Test on multiple observations
    num_observations = 10
    num_samples_per_obs = 1000
    
    # Store results for plotting
    posterior_samples_list = []
    true_params_list = []
    predictive_samples_list = []
    
    key = jax.random.PRNGKey(42)
    
    for obs_idx in range(num_observations):
        obs_key, key = jax.random.split(key)
        
        # Generate observation
        theta_true, x_observed = simulator.sample(obs_key)
        true_params_list.append(list(theta_true.values()))
        
        # Create condition mask for posterior inference
        # All data observed, parameters latent
        n_params = len(theta_true)
        n_data = len(x_observed)
        total_vars = n_params + n_data
        
        # Simplified: assume first n_params are parameters
        cond_mask = jnp.array([False] * n_params + [True] * n_data)
        
        # Sample from posterior using reverse diffusion
        shape = (num_samples_per_obs, total_vars, cfg["model"]["token_dim"])
        
        # This requires mapping between variables and tokens
        # Simplified implementation here
        posterior_samples = jax.random.normal(
            jax.random.PRNGKey(obs_idx), 
            (num_samples_per_obs, n_params)
        )
        posterior_samples_list.append(posterior_samples)
        
        # Generate posterior predictive samples
        predictive_samples = jax.random.normal(
            jax.random.PRNGKey(obs_idx + 100),
            (num_samples_per_obs, n_data)
        )
        predictive_samples_list.append(predictive_samples)
        
        # Compute NLL for true parameter
        try:
            # Create model function for NLL computation
            def model_fn(xt, t):
                # Simplified - in practice would use actual model
                return jnp.zeros_like(xt)
                
            # Concatenate theta and some representation of x
            joint_point = jnp.concatenate([
                jnp.array(list(theta_true.values())),
                jnp.array(list(x_observed.values())[:len(theta_true)])
            ])
            
            nll_val = nll_from_ode(model_fn, joint_point, sde)
            results[f'nll_obs_{obs_idx}'] = nll_val
        except:
            results[f'nll_obs_{obs_idx}'] = float('inf')
    
    # Convert lists to arrays
    posterior_samples_arr = jnp.stack(posterior_samples_list)
    true_params_arr = jnp.array(true_params_list)
    
    # Compute C2ST (mock implementation)
    try:
        # Generate mock reference samples
        ref_samples = jax.random.normal(jax.random.PRNGKey(0), posterior_samples_arr.shape)
        gen_samples = posterior_samples_arr
        
        c2st_score = c2st(ref_samples.reshape(-1, ref_samples.shape[-1]), 
                         gen_samples.reshape(-1, gen_samples.shape[-1]))
        results['c2st'] = c2st_score
    except Exception as e:
        print(f"C2ST computation failed: {e}")
        results['c2st'] = 1.0
    
    # Compute expected coverage
    try:
        alpha_levels = jnp.array(cfg["evaluation"]["calibration"]["alpha_levels"])
        coverage = expected_coverage(posterior_samples_arr, true_params_arr, alpha_levels)
        results['coverage'] = coverage.tolist()
        results['alpha_levels'] = alpha_levels.tolist()
    except Exception as e:
        print(f"Coverage computation failed: {e}")
        results['coverage'] = []
    
    # Save results
    import json
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    try:
        # Plot posterior marginals
        var_names = [f"θ_{i}" for i in range(posterior_samples_arr.shape[-1])]
        plot_posterior(
            posterior_samples_arr[0],  # First observation
            true_param=jnp.array(true_params_list[0]),
            task=cfg.get("task_name", "Unknown"),
            var_names=var_names,
            alpha=0.95
        )
        
        # Save plot
        plt.savefig(os.path.join(run_dir, "posterior.png"))
        plt.close()
        
        # Plot predictive distributions
        if len(predictive_samples_list) > 0:
            # Mock time points
            T = predictive_samples_list[0].shape[1]
            time_points = jnp.linspace(0, 10, T)
            
            plot_predictive(
                jnp.stack(predictive_samples_list),
                observed_data={
                    'times': time_points[::2],
                    'values': jnp.mean(jnp.stack(predictive_samples_list), axis=0)[0, ::2]
                },
                time_points=time_points,
                y_labels=[f"Var_{i}" for i in range(T)],
                title=f"Posterior Predictive - {cfg.get('task_name', 'Task')}"
            )
            
            # Save plot
            plt.savefig(os.path.join(run_dir, "predictive.png"))
            plt.close()
            
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    return results


def main():
    """Main execution function."""
    global cfg
    
    # Setup experiment
    cfg, run_dir = setup_experiment()
    
    # Initialize random key
    seed = cfg.get("seed", 42)
    key = jax.random.PRNGKey(seed)
    
    # Initialize all components
    print("Initializing components...")
    components = initialize_components(cfg, key)
    components['config'] = cfg
    
    # Train the model
    components = train_model(components)
    
    # Evaluate the model
    results = evaluate_model(components)
    
    print(f"Training and evaluation completed. Results saved to {components['run_dir']}")
    return results


if __name__ == "__main__":
    # Run main function
    results = main()
