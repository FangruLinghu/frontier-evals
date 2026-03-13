## eval/evaluator.py
"""
Evaluator class for Simformer that computes key metrics from the paper.

Implements quantitative and qualitative evaluation of the trained Simformer model
using:
- Classifier Two-Sample Test (C2ST) for posterior fidelity
- Expected coverage analysis for calibration assessment
- Negative log-likelihood estimation via probability flow ODE

The evaluator supports arbitrary conditional distributions and guided diffusion
for interval constraints as demonstrated in Figures 4-7 and Appendix.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from functools import partial

# Import required modules
try:
    from flax import linen as nn
except ImportError:
    raise ImportError("Please install flax: pip install flax")

# Import local modules
from model.diffusion import DiffusionSampler, VESDE, VPSDE
from data.simulator import BaseSimulator, get_simulator
from eval.metrics import c2st as compute_c2st, expected_coverage as compute_expected_coverage, nll_from_ode

# Configuration defaults (will be overridden by config.yaml)
DEFAULT_CONFIG = {
    "evaluation": {
        "c2st": {
            "n_samples": 1000,
            "n_trials": 10,
            "classifier": "logistic_regression"
        },
        "calibration": {
            "alpha_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
        "nll": {
            "method": "probability_flow_ode",
            "solver": "dopri5",
            "rtol": 1e-5,
            "atol": 1e-5,
            "timesteps": 1000
        }
    },
    "sampling": {
        "num_steps": 500
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
    "task_name": "default"
}


def load_config() -> dict:
    """
    Load configuration from global context or return defaults.
    In practice, this would integrate with Hydra.
    """
    return DEFAULT_CONFIG


class Evaluator:
    """
    Class to evaluate the performance of Simformer on various tasks and metrics.
    
    Supports computing C2ST, expected coverage, NLL, and generating posterior predictives.
    Can handle arbitrary conditionals and guided diffusion for interval constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        sampler: DiffusionSampler,
        test_tasks: List[str],
        config: Optional[Dict] = None,
        key: Optional[jnp.ndarray] = None
    ):
        """
        Initialize evaluator with trained model and sampling infrastructure.
        
        Args:
            model: Trained Simformer model (Flax module)
            sampler: DiffusionSampler instance configured with SDE and model
            test_tasks: List of task names to evaluate on
            config: Configuration dictionary containing evaluation settings
            key: PRNG key for reproducibility
        """
        self.model = model
        self.sampler = sampler
        self.test_tasks = test_tasks
        self.config = config or load_config()
        self.key = key or jax.random.PRNGKey(0)
        
        # Extract evaluation parameters
        eval_cfg = self.config.get("evaluation", {})
        self.c2st_cfg = eval_cfg.get("c2st", {})
        self.calib_cfg = eval_cfg.get("calibration", {})
        self.nll_cfg = eval_cfg.get("nll", {})
        
        # Extract sampling parameters
        sampling_cfg = self.config.get("sampling", {})
        self.num_steps = sampling_cfg.get("num_steps", 500)
        
        # Create simulator instances for each task
        self.simulators = {}
        for task in test_tasks:
            # Clean task name (replace underscores)
            clean_task = task.replace("_", " ")
            self.simulators[task] = get_simulator(task, self.key)
            self.key, _ = jax.random.split(self.key)

    def compute_c2st(
        self,
        true_samples: jnp.ndarray,
        generated_samples: jnp.ndarray
    ) -> float:
        """
        Compute Classifier Two-Sample Test (C2ST) accuracy between two sample sets.
        
        A score of 0.5 indicates indistinguishability (perfect alignment),
        while 1.0 indicates perfect separability (poor alignment).
        
        Implements methodology from Lueckmann et al. (2021).
        
        Args:
            true_samples: Array of shape (N, D) - samples from true distribution (e.g., MCMC)
            generated_samples: Array of shape (M, D) - samples from estimated distribution
            
        Returns:
            Classification accuracy (scalar)
        """
        n_samples = self.c2st_cfg.get("n_samples", 1000)
        n_trials = self.c2st_cfg.get("n_trials", 10)
        
        # Ensure we don't exceed available samples
        N_true = min(n_samples, len(true_samples))
        N_gen = min(n_samples, len(generated_samples))
        
        # Use standalone function from metrics.py
        return compute_c2st(
            true_samples[:N_true],
            generated_samples[:N_gen],
            n_trials=n_trials,
            random_seed=42
        )

    def expected_coverage(
        self,
        posterior_samples: jnp.ndarray,
        true_params: jnp.ndarray,
        alpha_levels: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute expected coverage for calibration analysis.
        
        For each alpha level, checks whether the true parameter lies within the 
        alpha-level highest density region of the estimated posterior.
        A well-calibrated model should have empirical coverage equal to nominal alpha.
        
        Implements methodology from Hermans et al. (2022); used in Appendix Figs. A9-A13.
        
        Args:
            posterior_samples: Array of shape (num_tasks, num_samples, param_dim)
                              - posterior samples for multiple observed datasets
            true_params: Array of shape (num_tasks, param_dim)
                        - true parameters corresponding to each dataset
            alpha_levels: Array of shape (K,) - credible levels to evaluate
                         If None, uses default from config
        
        Returns:
            Array of shape (K,) - empirical coverage rates for each alpha level
        """
        if alpha_levels is None:
            alpha_list = self.calib_cfg.get("alpha_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            alpha_levels = jnp.array(alpha_list)
            
        return compute_expected_coverage(posterior_samples, true_params, alpha_levels)

    def negative_log_likelihood(
        self,
        x0: jnp.ndarray,
        model_fn: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None
    ) -> float:
        """
        Compute negative log-likelihood using probability flow ODE integration.
        
        Solves the reverse-time probability flow ODE to compute exact log-likelihood
        under the score-based model. This is necessary because diffusion models do not
        directly provide likelihoods like normalizing flows.
        
        Implements methodology from Song et al. (2021b); used in Fig. A8.
        
        Args:
            x0: Ground truth point (theta, x) of shape (D,)
            model_fn: Function that takes (xt, t) and returns score s_phi(xt, t)
                     If None, uses self.model.apply
        
        Returns:
            Negative log-likelihood of x0 under the model
        """
        if model_fn is None:
            # Create a callable that wraps model.apply
            def model_fn_wrapper(xt, t):
                return self.model.apply(
                    {},  # params would be passed in real usage
                    xt.reshape(1, -1), 
                    jnp.array([t]), 
                    method=lambda m, x, t_: m(x, t_, None, None)
                ).reshape(-1)
        
        # Determine SDE type and instantiate
        sde_type = self.config["sde"].get("type", "VESDE")
        if sde_type == "VESDE":
            sde_cfg = self.config["sde"]["vesde"]
            sde_instance = VESDE(
                sigma_min=sde_cfg["sigma_min"],
                sigma_max=sde_cfg["sigma_max"],
                t_min=sde_cfg["t_min"],
                t_max=sde_cfg["t_max"]
            )
        else:
            sde_cfg = self.config["sde"]["vpsde"]
            sde_instance = VPSDE(
                beta_min=sde_cfg["beta_min"],
                beta_max=sde_cfg["beta_max"],
                t_min=sde_cfg["t_min"],
                t_max=sde_cfg["t_max"]
            )
        
        # Extract ODE solver parameters
        rtol = self.nll_cfg.get("rtol", 1e-5)
        atol = self.nll_cfg.get("atol", 1e-5)
        timesteps = self.nll_cfg.get("timesteps", 1000)
        
        return nll_from_ode(model_fn_wrapper, x0, sde_instance, timesteps=timesteps, rtol=rtol, atol=atol)

    def generate_posterior_predictive(
        self,
        theta_samples: jnp.ndarray,
        simulator_name: str
    ) -> jnp.ndarray:
        """
        Generate posterior predictive samples by running forward simulation.
        
        Used to validate that Simformer captures both posterior and data-generating process.
        See Fig. 5a-b and Fig. 7c,f for examples.
        
        Args:
            theta_samples: Array of shape (num_samples, param_dim) - posterior samples
            simulator_name: Name of the simulator to use
            
        Returns:
            Predictive samples of shape (num_samples, data_dim)
        """
        simulator = self.simulators[simulator_name]
        predictions = []
        
        for i in range(len(theta_samples)):
            # Convert array to dict based on simulator expectations
            # This requires knowledge of parameter ordering
            theta_dict = self._array_to_theta_dict(theta_samples[i], simulator_name)
            x_dict = simulator.simulate(theta_dict)
            
            # Extract values in consistent order
            pred_vals = self._dict_to_array(x_dict)
            predictions.append(pred_vals)
            
        return jnp.stack(predictions)

    def evaluate_task(
        self,
        task_name: str,
        num_observations: int = 10,
        num_samples_per_obs: int = 1000,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Perform full evaluation on a single task.
        
        Generates multiple observations, infers posteriors with Simformer,
        obtains ground-truth posteriors via MCMC, and computes all metrics.
        
        Args:
            task_name: Name of task to evaluate
            num_observations: Number of different observed datasets to test on
            num_samples_per_obs: Number of posterior samples per observation
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing all results and metrics
        """
        key = jax.random.PRNGKey(seed)
        simulator = self.simulators[task_name]
        
        # Results containers
        c2st_scores = []
        coverage_rates = []
        nll_values = []
        
        # True parameters and posterior samples for calibration
        true_params_list = []
        posterior_samples_list = []
        
        for obs_idx in range(num_observations):
            # Split key
            key, subkey = jax.random.split(key)
            
            # Generate observation
            theta_true, x_observed = simulator.sample(subkey)
            true_params_list.append(list(theta_true.values()))
            
            # Get ground-truth posterior samples via MCMC
            # Note: This is simplified; full implementation requires MCMC kernel
            # Here we assume access to reference_samples from external MCMC run
            # In practice, would use blackjax or custom HMC/slice sampler
            reference_samples = self._mock_mcmc_samples(simulator, x_observed, num_samples_per_obs)
            
            # Generate Simformer posterior samples
            simformer_samples = self._sample_with_simformer(x_observed, num_samples_per_obs)
            posterior_samples_list.append(simformer_samples)
            
            # Compute C2ST
            c2st_score = self.compute_c2st(reference_samples, simformer_samples)
            c2st_scores.append(c2st_score)
            
            # Compute NLL for true parameter
            # Concatenate theta_true and some representation of x_observed
            # Simplified here
            theta_array = jnp.array(list(theta_true.values()))
            x_array = jnp.array(list(x_observed.values())[:len(theta_array)])  # Match dimensions
            joint_point = jnp.concatenate([theta_array, x_array])
            nll_val = self.negative_log_likelihood(joint_point)
            nll_values.append(nll_val)
        
        # Stack arrays
        true_params_arr = jnp.array(true_params_list)
        posterior_samples_arr = jnp.stack(posterior_samples_list)
        
        # Compute expected coverage
        alpha_levels = jnp.array(self.calib_cfg.get("alpha_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        coverage_rates = self.expected_coverage(posterior_samples_arr, true_params_arr, alpha_levels)
        
        return {
            "task": task_name,
            "c2st_mean": float(np.mean(c2st_scores)),
            "c2st_std": float(np.std(c2st_scores)),
            "coverage_rates": coverage_rates.tolist(),
            "nll_mean": float(np.mean(nll_values)),
            "nll_std": float(np.std(nll_values)),
            "alpha_levels": alpha_levels.tolist()
        }

    def _sample_with_simformer(
        self,
        x_observed: Dict[str, float],
        num_samples: int,
        conditioning_strategy: str = "posterior"
    ) -> jnp.ndarray:
        """
        Sample from Simformer posterior given observed data.
        
        Args:
            x_observed: Dictionary of observed data points
            num_samples: Number of posterior samples to generate
            conditioning_strategy: How to set condition mask ("posterior", "guided")
            
        Returns:
            Array of shape (num_samples, num_params) - posterior samples
        """
        # This is a simplified placeholder
        # Full implementation requires mapping x_observed to token positions
        # and constructing appropriate cond_mask
        
        # For now, return mock samples
        key = jax.random.PRNGKey(42)
        return jax.random.normal(key, (num_samples, 2))  # Mock 2D parameter space

    def _mock_mcmc_samples(
        self,
        simulator: BaseSimulator,
        x_observed: Dict[str, float],
        num_samples: int
    ) -> jnp.ndarray:
        """
        Mock method to simulate MCMC sampling.
        
        In reality, this would run HMC/MH chains until convergence.
        As described in Appendix A2.2: slice + MH-MCMC.
        
        Args:
            simulator: Simulator instance
            x_observed: Observed data
            num_samples: Number of samples to return
            
        Returns:
            Mock posterior samples
        """
        key = jax.random.PRNGKey(43)
        return jax.random.normal(key, (num_samples, 2))  # Mock 2D parameter space

    def _array_to_theta_dict(self, arr: jnp.ndarray, task_name: str) -> Dict[str, float]:
        """
        Convert parameter array to dictionary format expected by simulator.
        
        Args:
            arr: Parameter vector
            task_name: Task name to determine mapping
            
        Returns:
            Dictionary of parameter names to values
        """
        # This is task-specific and simplified
        if task_name == "Gaussian_Linear":
            return {f"theta_{i}": float(arr[i]) for i in range(len(arr))}
        else:
            return {f"param_{i}": float(arr[i]) for i in range(len(arr))}

    def _dict_to_array(self, d: Dict[str, float]) -> jnp.ndarray:
        """
        Convert dictionary to sorted array.
        
        Args:
            d: Dictionary of values
            
        Returns:
            Array of values in sorted key order
        """
        return jnp.array([d[k] for k in sorted(d.keys())])

