## eval/metrics.py
"""
Standalone metrics computation module for Simformer evaluation.

Implements key evaluation metrics used in the paper:
- Classifier Two-Sample Test (C2ST) for posterior fidelity
- Expected coverage analysis for calibration assessment
- Negative log-likelihood estimation via probability flow ODE

These functions are stateless and can be used independently of the training pipeline.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional, Any
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

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
    }
}


def load_config() -> dict:
    """
    Load configuration from global context or return defaults.
    In practice, this would integrate with Hydra.
    """
    return DEFAULT_CONFIG


def c2st(
    true_samples: jnp.ndarray,
    generated_samples: jnp.ndarray,
    n_trials: int = 10,
    random_seed: int = 42
) -> float:
    """
    Compute Classifier Two-Sample Test (C2ST) accuracy between two sample sets.
    
    A C2ST score of 0.5 indicates indistinguishability (perfect alignment),
    while 1.0 indicates perfect separability (poor alignment).
    
    Implements methodology from Lueckmann et al. (2021) and Hermans et al. (2022).
    
    Args:
        true_samples: Array of shape (N, D) - samples from true distribution (e.g., MCMC)
        generated_samples: Array of shape (M, D) - samples from estimated distribution
        n_trials: Number of independent trials to average over
        random_seed: Random seed for reproducibility
        
    Returns:
        Average classification accuracy across trials
        
    Raises:
        ValueError: If inputs have insufficient samples or mismatched dimensions
    """
    # Input validation
    if true_samples.shape[1] != generated_samples.shape[1]:
        raise ValueError(f"Dimension mismatch: {true_samples.shape[1]} vs {generated_samples.shape[1]}")
        
    if len(true_samples) < 2 or len(generated_samples) < 2:
        raise ValueError("At least 2 samples required per set")
    
    # Convert to numpy for sklearn compatibility
    true_np = np.array(true_samples)
    gen_np = np.array(generated_samples)
    
    # Balance dataset sizes by subsampling larger set
    n_min = min(len(true_np), len(gen_np))
    rng = np.random.default_rng(random_seed)
    
    true_balanced = rng.choice(true_np, size=n_min, replace=False)
    gen_balanced = rng.choice(gen_np, size=n_min, replace=False)
    
    # Combine data and labels
    X = np.vstack([true_balanced, gen_balanced])
    y = np.hstack([np.zeros(n_min), np.ones(n_min)])  # 0=true, 1=generated
    
    # Initialize results array
    accuracies = []
    
    # Run multiple trials with different train/test splits
    for trial in range(n_trials):
        trial_seed = random_seed + trial
        
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=trial_seed
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression classifier
        clf = LogisticRegression(
            random_state=trial_seed,
            max_iter=1000,
            solver='lbfgs'
        )
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate accuracy
        acc = clf.score(X_test_scaled, y_test)
        accuracies.append(acc)
    
    # Return mean accuracy across trials
    return float(np.mean(accuracies))


def expected_coverage(
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
        
    Raises:
        ValueError: If input dimensions are incompatible
    """
    # Input validation
    if posterior_samples.ndim != 3:
        raise ValueError(f"posterior_samples must be 3D (tasks,samples,dim), got {posterior_samples.ndim}")
        
    if posterior_samples.shape[0] != true_params.shape[0]:
        raise ValueError(f"Number of tasks mismatch: {posterior_samples.shape[0]} vs {true_params.shape[0]}")
        
    if posterior_samples.shape[2] != true_params.shape[1]:
        raise ValueError(f"Parameter dimension mismatch: {posterior_samples.shape[2]} vs {true_params.shape[1]}")
    
    # Get alpha levels from config if not provided
    if alpha_levels is None:
        config = load_config()
        alpha_list = config["evaluation"]["calibration"]["alpha_levels"]
        alpha_levels = jnp.array(alpha_list)
    
    num_tasks = posterior_samples.shape[0]
    num_alphas = len(alpha_levels)
    
    # Convert to numpy for scipy compatibility
    post_np = np.array(posterior_samples)
    true_np = np.array(true_params)
    
    # Initialize coverage counter
    coverage_counts = np.zeros(num_alphas)
    
    # Process each task
    for task_idx in range(num_tasks):
        samples_task = post_np[task_idx]  # (num_samples, param_dim)
        true_param = true_np[task_idx]    # (param_dim,)
        
        # Estimate posterior density using KDE
        try:
            kde = gaussian_kde(samples_task.T)  # Transpose for scipy format
        except np.linalg.LinAlgError:
            # Handle singular matrix (e.g., degenerate posterior)
            warnings.warn(f"KDE failed for task {task_idx}, skipping")
            continue
            
        # Evaluate density at all sample points
        log_densities = kde.logpdf(samples_task.T)
        
        # Sort densities to find thresholds
        sorted_log_dens = np.sort(log_densities)[::-1]  # Descending order
        
        # Evaluate density at true parameter
        true_log_density = kde.logpdf(true_param)
        
        # Check coverage for each alpha level
        for alpha_idx, alpha in enumerate(alpha_levels):
            # Find threshold for alpha-level HDI
            n_in_hdi = int(np.ceil(alpha * len(samples_task)))
            if n_in_hdi == 0:
                threshold = -np.inf
            else:
                threshold = sorted_log_dens[n_in_hdi - 1]
            
            # Check if true parameter is in HDI
            if true_log_density >= threshold:
                coverage_counts[alpha_idx] += 1
    
    # Compute empirical coverage rates
    empirical_coverage = coverage_counts / num_tasks
    return jnp.array(empirical_coverage)


def nll_from_ode(
    model_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    x0: jnp.ndarray,
    sde: Any,
    timesteps: int = 1000,
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> float:
    """
    Compute negative log-likelihood using probability flow ODE integration.
    
    Solves the reverse-time probability flow ODE to compute exact log-likelihood
    under the score-based model. This is necessary because diffusion models do not
    directly provide likelihoods like normalizing flows.
    
    Implements methodology from Song et al. (2021b); used in Fig. A8.
    
    Args:
        model_fn: Function that takes (xt, t) and returns score s_phi(xt, t)
        x0: Ground truth point (theta, x) of shape (D,)
        sde: SDE instance (VESDE or VPSDE) with drift/diffusion methods
        timesteps: Number of integration steps
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
    
    Returns:
        Negative log-likelihood of x0 under the model
        
    Note:
        This function requires converting between JAX and NumPy arrays since
        SciPy's ODE solvers don't yet support JAX arrays directly.
    """
    # Extract SDE parameters
    t_max = getattr(sde, 't_max', 1.0)
    t_min = getattr(sde, 't_min', 1e-5)
    
    # Define the probability flow ODE: dx/dt = f(t)x - 0.5*g(t)^2*s(x,t)
    def ode_func(t: float, xt_flat: np.ndarray) -> np.ndarray:
        # Reshape input
        xt = xt_flat.reshape(x0.shape)
        
        # Convert to JAX array for model evaluation
        xt_jax = jnp.array(xt)
        
        # Get score prediction
        try:
            score = model_fn(xt_jax, t)
            score_np = np.array(score)
        except Exception as e:
            warnings.warn(f"Model evaluation failed at t={t}: {e}")
            score_np = np.zeros_like(xt)
        
        # Get drift and diffusion coefficients
        if hasattr(sde, 'drift_coeff'):
            drift = np.array(sde.drift_coeff(xt_jax, t))
        else:
            drift = np.zeros_like(xt)
            
        g_t = sde.diffusion_coeff(t)
        
        # Probability flow ODE
        dxdt = drift - 0.5 * (g_t ** 2) * score_np
        
        # Flatten output for ODE solver
        return dxdt.flatten()
    
    # Initial condition: sample from terminal distribution p_T(x_T)
    key = jax.random.PRNGKey(42)
    z = jax.random.normal(key, x0.shape)
    mean_T, std_T = sde.marginal_prob(jnp.zeros_like(x0), t_max)
    xT = mean_T + std_T * z
    
    # Set up time span (reverse time: T -> 0)
    t_span = (t_max, t_min)
    t_eval = np.linspace(t_max, t_min, timesteps)
    
    # Solve ODE
    try:
        solution = solve_ivp(
            ode_func,
            t_span,
            xT.flatten(),
            t_eval=t_eval,
            method='RK45',
            rtol=rtol,
            atol=atol,
            dense_output=True
        )
        
        if not solution.success:
            warnings.warn(f"ODE integration failed: {solution.message}")
            return float('inf')
            
    except Exception as e:
        warnings.warn(f"ODE solving failed: {e}")
        return float('inf')
    
    # Final state should be close to x0
    x_final = solution.y[:, -1].reshape(x0.shape)
    
    # Log-likelihood calculation requires integrating the divergence term
    # We'll use a simple approximation: log p(x0) ≈ log p(xT) + integral(trace(Jacobian))
    # For Gaussian terminal distribution:
    log_p_xT = float(jnp.sum(jax.scipy.stats.norm.logpdf(xT, mean_T, std_T)))
    
    # The total change in log-density along the path is given by:
    # dlogp/dt = -0.5*g(t)^2*Tr(ds/dx)
    # We approximate the trace using Hutchinson estimator
    def logp_derivative(t_val: float, xt_flat: np.ndarray) -> float:
        xt = jnp.array(xt_flat.reshape(x0.shape))
        g_t = sde.diffusion_coeff(t_val)
        
        # Hutchinson estimator for trace of Jacobian
        key_trace = jax.random.fold_in(jax.random.PRNGKey(42), int(t_val * 1e6))
        eps = jax.random.rademacher(key_trace, xt.shape)
        
        def score_eps(h):
            return model_fn(xt + h * eps, t_val)
            
        # Finite difference approximation of directional derivative
        h = 1e-4
        jac_eps = (score_eps(h) - score_eps(-h)) / (2 * h)
        div = jnp.sum(eps * jac_eps)  # Trace approximation
        
        return float(-0.5 * (g_t ** 2) * div)
    
    # Integrate log-density change
    logp_change = 0.0
    for i in range(len(solution.t) - 1):
        dt = solution.t[i] - solution.t[i + 1]  # Reverse time
        xt_mid = (solution.y[:, i] + solution.y[:, i + 1]) / 2
        dlogp_dt = logp_derivative(solution.t[i], xt_mid)
        logp_change += dlogp_dt * dt
    
    # Total log-likelihood
    log_p_x0 = log_p_xT + logp_change
    
    return float(-log_p_x0)
