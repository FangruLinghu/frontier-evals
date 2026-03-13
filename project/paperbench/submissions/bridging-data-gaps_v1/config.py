## config.py
"""Configuration and hyperparameters for similarity-guided diffusion model.

This module defines the Config class with all hyperparameters from the paper,
including factory functions for toy and FFHQ experiments, and noise schedule
parameter computation.

Hyperparameters:
    - gamma (λ): similarity guidance strength (recommended: 5.0)
    - omega (ω): learning rate for adversarial noise selection (recommended: 0.02)
    - J: number of gradient ascent iterations for adversarial noise (recommended: 10)
    - learning_rate (η): for updating adaptor parameters ψ (5e-5 for DDPMs, 1e-5 for LDMs)
    - c_factor: downsampling factor for adaptor layers (4 for DDPMs, 2 for LDMs)
    - d_dim: projection dimension for adaptor layers (8)
"""

import torch
from typing import Tuple, Dict, Optional


class Config:
    """Configuration class for similarity-guided diffusion model training.
    
    Contains all hyperparameters as described in the paper:
    - gamma: similarity guidance strength (λ in paper)
    - omega: learning rate for adversarial noise selection (ω in Equation 7)
    - J: number of gradient ascent iterations for adversarial noise selection
    - learning_rate: η for updating adaptor parameters ψ
    - c_factor: downsampling factor for adaptor layers (c=4 for DDPM, c=2 for LDM)
    - d_dim: projection dimension for adaptor layers (d=8)
    """
    
    def __init__(
        self,
        gamma: float = 5.0,
        omega: float = 0.02,
        J: int = 10,
        learning_rate: float = 5e-5,
        batch_size: int = 40,
        epochs: int = 10,
        iterations: int = 300,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        source_mean: Tuple[float, float] = (1.0, 1.0),
        target_mean: Tuple[float, float] = (-1.0, -1.0),
        source_variance: float = 1.0,
        few_shot_samples: int = 10,
        c_factor: int = 4,
        d_dim: int = 8,
        model_type: str = 'ddpm',
        device: str = 'cuda',
        eval_num_samples: int = 1000,
        eval_num_steps: int = 1000,
        save_dir: str = './checkpoints',
        log_interval: int = 10,
        eval_interval: int = 50
    ) -> None:
        """Initialize configuration with hyperparameters.
        
        Sets default values based on paper recommendations:
        - γ=5.0, ω=0.02, J=10, learning_rate=5e-5 for DDPMs
        - batch_size=40, iterations=300
        - c=4, d=8 for DDPMs
        
        Args:
            gamma: Hyperparameter controlling similarity guidance strength (λ in paper).
                   Controls trade-off between standard denoising loss and similarity-guided
                   gradient. Recommended value: 5.0
            omega: Learning rate for gradient ascent in adversarial noise selection
                   (ω in Equation 7). Controls step size when finding worst-case noise ε*.
                   Recommended value: 0.02
            J: Number of gradient ascent iterations for adversarial noise selection
               (J in Algorithm 1). Iterates j = 0, 1, ..., J-1 to find ε*.
               Recommended value: 10
            learning_rate: Learning rate η for updating adaptor parameters ψ
                          (from Algorithm 1 step 6). Recommended: 5×10^-5 for DDPMs,
                          1×10^-5 for LDMs
            batch_size: Batch size for training. Paper uses 40 in experiments
            epochs: Number of training epochs. Note: Paper recommends ~300 iterations
                   total (not epochs), stored in iterations parameter
            iterations: Total training iterations. Paper recommends 300 iterations
                       for convergence
            timesteps: Number of diffusion timesteps T. Standard DDPM uses T=1000,
                      but can be reduced for efficiency
            beta_start: Starting value for noise schedule β_1. Standard: 0.0001
            beta_end: Ending value for noise schedule β_T. Standard: 0.02
            source_mean: Mean of source distribution for toy experiments. Paper uses (1, 1)
            target_mean: Mean of target distribution for toy experiments. Paper uses (-1, -1)
            source_variance: Variance of source/target distributions (isotropic Gaussian).
                            Paper uses 1.0 (identity matrix I)
            few_shot_samples: Number of few-shot samples from target domain.
                             Paper uses 10 samples in toy experiment
            c_factor: Downsampling factor for adaptor layers in DDPMs. Paper: c=4
            d_dim: Projection dimension for adaptor layers. Paper: d=8
            model_type: Type of diffusion model: 'ddpm' or 'ldm'.
                       Affects c_factor (4 for DDPM, 2 for LDM)
            device: Device for training: 'cuda' or 'cpu'
            eval_num_samples: Number of samples for evaluation (Intra-LPIPS computation)
            eval_num_steps: Number of denoising steps for sampling during evaluation
            save_dir: Directory to save checkpoints and results
            log_interval: Interval for logging training progress
            eval_interval: Interval for evaluation during training
        """
        self.gamma = gamma
        self.omega = omega
        self.J = J
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = iterations
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.source_mean = source_mean
        self.target_mean = target_mean
        self.source_variance = source_variance
        self.few_shot_samples = few_shot_samples
        self.c_factor = c_factor
        self.d_dim = d_dim
        self.model_type = model_type
        self.device = device
        self.eval_num_samples = eval_num_samples
        self.eval_num_steps = eval_num_steps
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization or passing to functions.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'gamma': self.gamma,
            'omega': self.omega,
            'J': self.J,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'iterations': self.iterations,
            'timesteps': self.timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'source_mean': self.source_mean,
            'target_mean': self.target_mean,
            'source_variance': self.source_variance,
            'few_shot_samples': self.few_shot_samples,
            'c_factor': self.c_factor,
            'd_dim': self.d_dim,
            'model_type': self.model_type,
            'device': self.device,
            'eval_num_samples': self.eval_num_samples,
            'eval_num_steps': self.eval_num_steps,
            'save_dir': self.save_dir,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval
        }
    
    def get_adaptor_config(self) -> Dict:
        """Get adaptor-specific configuration based on model_type.
        
        Returns:
            Dictionary with c_factor and d_dim
        """
        return {
            'c_factor': self.c_factor,
            'd_dim': self.d_dim
        }
    
    def update(self, kwargs: Dict) -> None:
        """Update configuration parameters with new values.
        
        Useful for hyperparameter sweeps or modifying specific settings.
        
        Args:
            kwargs: Dictionary of parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
        
        # Auto-update c_factor based on model_type if model_type is changed
        if 'model_type' in kwargs:
            self._update_c_factor()
    
    def _update_c_factor(self) -> None:
        """Update c_factor based on model_type."""
        if self.model_type == 'ddpm':
            self.c_factor = 4
        elif self.model_type == 'ldm':
            self.c_factor = 2
        else:
            raise ValueError(
                f"Invalid model_type: {self.model_type}. Must be 'ddpm' or 'ldm'"
            )


def create_toy_config() -> Config:
    """Factory function to create configuration optimized for toy data experiments (2D Gaussian).
    
    Uses source_mean=(1,1), target_mean=(-1,-1), few_shot_samples=10
    as described in paper Section 5.1 toy experiment.
    
    Returns:
        Config instance configured for toy experiments
    """
    return Config(
        gamma=5.0,
        omega=0.02,
        J=10,
        learning_rate=5e-5,
        batch_size=40,
        epochs=10,
        iterations=300,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        source_mean=(1, 1),
        target_mean=(-1, -1),
        source_variance=1.0,
        few_shot_samples=10,
        c_factor=4,
        d_dim=8,
        model_type='ddpm',
        device='cuda',
        eval_num_samples=1000,
        eval_num_steps=1000,
        save_dir='./checkpoints/toy',
        log_interval=10,
        eval_interval=50
    )


def create_ffhq_config(target_domain: str = 'sketches') -> Config:
    """Factory function to create configuration for FFHQ experiments.
    
    Args:
        target_domain: Target domain for adaptation. Options include:
            - 'sketches': FFHQ → Sketches
            - 'amedeo': FFHQ → Amedeo's paintings
            - 'sunglasses': FFHQ → Sunglasses
            - Other custom target domains
    
    Returns:
        Config instance configured for FFHQ experiments
    """
    valid_domains = ['sketches', 'amedeo', 'sunglasses', 'custom']
    if target_domain not in valid_domains:
        raise ValueError(
            f"Unknown target_domain: {target_domain}. "
            f"Must be one of {valid_domains}"
        )
    
    # Set appropriate save directory based on target domain
    save_dir = f'./checkpoints/ffhq_{target_domain}'
    
    return Config(
        gamma=5.0,
        omega=0.02,
        J=10,
        learning_rate=5e-5,  # 5×10^-5 for DDPMs
        batch_size=40,
        epochs=10,
        iterations=300,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        source_mean=(0.5, 0.5, 0.5),  # Image mean for normalization
        target_mean=(-1, -1, -1),  # Different target mean for images
        source_variance=1.0,
        few_shot_samples=10,
        c_factor=4,  # c=4 for DDPMs
        d_dim=8,  # d=8
        model_type='ddpm',
        device='cuda',
        eval_num_samples=1000,
        eval_num_steps=1000,
        save_dir=save_dir,
        log_interval=10,
        eval_interval=50
    )


def get_noise_schedule_params(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02
) -> Dict[str, torch.Tensor]:
    """Get linear noise schedule parameters.
    
    Computes β_t values and derived parameters:
    - β_t: linear noise schedule from beta_start to beta_end
    - α_t: 1 - β_t (one minus noise coefficient)
    - α_t_cumprod: cumulative product of α_t (denoted as ᾱ_t in paper)
    - σ̂_t: standard deviation estimate for denoising (σ̂_t in Equation 5)
    
    The derived parameters are used in:
    - Forward process: x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε
    - Reverse process: μ_t = (x_t - √(1 - α_t) * ε_θ(x_t,t)) / √(α_t)
    - Similarity-guided loss: gradient includes σ̂_t terms
    
    Args:
        timesteps: Number of diffusion timesteps T
        beta_start: Starting value for noise schedule β_1. Standard: 0.0001
        beta_end: Ending value for noise schedule β_T. Standard: 0.02
    
    Returns:
        Dictionary containing:
            - 'betas': β_t values for all timesteps (shape: [T])
            - 'alphas': α_t = 1 - β_t values (shape: [T])
            - 'alphas_cumprod': cumulative product of α_t, ᾱ_t (shape: [T])
            - 'sigma_hat': σ̂_t standard deviation estimates (shape: [T])
    """
    # Linear noise schedule: β_t linearly interpolated from beta_start to beta_end
    betas = torch.linspace(beta_start, beta_end, timesteps)
    
    # α_t = 1 - β_t
    alphas = 1.0 - betas
    
    # ᾱ_t = ∏_{i=1}^t α_i (cumulative product)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # σ̂_t = sqrt((1 - α_{t-1}) * α_t / (1 - ᾱ_t))
    # For t=1, we define α_0 = 1 (no noise at t=0)
    alphas_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
    # Compute sigma_hat_t for all timesteps
    sigma_hat = torch.sqrt((1 - alphas_prev) * betas / (1 - alphas_cumprod))
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sigma_hat': sigma_hat
    }