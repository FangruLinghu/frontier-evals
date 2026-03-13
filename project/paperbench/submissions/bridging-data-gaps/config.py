## config.py
"""
Centralized configuration module for DPMs-ANT (Diffusion Probabilistic Models with Adversarial Noise-based Transfer Learning).

This module provides a structured, type-safe interface to all hyperparameters and settings used across the system.
It ensures consistency, reproducibility, and modularity by serving as the single source of truth for:
- Training loop control
- Model architecture (especially adaptor layers)
- Diffusion process parameters
- Adversarial noise generation
- Similarity-guided training
- Classifier setup
- Dataset handling
- Evaluation metrics
- Logging and ablation studies

All values are derived from the provided config.yaml and the paper's specifications.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

# Define types for clarity and type checking
NormMethod = Literal["batch_norm", "clip_and_scale", "project_sphere"]
OptimizerType = Literal["Adam", "AdamW"]

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    iterations: int = 300
    batch_size: int = 40
    lr_ddpm: float = 5e-5
    lr_ldm: float = 1e-5
    optimizer: OptimizerType = "Adam"

@dataclass
class ModelConfig:
    """Configuration for the diffusion model and similarity guidance."""
    T: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    gamma: float = 5.0
    use_ema: bool = False

@dataclass
class AdaptorConfig:
    """Configuration for the parameter-efficient adaptor module."""
    enabled: bool = True
    c_ddpm: int = 4  # spatial reduction factor for DDPM
    c_ldm: int = 2   # spatial reduction factor for LDM
    d: int = 8       # bottleneck dimension in adaptor MLP
    init_zero: bool = True  # initialize adaptor weights to zero

@dataclass
class AdversarialNoiseConfig:
    """Configuration for adversarial noise selection."""
    J: int = 10      # number of inner gradient ascent steps
    omega: float = 0.02  # step size for noise update
    norm_method: NormMethod = "batch_norm"  # method to normalize noise during ascent

@dataclass
class ClassifierConfig:
    """Configuration for the binary classifier used in similarity guidance."""
    train_from_scratch: bool = True
    num_shots: int = 10
    freeze_after_train: bool = True
    input_type: str = "noised_image"  # operates on xt, not x0
    t_range: str = "uniform"  # timestep sampling strategy

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    fid_real_dataset_size: int = 2500  # minimum real images needed for FID
    intra_lpips_num_samples: int = 1000  # number of generated images for Intra-LPIPS
    compute_fid: bool = True
    compute_intra_lpips: bool = True

@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    source_domains: List[str] = field(default_factory=lambda: ["FFHQ", "LSUN_Church"])
    target_domains_10shot: List[str] = field(default_factory=lambda: [
        "Babies", "Sunglasses", "Raphael_Peale", "Sketches", 
        "Amedeo_Modigliani", "Haunted_Houses", "Landscape_Drawings"
    ])
    image_size: int = 256  # standard resolution based on LDM usage
    center_crop: bool = True  # apply center crop before resizing

@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""
    save_checkpoints: bool = True
    log_frequency: int = 10
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    w_o_an: bool = False  # without adversarial noise
    w_o_sg: bool = False  # without similarity guidance
    full_model: bool = True  # full DPMs-ANT method

@dataclass
class Config:
    """
    Main configuration class that aggregates all sub-configurations.
    
    This is the primary interface for accessing configuration values throughout the codebase.
    """
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adaptor: AdaptorConfig = field(default_factory=AdaptorConfig)
    adversarial_noise: AdversarialNoiseConfig = field(default_factory=AdversarialNoiseConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

# Global configuration instance
config = Config()

# Convenience accessors for commonly used values
def get_adaptor_reduction_factor(model_type: str) -> int:
    """
    Get the spatial reduction factor 'c' for adaptor based on model type.
    
    Args:
        model_type: Either 'ddpm' or 'ldm'
        
    Returns:
        The reduction factor c
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == 'ddpm':
        return config.adaptor.c_ddpm
    elif model_type.lower() == 'ldm':
        return config.adaptor.c_ldm
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'ddpm' or 'ldm'.")

def get_learning_rate(model_type: str) -> float:
    """
    Get the appropriate learning rate for the given model type.
    
    Args:
        model_type: Either 'ddpm' or 'ldm'
        
    Returns:
        The learning rate
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == 'ddpm':
        return config.training.lr_ddpm
    elif model_type.lower() == 'ldm':
        return config.training.lr_ldm
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'ddpm' or 'ldm'.")
