## utils/config.py
"""
Configuration management for Simformer using Hydra.

Implements the ConfigManager class that loads and manages experiment configuration
from config.yaml and command-line overrides. Provides centralized access to all
hyperparameters and settings used throughout the pipeline.
"""

import os
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore

# Define structured configuration classes to ensure type safety
@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 1000
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    early_stopping: bool = True
    validation_split: float = 0.1
    max_epochs: Optional[int] = None
    patience: int = 20
    min_delta: float = 1e-4


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    architecture: str = "Simformer"
    token_dim: int = 50
    time_embedding_dim: int = 128
    num_layers: int = 6
    num_heads: int = 4
    attention_size: int = 10
    widening_factor: int = 3
    use_fourier_features: bool = True
    fourier_scale: float = 1.0


@dataclass
class VESDEConfig:
    """Variance Exploding SDE parameters."""
    sigma_min: float = 0.0001
    sigma_max: float = 15.0
    t_min: float = 1e-5
    t_max: float = 1.0


@dataclass
class VPSDEConfig:
    """Variance Preserving SDE parameters."""
    beta_min: float = 0.01
    beta_max: float = 10.0
    t_min: float = 1e-5
    t_max: float = 1.0


@dataclass
class SDEConfig:
    """SDE configuration with nested types."""
    type: str = "VESDE"
    vesde: VESDEConfig = field(default_factory=VESDEConfig)
    vpsde: VPSDEConfig = field(default_factory=VPSDEConfig)


@dataclass
class MaskingConfig:
    """Masking strategy configuration."""
    condition_mask_sampling: Dict[str, Any] = field(default_factory=lambda: {
        "strategies": ["joint", "posterior", "likelihood", "random_p03", "random_p07"],
        "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2]
    })
    attention_mask_type: str = "undirected"
    dynamic_mask_adaptation: bool = True


@dataclass
class SamplingConfig:
    """Sampling configuration."""
    reverse_sde_solver: str = "euler_maruyama"
    num_steps: int = 500
    self_recurrence_steps: int = 0


@dataclass
class GuidanceConfig:
    """Guidance configuration."""
    scaling_function: str = "1/sigma(t)**2"
    constraint_temperature: Optional[float] = None


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration."""
    c2st: Dict[str, Any] = field(default_factory=lambda: {
        "classifier": "logistic_regression",
        "n_samples": 1000,
        "n_trials": 10
    })
    calibration: Dict[str, Any] = field(default_factory=lambda: {
        "alpha_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    nll: Dict[str, Any] = field(default_factory=lambda: {
        "method": "probability_flow_ode",
        "solver": "dopri5",
        "rtol": 1e-5,
        "atol": 1e-5
    })


@dataclass
class TaskSpecificConfig:
    """Task-specific model overrides."""
    Lotka_Volterra: Dict[str, int] = field(default_factory=lambda: {"num_layers": 8})
    SIRD: Dict[str, int] = field(default_factory=lambda: {"num_layers": 8})
    Hodgkin_Huxley: Dict[str, int] = field(default_factory=lambda: {"num_layers": 8})


@dataclass
class RootConfig:
    """Root configuration containing all nested configs."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task_specific: TaskSpecificConfig = field(default_factory=TaskSpecificConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    tasks: list = field(default_factory=lambda: [
        "Gaussian_Linear", "Gaussian_Mixture", "Two_Moons", "SLCP", 
        "Tree", "HMM", "Lotka_Volterra", "SIRD", "Hodgkin_Huxley"
    ])
    simulation_budgets: list = field(default_factory=lambda: [1000, 10000, 100000])
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hydra: Dict[str, Any] = field(default_factory=lambda: {
        "run": {"dir": "outputs/${now:%Y-%m-%d}/${task_name}_${simulation_budget}"},
        "sweep": {
            "dir": "multirun/${now:%Y-%m-%d}/${task_name}",
            "override_dirname": "lr_${learning_rate}_bs_${batch_size}_budget_${simulation_budget}"
        }
    })


# Register the config schema with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=RootConfig)


class ConfigManager:
    """
    Centralized configuration manager for Simformer experiments.
    
    Loads configuration from config.yaml and command-line overrides using Hydra.
    Applies task-specific overrides and provides unified access to all settings.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize configuration manager with loaded Hydra config.
        
        Args:
            config: Hydra DictConfig object loaded by @hydra.main decorator
        """
        self._config = config
        
        # Resolve task-specific overrides if task_name is specified
        if hasattr(config, 'task_name') and config.task_name:
            self._apply_task_specific_overrides()
            
        # Validate critical configuration values
        self._validate_config()

    def _apply_task_specific_overrides(self) -> None:
        """Apply task-specific configuration overrides (e.g., more layers)."""
        task_name = self._config.task_name
        
        # Handle underscore vs space in task names
        clean_task_name = task_name.replace(" ", "_")
        
        if hasattr(self._config.task_specific, clean_task_name):
            task_cfg = getattr(self._config.task_specific, clean_task_name)
            
            # Apply model overrides
            if "num_layers" in task_cfg:
                self._config.model.num_layers = task_cfg["num_layers"]
                
            # Add other task-specific overrides here as needed
    
    def _validate_config(self) -> None:
        """Validate configuration values are within expected ranges."""
        # Validate training settings
        assert self._config.training.batch_size > 0, "Batch size must be positive"
        assert 0 < self._config.training.validation_split < 1, "Validation split must be between 0 and 1"
        
        # Validate model settings
        assert self._config.model.token_dim > 0, "Token dimension must be positive"
        assert self._config.model.num_layers > 0, "Number of layers must be positive"
        assert self._config.model.num_heads > 0, "Number of heads must be positive"
        
        # Validate SDE settings
        sde_type = self._config.sde.type
        assert sde_type in ["VESDE", "VPSDE"], f"Unknown SDE type: {sde_type}"
        
        if sde_type == "VESDE":
            vesde = self._config.sde.vesde
            assert vesde.sigma_min > 0, "VESDE sigma_min must be positive"
            assert vesde.sigma_max > vesde.sigma_min, "VESDE sigma_max must exceed sigma_min"
            assert 0 <= vesde.t_min < vesde.t_max <= 1, "VESDE time range invalid"
        else:  # VPSDE
            vpsde = self._config.sde.vpsde
            assert vpsde.beta_min >= 0, "VPSDE beta_min must be non-negative"
            assert vpsde.beta_max > vpsde.beta_min, "VPSDE beta_max must exceed beta_min"
            assert 0 <= vpsde.t_min < vpsde.t_max <= 1, "VPSDE time range invalid"
        
        # Validate sampling settings
        assert self._config.sampling.num_steps > 0, "Number of sampling steps must be positive"
        assert self._config.sampling.self_recurrence_steps >= 0, "Self-recurrence steps cannot be negative"
        
        # Validate masking settings
        strategies = self._config.masking.condition_mask_sampling.strategies
        probs = self._config.masking.condition_mask_sampling.probabilities
        assert len(strategies) == len(probs), "Masking strategies and probabilities length mismatch"
        assert abs(sum(probs) - 1.0) < 1e-6, "Masking probabilities must sum to 1"
        assert self._config.masking.attention_mask_type in ["undirected", "directed"], \
            "Attention mask type must be 'undirected' or 'directed'"
    
    def get_config(self) -> DictConfig:
        """
        Get the complete configuration object.
        
        Returns:
            Hydra DictConfig object with all resolved settings
        """
        return self._config
    
    def save_config(self, output_dir: str) -> None:
        """
        Save the current configuration to a YAML file in the output directory.
        
        Args:
            output_dir: Directory where configuration should be saved
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration as YAML
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, 'w') as f:
            OmegaConf.save(self._config, f)
        
        # Also save as pretty-printed version for readability
        pretty_path = os.path.join(output_dir, "config.pretty.yaml")
        with open(pretty_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(self._config, resolve=True), 
                     f, default_flow_style=False, indent=2)

    def print_config(self) -> None:
        """Print the configuration in a readable format."""
        print("=== Simformer Configuration ===")
        print(OmegaConf.to_yaml(self._config))
        print("================================")

    def get_simulation_budget(self) -> int:
        """Get the simulation budget setting."""
        return getattr(self._config, 'simulation_budget', 10000)

    def get_task_name(self) -> str:
        """Get the task name."""
        return getattr(self._config, 'task_name', 'default')


# Global variable to hold the configuration (set by Hydra)
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    This function should be called after the ConfigManager has been initialized
    by the @hydra.main decorator in main.py.
    
    Returns:
        ConfigManager instance
        
    Raises:
        RuntimeError: If called before configuration is loaded
    """
    if _global_config_manager is None:
        raise RuntimeError("Configuration not yet loaded. Call this after @hydra.main initialization.")
    return _global_config_manager


# The @hydra.main decorator will automatically call this function when the script runs
@hydra.main(config_path=".", config_name="config")
def load_config(config: DictConfig) -> None:
    """
    Load configuration using Hydra and initialize the global ConfigManager.
    
    This function is automatically called by Hydra when the script runs.
    It sets up the global configuration manager that can be accessed via get_config_manager().
    
    Args:
        config: Hydra DictConfig object loaded from config.yaml and CLI overrides
    """
    global _global_config_manager
    _global_config_manager = ConfigManager(config)
    
    # Print configuration for debugging
    _global_config_manager.print_config()


# Make the load_config function available at module level
__all__ = ['ConfigManager', 'get_config_manager', 'load_config']
