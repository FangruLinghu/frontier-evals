## optimizer/optimizer.py
```python
"""Optimizer module for similarity-guided diffusion model adaptation.

This module implements the optimizer setup for adaptor parameters ψ as described
in Algorithm 1 of the paper. The optimizer only updates the adaptor parameters ψ
while keeping the frozen pre-trained U-Net θ unchanged.

Key components:
- DiffusionOptimizer: Wrapper for PyTorch optimizers (Adam/SGD) for adaptor parameters
- OptimizerFactory: Factory to create optimizer from model, extracting only ψ parameters
- LRScheduler: Learning rate scheduling (constant, linear_decay, cosine, step)
- Helper functions: get_default_optimizer_config, extract_adaptor_parameters

The optimizer performs the update:
    ψ = ψ - η∇_ψ L(ψ)
as per Algorithm 1 line 6.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from typing import Iterator, Optional, Tuple


class DiffusionOptimizer:
    """Optimizer wrapper for adaptor parameters ψ in similarity-guided diffusion training.
    
    This class wraps PyTorch optimizers (Adam or SGD) for updating only the adaptor
    parameters ψ while keeping the frozen pre-trained U-Net θ unchanged.
    
    The update rule follows Algorithm 1 line 6:
        ψ = ψ - η∇_ψ L(ψ)
    
    Attributes:
        optimizer: PyTorch optimizer instance (Adam or SGD) for adaptor parameters
        lr: Learning rate η from Algorithm 1
        optim_type: Optimizer type: 'adam' or 'sgd'
    """
    
    def __init__(
        self,
        adaptor_parameters: Iterator[nn.Parameter],
        learning_rate: float,
        optim_type: str = 'adam',
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        momentum: float = 0.9
    ) -> None:
        """Initialize optimizer for adaptor parameters ψ.
        
        Only ψ parameters are updated while frozen pre-trained U-Net θ remains unchanged.
        
        Args:
            adaptor_parameters: Iterator over adaptor parameters ψ to optimize
            learning_rate: Learning rate η from Algorithm 1
            optim_type: Optimizer type: 'adam' or 'sgd' (default: 'adam')
            betas: Adam exponential decay rates for moment estimates (default: (0.9, 0.999))
            weight_decay: Weight decay coefficient (default: 0.0, no weight decay as per paper)
            momentum: SGD momentum (default: 0.9)
        
        Raises:
            ValueError: If optim_type is not 'adam' or 'sgd'
        """
        self.lr = learning_rate
        self.optim_type = optim_type.lower()
        
        # Convert iterator to list for multiple passes
        params_list = list(adaptor_parameters)
        
        if self.optim_type == 'adam':
            # Adam optimizer defaults: betas=(0.9, 0.999), eps=1e-8
            self.optimizer = Adam(
                params_list,
                lr=learning_rate,
                betas=betas,
                eps=1e-8,
                weight_decay=weight_decay
            )
        elif self.optim_type == 'sgd':
            self.optimizer = SGD(
                params_list,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(
                f"Invalid optim_type: {optim_type}. Must be 'adam' or 'sgd'"
            )
    
    def zero_grad(self) -> None:
        """Clear gradients of all optimized adaptor parameters.
        
        Called before loss.backward() in training loop.
        Clears the gradients of ψ parameters only, leaving θ (U-Net) unchanged.
        """
        self.optimizer.zero_grad()
    
    def step(self) -> None:
        """Perform a single optimization step.
        
        Updates ψ = ψ - η∇_ψ L(ψ) as per Algorithm 1 line 6.
        This updates only the adaptor parameters while keeping the frozen
        pre-trained U-Net θ unchanged.
        """
        self.optimizer.step()
    
    def state_dict(self) -> dict:
        """Returns the state of the optimizer as a dict.
        
        Includes tensor states for potential checkpointing.
        Useful for saving and resuming training.
        
        Returns:
            Dictionary containing optimizer state including state tensors
        """
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the optimizer state from a dict.
        
        Useful for resuming training from a checkpoint.
        
        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.optimizer.load_state_dict(state_dict)
    
    def get_lr(self) -> float:
        """Returns the current learning rate.
        
        Returns:
            Current learning rate η
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def set_lr(self, lr: float) -> None:
        """Sets the learning rate to a new value.
        
        Useful for learning rate scheduling during training.
        
        Args:
            lr: New learning rate value
        """
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class OptimizerFactory:
    """Factory class for creating optimizers for adaptor parameters.
    
    Provides methods to create DiffusionOptimizer or directly create PyTorch
    optimizers from a model, automatically extracting only adaptor parameters ψ.
    """
    
    def __init__(self) -> None:
        """Initialize optimizer factory."""
        pass
    
    def create_optimizer(
        self,
        model: nn.Module,
        learning_rate: float,
        optim_type: str = 'adam',
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9
    ) -> DiffusionOptimizer:
        """Create DiffusionOptimizer from a model.
        
        Extracts only adaptor parameters ψ from the model for optimization.
        
        Args:
            model: Model containing adaptor parameters ψ
            learning_rate: Learning rate η for updating ψ
            optim_type: Optimizer type: 'adam' or 'sgd' (default: 'adam')
            weight_decay: Weight decay coefficient (default: 0.0)
            betas: Adam betas (default: (0.9, 0.999))
            momentum: SGD momentum (default: 0.9)
        
        Returns:
            DiffusionOptimizer instance configured for adaptor parameters
        
        Example:
            >>> from model.diffusion_model import DiffusionUNet
            >>> from config import create_toy_config
            >>> config = create_toy_config()
            >>> model = DiffusionUNet(config)
            >>> optimizer_factory = OptimizerFactory()
            >>> optimizer = optimizer_factory.create_optimizer(
            ...     model=model,
            ...     learning_rate=5e-5,
            ...     optim_type='adam'
            ... )
        """
        # Extract adaptor parameters from model
        adaptor_params = extract_adaptor_parameters(model)
        
        # Create DiffusionOptimizer
        return DiffusionOptimizer(
            adaptor_parameters=adaptor_params,
            learning_rate=learning_rate,
            optim_type=optim_type,
            betas=betas,
            weight_decay=weight_decay,
            momentum=momentum
        )
    
    def create_adam_optimizer(
        self,
        model: nn.Module,
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0
    ) -> Adam:
        """Creates Adam optimizer for adaptor parameters.
        
        Adam defaults: betas=(0.9, 0.999), eps=1e-8 as per paper.
        
        Args:
            model: Model containing adaptor parameters ψ
            learning_rate: Learning rate η
            betas: Adam exponential decay rates (default: (0.9, 0.999))
            weight_decay: Weight decay coefficient (default: 0.0)
        
        Returns:
            torch.optim.Adam optimizer for adaptor parameters
        
        Example:
            >>> optimizer_factory = OptimizerFactory()
            >>> adam_opt = optimizer_factory.create_adam_optimizer(
            ...     model=model,
            ...     learning_rate=5e-5,
            ...     betas=(0.9, 0.999)
            ... )
        """
        adaptor_params = extract_adaptor_parameters(model)
        
        return Adam(
            list(adaptor_params),
            lr=learning_rate,
            betas=betas,
            eps=1e-8,
            weight_decay=weight_decay
        )
    
    def create_sgd_optimizer(
        self,
        model: nn.Module,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0
    ) -> SGD:
        """Creates SGD optimizer for adaptor parameters with momentum.
        
        Args:
            model: Model containing adaptor parameters ψ
            learning_rate: Learning rate η
            momentum: SGD momentum (default: 0.9)
            weight_decay: Weight decay coefficient (default: 0.0)
        
        Returns:
            torch.optim.SGD optimizer for adaptor parameters
        """
        adaptor_params = extract_adaptor_parameters(model)
        
        return SGD(
            list(adaptor_params),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )


class LRScheduler:
    """Learning rate scheduler for adaptor parameter optimization.
    
    Supports multiple scheduling strategies:
    - constant: Keep learning rate constant throughout training
    - linear_decay: Linear decay from initial_lr to min_lr
    - cosine: Cosine annealing from initial_lr to min_lr
    - step: Step decay at specified milestones
    
    Attributes:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduling strategy
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps (for learning rate warmup)
        min_lr: Minimum learning rate (default: 1e-6)
    """
    
    def __init__(
        self,
        optimizer: DiffusionOptimizer,
        scheduler_type: str = 'constant',
        total_steps: int = 1000,
        warmup_steps: int = 0,
        min_lr: float = 1e-6
    ) -> None:
        """Initialize learning rate scheduler.
        
        Args:
            optimizer: DiffusionOptimizer to schedule
            scheduler_type: Type of scheduling:
                - 'constant': Keep learning rate constant
                - 'linear_decay': Linear decay to min_lr
                - 'cosine': Cosine annealing to min_lr
                - 'step': Step decay at milestones
            total_steps: Total number of training steps for scheduling
            warmup_steps: Number of warmup steps (default: 0)
            min_lr: Minimum learning rate (default: 1e-6)
        
        Raises:
            ValueError: If scheduler_type is invalid
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type.lower()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.get_lr()
        self.current_step = 0
        
        # Validate scheduler_type
        valid_types = ['constant', 'linear_decay', 'cosine', 'step']
        if self.scheduler_type not in valid_types:
            raise ValueError(
                f"Invalid scheduler_type: {scheduler_type}. "
                f"Must be one of {valid_types}"
            )
    
    def step(self, step: Optional[int] = None) -> float:
        """Update learning rate based on current step.
        
        Computes new learning rate based on scheduler_type and current step.
        
        Args:
            step: Current training step. If None, uses internal counter.
        
        Returns:
            New learning rate after the update
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Compute new learning rate based on scheduler type
        if self.scheduler_type == 'constant':
            new_lr = self._constant_schedule()
        elif self.scheduler_type == 'linear_decay':
            new_lr = self._linear_decay_schedule()
        elif self.scheduler_type == 'cosine':
            new_lr = self._cosine_schedule()
        elif self.scheduler_type == 'step':
            new_lr = self._step_schedule()
        else:
            new_lr = self.initial_lr
        
        # Apply warmup if applicable
        if self.current_step < self.warmup_steps:
            warmup_lr = self.initial_lr * (self.current_step / self.warmup_steps)
            new_lr = min(warmup_lr, new_lr)
        
        # Update optimizer learning rate
        self.optimizer.set_lr(new_lr)
        
        return new_lr
    
    def _constant_schedule(self) -> float:
        """Constant learning rate schedule.
        
        Returns:
            Initial learning rate (constant throughout training)
        """
        return self.initial_lr
    
    def _linear_decay_schedule(self) -> float:
        """Linear decay learning rate schedule.
        
        Decays linearly from initial_lr to min_lr over total_steps.
        
        Returns:
            Learning rate after linear decay
        """
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.initial_lr - (self.initial_lr - self.min_lr) * progress
    
    def _cosine_schedule(self) -> float:
        """Cosine annealing learning rate schedule.
        
        Uses cosine annealing from initial_lr to min_lr.
        
        Returns:
            Learning rate after cosine annealing
        """
        import math
        progress = min(self.current_step / self.total_steps, 1.0)
        # Cosine annealing: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * progress))
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
    
    def _step_schedule(self) -> float:
        """Step decay learning rate schedule.
        
        Decays learning rate by half at 50% and 75% of total_steps.
        
        Returns:
            Learning rate after step decay
        """
        milestones = [0.5, 0.75]  # Decay at 50% and 75%
        decay_factor = 0.5
        
        new_lr = self.initial_lr
        for milestone in milestones:
            if self.current_step >= self.total_steps * milestone:
                new_lr *= decay_factor
        
        # Ensure we don't go below min_lr
        new_lr = max(new_lr, self.min_lr)
        
        return new_lr
    
    def get_last_lr(self) -> list:
        """Returns list of current learning rates for each parameter group.
        
        Returns:
            List containing current learning rate (for compatibility with PyTorch)
        """
        return [self.optimizer.get_lr()]


def get_default_optimizer_config(model_type: str = 'ddpm') -> dict:
    """Returns default optimizer configuration based on model type.
    
    Based on Section 5.1 of the paper:
    - DDPMs: lr=5e-5
    - LDMs: lr=1e-5
    
    Uses Adam optimizer by default with paper-recommended settings:
    - betas=(0.9, 0.999), eps=1e-8
    - weight_decay=0.0 (no weight decay as per paper experiments)
    
    Args:
        model_type: Type of diffusion model ('ddpm' or 'ldm')
    
    Returns:
        Dictionary containing default optimizer configuration:
            - learning_rate: η (5e-5 for DDPM, 1e-5 for LDM)
            - optim_type: 'adam'
            - betas: (0.9, 0.999)
            - weight_decay: 0.0
            - momentum: 0.9
    
    Example:
        >>> # Get default config for DDPM
        >>> config = get_default_optimizer_config('ddpm')
        >>> print(config['learning_rate'])  # 5e-5
        >>>
        >>> # Get default config for LDM
        >>> config = get_default_optimizer_config('ldm')
        >>> print(config['learning_rate'])  # 1e-5
    """
    model_type = model_type.lower()
    
    if model_type == 'ddpm':
        learning_rate = 5e-5  # 5×10^-5 for DDPMs as per paper Section 5.1
    elif model_type == 'ldm':
        learning_rate = 1e-5  # 1×10^-5 for LDMs as per paper Section 5.1
    else:
        # Default to DDPM settings for unknown types
        learning_rate = 5e-5
    
    return {
        'learning_rate': learning_rate,
        'optim_type': 'adam',
        'betas': (0.9, 0.999),
        'weight_decay': 0.0,  # No weight decay as per paper experiments
        'momentum': 0.9
    }


def extract_adaptor_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Extracts only adaptor parameters ψ from the model.
    
    Filters model.parameters() to only include ψ parameters (adaptor layers),
    excluding frozen pre-trained U-Net parameters θ.
    
    Uses parameter name containing 'adapter' or 'adaptor' to identify
    adaptable parameters.
    
    Args:
        model: Model containing both adaptor parameters ψ and frozen U-Net parameters θ
    
    Yields:
        Iterator over adaptor parameters ψ only
    
    Example:
        >>> # Extract only adaptor parameters for optimization
        >>> adaptor_params = extract_adaptor_parameters(model)
        >>> optimizer = DiffusionOptimizer(adaptor_params, learning_rate=5e-5)
        >>>
        >>> # Verify only adaptor params are optimized
        >>> for name, param in model.named_parameters():
        ...     if 'adapter' in name or 'adaptor' in name:
        ...         assert param.requires_grad, f"{name} should be trainable"
        ...     else:
        ...         assert not param.requires_grad, f"{name} should be frozen"
    """
    for name, param in model.named_parameters():
        # Check if parameter name contains 'adapter' or 'adaptor'
        # This identifies parameters belonging to adaptor layers ψ
        if 'adapter' in name.lower() or 'adaptor' in name.lower():
            yield param
        # Also check for 'adapt' as a substring for flexibility
        elif 'adapt' in name.lower():
            yield param