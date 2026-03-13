Looking at the architecture and the review resolutions, I need to create `data/loader.py` with a function `get_few_shot_dataloader(config, data_dir, batch_size=40)` that creates DataLoaders for few-shot training experiments.

Let me examine the existing data modules and create the loader:

```python
## data/loader.py
"""Data loader for few-shot training in similarity-guided diffusion model adaptation.

This module implements data loading utilities for both toy experiments (2D Gaussian)
and FFHQ real image experiments. It provides DataLoaders that support the few-shot
setting described in the paper, where only a small number of target domain samples
(typically 10) are available for adaptation.

Key functions:
    get_few_shot_dataloader: Main entry point for creating few-shot data loaders
    create_toy_dataloaders: Create source/target loaders for toy experiments
    create_ffhq_dataloaders: Create source/target loaders for FFHQ experiments

The data loader handles:
    - Source domain: Unlimited sampling from source distribution
    - Target domain: Few-shot sampling with replacement (10 samples in paper)
    - Batched loading with proper shuffling for source
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import Optional, Tuple, Literal, Union
from pathlib import Path

from config import Config
from data.toy_data import ToyDataset, create_toy_source_dataset, create_toy_target_dataset
from data.ffhq import FFHQDataset, create_ffhq_sketches_dataset, create_ffhq_amedeo_dataset


def get_few_shot_dataloader(
    config: Config,
    data_dir: Optional[str] = None,
    batch_size: int = 40,
    num_workers: int = 0,
    shuffle_source: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for few-shot training.
    
    This is the main entry point for creating data loaders. It automatically
    determines the experiment type (toy vs FFHQ) based on configuration and
    creates appropriate source/target data loaders.
    
    For toy experiments:
        - Source: 2D Gaussian with mean (1,1), variance I
        - Target: 2D Gaussian with mean (-1,-1), variance I (few_shot_samples=10)
    
    For FFHQ experiments:
        - Source: FFHQ face images (real photos)
        - Target: Sketches or Amedeo paintings (few_shot_samples=10)
    
    Args:
        config: Configuration object with experiment parameters.
               Must contain:
               - few_shot_samples: Number of target samples (default: 10)
               - source_mean: Source distribution mean
               - target_mean: Target distribution mean
        data_dir: Root directory for data (required for FFHQ, optional for toy)
                 Expected structure for FFHQ:
                 - data_dir/ffhq/ (source images)
                 - data_dir/sketches/ (target images)
                 - data_dir/amedeo/ (target images)
        batch_size: Batch size for training (default: 40 from paper)
        num_workers: Number of worker processes for data loading (default: 0)
        shuffle_source: Whether to shuffle source data (default: True)
    
    Returns:
        Tuple of (source_loader, target_loader):
            - source_loader: DataLoader for source domain (unlimited sampling)
            - target_loader: DataLoader for target domain (few-shot with replacement)
    
    Raises:
        ValueError: If data_dir is required but not provided
        FileNotFoundError: If FFHQ data directory not found
    
    Example:
        >>> from config import create_toy_config
        >>> config = create_toy_config()
        >>> source_loader, target_loader = get_few_shot_dataloader(
        ...     config=config,
        ...     data_dir=None,  # Not needed for toy
        ...     batch_size=40
        ... )
        >>>
        >>> # FFHQ example
        >>> from config import create_ffhq_config
        >>> config = create_ffhq_config(target_domain='sketches')
        >>> source_loader, target_loader = get_few_shot_dataloader(
        ...     config=config,
        ...     data_dir='./data',
        ...     batch_size=40
        ... )
    """
    # Determine experiment type based on source_mean dimension
    source_mean = config.source_mean
    
    if isinstance(source_mean, tuple) and len(source_mean) == 2:
        # Toy experiment (2D Gaussian)
        return create_toy_dataloaders(
            config=config,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_source=shuffle_source
        )
    elif isinstance(source_mean, tuple) and len(source_mean) == 3:
        # FFHQ experiment (RGB images)
        if data_dir is None:
            raise ValueError(
                "data_dir is required for FFHQ experiments. "
                "Please provide the path to FFHQ data directory."
            )
        return create_ffhq_dataloaders(
            config=config,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_source=shuffle_source
        )
    else:
        raise ValueError(
            f"Invalid source_mean format: {source_mean}. "
            f"Expected tuple of length 2 (toy) or 3 (FFHQ)."
        )


def create_toy_dataloaders(
    config: Config,
    batch_size: int = 40,
    num_workers: int = 0,
    shuffle_source: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for toy experiment (2D Gaussian).
    
    Creates source and target data loaders for the toy experiment:
    - Source: Gaussian with mean (1,1), variance I
    - Target: Gaussian with mean (-1,-1), variance I
    
    The source loader provides unlimited sampling, while the target loader
    uses only few_shot_samples with replacement (sampling with replacement
    to handle small dataset).
    
    Args:
        config: Configuration object with toy experiment parameters:
            - source_mean: (1, 1) for toy
            - target_mean: (-1, -1) for toy
            - few_shot_samples: Number of target samples (default: 10)
            - source_variance: Variance of Gaussian (default: 1.0)
        batch_size: Batch size for training (default: 40)
        num_workers: Number of worker processes (default: 0)
        shuffle_source: Whether to shuffle source data (default: True)
    
    Returns:
        Tuple of (source_loader, target_loader):
            - source_loader: Unlimited source sampling
            - target_loader: Few-shot target sampling with replacement
    
    Example:
        >>> from config import create_toy_config
        >>> config = create_toy_config()
        >>> source_loader, target_loader = create_toy_dataloaders(
        ...     config=config,
        ...     batch_size=40
        ... )
        >>> print(f"Source batches: {len(source_loader)}")
        >>> print(f"Target samples: {len(target_loader.dataset)}")  # 10
    """
    # Extract configuration
    source_mean = config.source_mean
    target_mean = config.target_mean
    few_shot_samples = config.few_shot_samples
    variance = config.source_variance
    
    # Create source dataset (unlimited sampling)
    # Use a large number of samples for source to approximate unlimited
    source_num_samples = max(10000, few_shot_samples * 100)
    source_dataset = ToyDataset(
        mean=source_mean,
        num_samples=source_num_samples,
        source=True,
        variance=variance
    )
    
    # Create target dataset (few-shot)
    target_dataset = ToyDataset(
        mean=target_mean,
        num_samples=few_shot_samples,
        source=False,
        variance=variance
    )
    
    # Create source DataLoader
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=shuffle_source,
        num_workers=num_workers,
        drop_last=True
    )
    
    # Create target DataLoader
    # Use drop_last=False to handle cases where batch_size > few_shot_samples
    # Use sampler with replacement since we have few samples
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for randomness in few-shot samples
        num_workers=num_workers,
        drop_last=False,
        # For very small datasets, use random sampling with replacement
        sampler=torch.utils.data.RandomSampler(
            target_dataset, 
            replacement=True, 
            num_samples=batch_size * 100  # Generate enough samples for training
        ) if few_shot_samples < batch_size else None
    )
    
    return source_loader, target_loader


def create_ffhq_dataloaders(
    config: Config,
    data_dir: str,
    batch_size: int = 40,
    num_workers: int = 4,
    shuffle_source: bool = True,
    image_size: int = 256
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for FFHQ experiment.
    
    Creates source and target data loaders for FFHQ domain adaptation:
    - Source: FFHQ face images (real photos)
    - Target: Sketches or Amedeo paintings
    
    Args:
        config: Configuration object with FFHQ experiment parameters:
            - few_shot_samples: Number of target samples (default: 10)
            - model_type: 'ddpm' or 'ldm' (affects image processing)
        data_dir: Root directory containing FFHQ and target domain images.
                 Expected structure:
                 - data_dir/ffhq/ (source images)
                 - data_dir/sketches/ (target images)
                 - data_dir/amedeo/ (target images)
        batch_size: Batch size for training (default: 40)
        num_workers: Number of worker processes (default: 4)
        shuffle_source: Whether to shuffle source data (default: True)
        image_size: Size to resize images to (default: 256)
    
    Returns:
        Tuple of (source_loader, target_loader):
            - source_loader: Unlimited source sampling from FFHQ
            - target_loader: Few-shot target sampling with replacement
    
    Raises:
        FileNotFoundError: If required data directories are not found
        ValueError: If target domain is invalid
    
    Example:
        >>> from config import create_ffhq_config
        >>> config = create_ffhq_config(target_domain='sketches')
        >>> source_loader, target_loader = create_ffhq_dataloaders(
        ...     config=config,
        ...     data_dir='./data',
        ...     batch_size=40
        ... )
        >>> print(f"Source batches: {len(source_loader)}")
        >>> print(f"Target samples: {len(target_loader.dataset)}")  # 10
    """
    # Determine target domain from config
    # The config is created with either 'sketches' or 'amedeo' domain
    # We can infer from the save_dir or add explicit domain parameter
    few_shot_samples = config.few_shot_samples
    
    # Determine target domain from save_dir if available
    save_dir = getattr(config, 'save_dir', './checkpoints')
    if 'sketches' in save_dir.lower():
        target_domain = 'sketches'
    elif 'amedeo' in save_dir.lower():
        target_domain = 'amedeo'
    else:
        # Default to sketches
        target_domain = 'sketches'
    
    # Create source dataset (FFHQ - unlimited)
    source_dataset = FFHQDataset(
        data_dir=data_dir,
        few_shot=few_shot_samples,
        domain=target_domain,
        image_size=image_size,
        is_source=True
    )
    
    # Create target dataset (few-shot)
    target_dataset = FFHQDataset(
        data_dir=data_dir,
        few_shot=few_shot_samples,
        domain=target_domain,
        image_size=image_size,
        is_source=False
    )
    
    # Check if data was loaded successfully
    if len(source_dataset) == 0:
        print(f"Warning: No source images found in {data_dir}/ffhq/")
    
    if len(target_dataset) == 0:
        print(f"Warning: No target images found in {data_dir}/{target_domain}/")
    
    # Create source DataLoader
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=shuffle_source,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create target DataLoader
    # Use random sampling with replacement for few-shot target
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        # Sample with replacement to handle small dataset
        sampler=torch.utils.data.RandomSampler(
            target_dataset,
            replacement=True,
            num_samples=batch_size * 100  # Enough samples for training
        ) if len(target_dataset) < batch_size else None
    )
    
    return source_loader, target_loader


class FewShotDataIterator:
    """Iterator for few-shot training with source and target data.
    
    This class provides an iterator interface for few-shot training,
    handling the case where source data is unlimited but target data
    is limited to few_shot_samples.
    
    Attributes:
        source_loader: DataLoader for source domain
        target_loader: DataLoader for target domain
        source_iter: Iterator over source loader
        target_iter: Iterator over target loader
    """
    
    def __init__(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader
    ) -> None:
        """Initialize few-shot data iterator.
        
        Args:
            source_loader: DataLoader for source domain
            target_loader: DataLoader for target domain
        """
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iter = iter(source_loader)
        self.target_iter = iter(target_loader)
    
    def __iter__(self) -> 'FewShotDataIterator':
        """Return iterator object."""
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of source and target data.
        
        Returns:
            Tuple of (source_batch, target_batch):
                - source_batch: [batch_size, ...] source domain images
                - target_batch: [batch_size, ...] target domain images
        
        Raises:
            StopIteration: When source data is exhausted
        """
        try:
            source_batch = next(self.source_iter)
        except StopIteration:
            # Reset source iterator when exhausted
            self.source_iter = iter(self.source_loader)
            source_batch = next(self.source_iter)
        
        try:
            target_batch = next(self.target_iter)
        except StopIteration:
            # Reset target iterator when exhausted
            self.target_iter = iter(self.target_loader)
            target_batch = next(self.target_iter)
        
        return source_batch, target_batch
    
    def get_source_batch(self) -> torch.Tensor:
        """Get next source batch, resetting iterator if needed.
        
        Returns:
            Source batch tensor
        """
        try:
            return next(self.source_iter)
        except StopIteration:
            self.source_iter = iter(self.source_loader)
            return next(self.source_iter)
    
    def get_target_batch(self) -> torch.Tensor:
        """Get next target batch, resetting iterator if needed.
        
        Returns:
            Target batch tensor
        """
        try:
            return next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.target_loader)
            return next(self.target_iter)


def create_few_shot_iterator(
    config: Config,
    data_dir: Optional[str] = None,
    batch_size: int = 40,
    num_workers: int = 0
) -> FewShotDataIterator:
    """Create FewShotDataIterator for training.
    
    Convenience function to create an iterator that provides
    source and target batches for few-shot training.
    
    Args:
        config: Configuration object
        data_dir: Root directory for data (required for FFHQ)
        batch_size: Batch size for training
        num_workers: Number of worker processes
    
    Returns:
        FewShotDataIterator instance
    
    Example:
        >>> config = create_toy_config()
        >>> iterator = create_few_shot_iterator(config, batch_size=40)
        >>> 
        >>> for i in range(100):
        ...     source_batch, target_batch = next(iterator)
        ...     # Training step with both batches
    """
    source_loader, target_loader = get_few_shot_dataloader(
        config=config,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return FewShotDataIterator(source_loader, target_loader)


def get_dataloader_for_evaluation(
    config: Config,
    data_dir: str,
    domain: Literal['source', 'target'] = 'target',
    batch_size: int = 50,
    num_workers: int = 4
) -> DataLoader:
    """Create DataLoader for evaluation (source or target only).
    
    Creates a DataLoader for either source or target domain for evaluation
    purposes (e.g., computing Intra-LPIPS).
    
    Args:
        config: Configuration object
        data_dir: Root directory for data
        domain: 'source' or 'target' domain
        batch_size: Batch size for evaluation
        num_workers: Number of worker processes
    
    Returns:
        DataLoader for specified domain
    
    Example:
        >>> # Get target samples for evaluation
        >>> config = create_ffhq_config('sketches')
        >>> target_loader = get_dataloader_for_evaluation(
        ...     config=config,
        ...     data_dir='./data',
        ...     domain='target',
        ...     batch_size=50
        ... )
    """
    is_source = (domain == 'source')
    
    # Determine target domain
    save_dir = getattr(config, 'save_dir', './checkpoints')
    if 'sketches' in save_dir.lower():
        target_domain = 'sketches'
    elif 'amedeo' in save_dir.lower():
        target_domain = 'amedeo'
    else:
        target_domain = 'sketches'
    
    # Create appropriate dataset
    if isinstance(config.source_mean, tuple) and len(config.source_mean) == 2:
        # Toy experiment
        mean = config.source_mean if is_source else config.target_mean
        dataset = ToyDataset(
            mean=mean,
            num_samples=1000,  # Use more samples for evaluation
            source=is_source,
            variance=config.source_variance
        )
    else:
        # FFHQ experiment
        image_size = 256  # Default for FFHQ
        dataset = FFHQDataset(
            data_dir=data_dir,
            few_shot=config.few_shot_samples,
            domain=target_domain,
            image_size=image_size,
            is_source=is_source
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(domain == 'source'),  # Only shuffle source for randomness
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )