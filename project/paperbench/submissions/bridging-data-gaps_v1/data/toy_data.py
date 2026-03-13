## data/toy_data.py

```python
"""Toy data module for similarity-guided diffusion model experiments.

This module implements the 2D Gaussian toy dataset used in the paper's Section 5.1
for evaluating the similarity-guided diffusion model adaptation method.

The toy dataset consists of:
- Source distribution: 2D Gaussian with mean (1, 1) and variance I (identity matrix)
- Target distribution: 2D Gaussian with mean (-1, -1) and variance I (identity matrix)

In the few-shot setting (used in paper experiments), only 10 samples from the target
distribution are available for adaptation, while the source distribution can be sampled
unlimitedly for training.

The 2D toy data provides a simple testbed to verify:
1. The model learns to transform source-like outputs toward target distribution
2. The adaptor layers can efficiently adapt with few target samples
3. The similarity-guided loss correctly guides the generation process
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple
from torch import Tensor


class ToyDataset(Dataset):
    """2D Gaussian toy dataset for few-shot domain adaptation experiments.
    
    This dataset generates samples from a 2D isotropic Gaussian distribution,
    either from the source domain (mean=(1,1)) or target domain (mean=(-1,-1)).
    
    Used in paper Section 5.1 "Toy Experiment" to validate the similarity-guided
    adaptation approach with minimal complexity before applying to real images.
    
    Attributes:
        mean: Mean vector of the 2D Gaussian distribution
        num_samples: Total number of samples in the dataset
        source: Boolean flag indicating if this is source (True) or target (False) domain
        variance: Variance of the isotropic Gaussian (default: 1.0 for identity covariance)
        data: Pre-generated sample tensor of shape [num_samples, 2]
    
    Example:
        >>> # Source dataset (unlimited sampling)
        >>> source_ds = ToyDataset(mean=(1, 1), num_samples=1000, source=True)
        >>> 
        >>> # Target dataset (few-shot: only 10 samples as per paper)
        >>> target_ds = ToyDataset(mean=(-1, -1), num_samples=10, source=False)
        >>> 
        >>> # Sample from source
        >>> x = source_ds[0]  # Shape: [2]
    """
    
    def __init__(
        self,
        mean: Tuple[float, float],
        num_samples: int,
        source: bool,
        variance: float = 1.0
    ) -> None:
        """Initialize the 2D Gaussian toy dataset.
        
        Args:
            mean: Mean vector of the 2D Gaussian (source: (1,1), target: (-1,-1))
            num_samples: Number of samples to generate
                  - For source domain: can be any number (paper uses large number)
                  - For target domain: typically 10 (few-shot setting from paper)
            source: Boolean flag:
                - True: source domain distribution (mean=(1,1))
                - False: target domain distribution (mean=(-1,-1))
            variance: Variance of isotropic Gaussian (default: 1.0 for identity covariance I)
        
        Raises:
            ValueError: If variance is not positive
        """
        if variance <= 0:
            raise ValueError(f"Variance must be positive, got {variance}")
        
        self.mean = mean
        self.num_samples = num_samples
        self.source = source
        self.variance = variance
        
        # Pre-generate all samples at initialization
        # Using standard deviation = sqrt(variance)
        std = torch.sqrt(torch.tensor(variance))
        
        # Generate samples from N(mean, variance * I)
        # Shape: [num_samples, 2]
        self.data = torch.randn(num_samples, 2) * std + torch.tensor(mean)
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.
        
        Returns:
            Number of samples (num_samples)
        """
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tensor:
        """Get a single sample from the dataset by index.
        
        Args:
            idx: Index of the sample to retrieve (0 <= idx < num_samples)
        
        Returns:
            2D tensor of shape [2] containing the sample coordinates (x, y)
        
        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {self.num_samples}"
            )
        return self.data[idx]
    
    def get_distribution_params(self) -> dict:
        """Get the distribution parameters for this dataset.
        
        Returns:
            Dictionary containing mean, variance, and source flag
        """
        return {
            'mean': self.mean,
            'variance': self.variance,
            'source': self.source
        }


def create_toy_source_dataset(num_samples: int = 1000) -> ToyDataset:
    """Factory function to create the source domain toy dataset.
    
    Creates a dataset with source distribution N((1,1), I) as described
    in the paper's toy experiment (Section 5.1).
    
    Args:
        num_samples: Number of samples to generate (default: 1000 for training)
    
    Returns:
        ToyDataset instance for source domain
    
    Example:
        >>> source_ds = create_toy_source_dataset(num_samples=5000)
        >>> print(len(source_ds))  # 5000
    """
    return ToyDataset(
        mean=(1.0, 1.0),
        num_samples=num_samples,
        source=True,
        variance=1.0
    )


def create_toy_target_dataset(num_samples: int = 10) -> ToyDataset:
    """Factory function to create the target domain toy dataset.
    
    Creates a dataset with target distribution N((-1,-1), I) as described
    in the paper's toy experiment (Section 5.1).
    
    Uses the few-shot setting with only 10 samples, which is the key
    challenge that the similarity-guided adaptation method addresses.
    
    Args:
        num_samples: Number of samples to generate (default: 10 for few-shot)
    
    Returns:
        ToyDataset instance for target domain
    
    Example:
        >>> target_ds = create_toy_target_dataset()  # Default 10 samples
        >>> print(len(target_ds))  # 10
    """
    return ToyDataset(
        mean=(-1.0, -1.0),
        num_samples=num_samples,
        source=False,
        variance=1.0
    )