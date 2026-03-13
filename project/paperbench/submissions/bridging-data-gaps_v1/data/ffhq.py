## data/ffhq.py

```python
"""FFHQ dataset for few-shot domain adaptation experiments.

This module implements the FFHQ dataset used in the paper's evaluation of
similarity-guided diffusion model adaptation:

- Source domain: FFHQ face images (real photos)
- Target domains: 
  - Sketches: FFHQ → Sketches
  - Amedeo paintings: FFHQ → Amedeo's paintings

The dataset supports few-shot adaptation with a small number of target domain
samples (typically 10 as per paper settings in Section 5.1).

Classes:
    FFHQDataset: PyTorch Dataset for FFHQ source/target domain adaptation
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Literal
from torch import Tensor
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class FFHQDataset(Dataset):
    """FFHQ dataset for few-shot domain adaptation experiments.
    
    This dataset loads FFHQ source images and target domain images (sketches or 
    Amedeo paintings) for few-shot domain adaptation experiments as described
    in the paper's evaluation.
    
    The dataset follows the few-shot setting where only a small number of
    target domain samples are available (e.g., 10 samples).
    
    Attributes:
        data_dir: Root directory containing FFHQ and target domain images
        few_shot: Number of few-shot target samples to use
        domain: Target domain name ('sketches' or 'amedeo')
        image_size: Size to resize images to (default: 256)
        source_images: List of source image paths
        target_images: List of target image paths (limited to few_shot)
        transform: Transform pipeline for image preprocessing
        is_source: Boolean flag for source vs target mode
    
    Example:
        >>> # Create FFHQ -> Sketches dataset with 10 target samples
        >>> dataset = FFHQDataset(
        ...     data_dir='./data',
        ...     few_shot=10,
        ...     domain='sketches',
        ...     image_size=256
        ... )
        >>> print(len(dataset))  # Number of samples
    """
    
    def __init__(
        self,
        data_dir: str,
        few_shot: int = 10,
        domain: Literal['sketches', 'amedeo'] = 'sketches',
        image_size: int = 256,
        is_source: bool = True
    ) -> None:
        """Initialize FFHQ dataset for few-shot domain adaptation.
        
        Args:
            data_dir: Root directory containing 'ffhq' and target domain subdirectories.
                     Expected structure:
                     - data_dir/ffhq/ - source images
                     - data_dir/sketches/ - target sketches
                     - data_dir/amedeo/ - target paintings
            few_shot: Number of few-shot target samples to use (default: 10)
                     This is the key setting from paper Section 5.1
            domain: Target domain name. Options:
                    - 'sketches': FFHQ → Sketches
                    - 'amedeo': FFHQ → Amedeo's paintings
            image_size: Size to resize images to (default: 256 for FFHQ)
            is_source: If True, load source images (FFHQ). If False, load target images.
                      This allows separate handling of source and target distributions.
        
        Raises:
            ValueError: If domain is not 'sketches' or 'amedeo'
            FileNotFoundError: If required data directories are not found
        """
        # Validate domain parameter
        valid_domains = ['sketches', 'amedeo']
        if domain not in valid_domains:
            raise ValueError(
                f"Invalid domain: {domain}. Must be one of {valid_domains}"
            )
        
        self.data_dir = Path(data_dir)
        self.few_shot = few_shot
        self.domain = domain
        self.image_size = image_size
        self.is_source = is_source
        
        # Define image transform pipeline for FFHQ images
        # Following standard diffusion model preprocessing:
        # - Resize to target size
        # - Convert to tensor [0, 1]
        # - Normalize to [-1, 1] range (matching diffusion model expectations)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load source and target image paths
        self.source_images: List[Path] = []
        self.target_images: List[Path] = []
        
        # Check for required directories and load images
        self._load_image_paths()
    
    def _load_image_paths(self) -> None:
        """Load image paths from data directory.
        
        Searches for:
        - Source: data_dir/ffhq/ containing FFHQ face images
        - Target: data_dir/{domain}/ containing target domain images
        """
        # Load source images (FFHQ)
        ffhq_dir = self.data_dir / 'ffhq'
        if ffhq_dir.exists():
            # Support both .png and .jpg extensions
            self.source_images = sorted(
                list(ffhq_dir.glob('*.png')) + 
                list(ffhq_dir.glob('*.jpg')) +
                list(ffhq_dir.glob('*.jpeg'))
            )
        else:
            # If source directory doesn't exist, create empty list
            # This allows for lazy loading or custom data setup
            self.source_images = []
        
        # Load target images (limited to few_shot samples)
        target_dir = self.data_dir / self.domain
        if target_dir.exists():
            target_all = sorted(
                list(target_dir.glob('*.png')) + 
                list(target_dir.glob('*.jpg')) +
                list(target_dir.glob('*.jpeg'))
            )
            # Take only few_shot samples (first N for reproducibility)
            self.target_images = target_all[:self.few_shot]
        else:
            self.target_images = []
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.
        
        Returns:
            Number of samples (source images if is_source=True, 
                             target images if is_source=False)
        """
        if self.is_source:
            return len(self.source_images)
        else:
            return len(self.target_images)
    
    def __getitem__(self, idx: int) -> Tensor:
        """Get a single sample from the dataset by index.
        
        Args:
            idx: Index of the sample to retrieve (0 <= idx < len(dataset))
        
        Returns:
            Image tensor of shape [3, image_size, image_size]
            Normalized to [-1, 1] range
        
        Raises:
            IndexError: If idx is out of bounds
            FileNotFoundError: If image file cannot be loaded
        """
        if self.is_source:
            if idx >= len(self.source_images):
                raise IndexError(
                    f"Source index {idx} out of bounds for {len(self.source_images)} images"
                )
            img_path = self.source_images[idx]
        else:
            if idx >= len(self.target_images):
                raise IndexError(
                    f"Target index {idx} out of bounds for {len(self.target_images)} images"
                )
            img_path = self.target_images[idx]
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        return image_tensor
    
    def get_source_sample(self, idx: int) -> Tensor:
        """Get a source domain sample (FFHQ) by index.
        
        Convenience method to explicitly get source samples.
        
        Args:
            idx: Index of source sample
        
        Returns:
            Source image tensor [3, image_size, image_size]
        """
        if idx >= len(self.source_images):
            raise IndexError(
                f"Source index {idx} out of bounds for {len(self.source_images)} images"
            )
        
        img_path = self.source_images[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)
    
    def get_target_sample(self, idx: int) -> Tensor:
        """Get a target domain sample by index.
        
        Convenience method to explicitly get target samples.
        
        Args:
            idx: Index of target sample (0 <= idx < few_shot)
        
        Returns:
            Target image tensor [3, image_size, image_size]
        """
        if idx >= len(self.target_images):
            raise IndexError(
                f"Target index {idx} out of bounds for {len(self.target_images)} images"
            )
        
        img_path = self.target_images[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)
    
    def get_distribution_info(self) -> dict:
        """Get information about the dataset distribution.
        
        Returns:
            Dictionary containing dataset distribution information
        """
        return {
            'domain': self.domain,
            'few_shot': self.few_shot,
            'num_source_images': len(self.source_images),
            'num_target_images': len(self.target_images),
            'image_size': self.image_size,
            'is_source': self.is_source
        }


def create_ffhq_sketches_dataset(
    data_dir: str,
    few_shot: int = 10,
    image_size: int = 256,
    is_source: bool = True
) -> FFHQDataset:
    """Factory function to create FFHQ → Sketches dataset.
    
    Creates a dataset for the FFHQ to Sketches domain adaptation task
    as used in the paper's evaluation.
    
    Args:
        data_dir: Root directory containing 'ffhq' and 'sketches' subdirectories
        few_shot: Number of few-shot target samples (default: 10 from paper)
        image_size: Size to resize images to (default: 256)
        is_source: If True, load source images (FFHQ). If False, load target (sketches)
    
    Returns:
        FFHQDataset instance configured for sketches target domain
    
    Example:
        >>> # Create source dataset (FFHQ photos)
        >>> source_ds = create_ffhq_sketches_dataset(
        ...     data_dir='./data',
        ...     few_shot=10,
        ...     is_source=True
        ... )
        >>>
        >>> # Create target dataset (Sketches)
        >>> target_ds = create_ffhq_sketches_dataset(
        ...     data_dir='./data',
        ...     few_shot=10,
        ...     is_source=False
        ... )
    """
    return FFHQDataset(
        data_dir=data_dir,
        few_shot=few_shot,
        domain='sketches',
        image_size=image_size,
        is_source=is_source
    )


def create_ffhq_amedeo_dataset(
    data_dir: str,
    few_shot: int = 10,
    image_size: int = 256,
    is_source: bool = True
) -> FFHQDataset:
    """Factory function to create FFHQ → Amedeo paintings dataset.
    
    Creates a dataset for the FFHQ to Amedeo's paintings domain adaptation task
    as used in the paper's evaluation.
    
    Args:
        data_dir: Root directory containing 'ffhq' and 'amedeo' subdirectories
        few_shot: Number of few-shot target samples (default: 10 from paper)
        image_size: Size to resize images to (default: 256)
        is_source: If True, load source images (FFHQ). If False, load target (amedeo)
    
    Returns:
        FFHQDataset instance configured for Amedeo paintings target domain
    
    Example:
        >>> # Create source dataset (FFHQ photos)
        >>> source_ds = create_ffhq_amedeo_dataset(
        ...     data_dir='./data',
        ...     few_shot=10,
        ...     is_source=True
        ... )
        >>>
        >>> # Create target dataset (Amedeo paintings)
        >>> target_ds = create_ffhq_amedeo_dataset(
        ...     data_dir='./data',
        ...     few_shot=10,
        ...     is_source=False
        ... )
    """
    return FFHQDataset(
        data_dir=data_dir,
        few_shot=few_shot,
        domain='amedeo',
        image_size=image_size,
        is_source=is_source
    )


class FFHQDomainPair:
    """paired dataset containing both source and target FFHQ domains.
    
    This class provides a unified interface for accessing both source (FFHQ)
    and target (Sketches or Amedeo) domains simultaneously, which is useful
    for training with both distributions.
    
    Attributes:
        source_dataset: FFHQDataset for source domain
        target_dataset: FFHQDataset for target domain
    """
    
    def __init__(
        self,
        data_dir: str,
        few_shot: int = 10,
        domain: Literal['sketches', 'amedeo'] = 'sketches',
        image_size: int = 256
    ) -> None:
        """Initialize paired source and target datasets.
        
        Args:
            data_dir: Root directory containing FFHQ and target domain images
            few_shot: Number of few-shot target samples
            domain: Target domain name ('sketches' or 'amedeo')
            image_size: Size to resize images to
        """
        self.source_dataset = FFHQDataset(
            data_dir=data_dir,
            few_shot=few_shot,
            domain=domain,
            image_size=image_size,
            is_source=True
        )
        
        self.target_dataset = FFHQDataset(
            data_dir=data_dir,
            few_shot=few_shot,
            domain=domain,
            image_size=image_size,
            is_source=False
        )
        
        self.domain = domain
    
    def get_source_batch(self, batch_size: int) -> Tensor:
        """Get a batch of source images.
        
        Args:
            batch_size: Number of images to sample
        
        Returns:
            Tensor of source images [batch_size, 3, image_size, image_size]
        """
        indices = torch.randint(len(self.source_dataset), (batch_size,))
        return torch.stack([self.source_dataset[i.item()] for i in indices])
    
    def get_target_batch(self, batch_size: int) -> Tensor:
        """Get a batch of target images.
        
        Args:
            batch_size: Number of images to sample (will repeat if fewer targets)
        
        Returns:
            Tensor of target images [batch_size, 3, image_size, image_size]
        """
        # Sample with replacement if batch_size > few_shot
        if batch_size <= len(self.target_dataset):
            indices = torch.randperm(len(self.target_dataset))[:batch_size]
        else:
            # Repeat indices to match batch_size
            indices = torch.randint(len(self.target_dataset), (batch_size,))
        
        return torch.stack([self.target_dataset[i.item()] for i in indices])
    
    def __len__(self) -> Tuple[int, int]:
        """Return (num_source, num_target) sample counts."""
        return (len(self.source_dataset), len(self.target_dataset))