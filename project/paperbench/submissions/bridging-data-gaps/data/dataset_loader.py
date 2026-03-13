## data/dataset_loader.py
"""
Dataset loading module for DPMs-ANT.

Implements DatasetLoader class to handle 10-shot target datasets and full source datasets.
Provides standardized preprocessing (center crop, resize, normalization) and DataLoader creation.
All configurations are derived from config.yaml to ensure consistency across the pipeline.
"""

import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Import configuration
from config import config


@dataclass
class TargetImage:
    """Represents a single target image with metadata."""
    path: str
    domain: str
    index: int


class LimitedImageFolder(Dataset):
    """Custom dataset that loads only a specified number of images from a folder."""
    
    def __init__(self, root: str, transform=None, num_images: int = 10):
        """
        Initialize dataset that loads exactly num_images from root directory.
        
        Args:
            root: Path to directory containing images
            transform: Optional transform to apply to images
            num_images: Number of images to load (default: 10)
        """
        self.root = root
        self.transform = transform
        self.num_images = num_images
        
        # Get all image files, sorted for consistency
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        all_files = [f for f in sorted(os.listdir(root)) if f.lower().endswith(valid_extensions)]
        
        # Ensure we have enough images
        if len(all_files) < num_images:
            raise ValueError(f"Insufficient images in {root}: found {len(all_files)}, need {num_images}")
            
        self.image_files = all_files[:num_images]
        self.file_paths = [os.path.join(root, f) for f in self.image_files]
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and transform image at given index.
        
        Args:
            idx: Index of image to retrieve
            
        Returns:
            Transformed image tensor
        """
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


class DatasetLoader:
    """Handles loading and preprocessing of source and target datasets."""
    
    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize dataset loader with configuration from config.yaml.
        
        Args:
            data_root: Optional override for data directory path. If None, uses default structure.
        """
        self.image_size = config.dataset.image_size
        self.center_crop = config.dataset.center_crop
        self.batch_size = config.training.batch_size
        
        # Set data root path
        if data_root is None:
            self.data_root = "data"
        else:
            self.data_root = data_root
            
        # Create transformation pipeline
        self.transform = self._build_transform()
        
        # Validate dataset directories exist
        self._validate_dataset_structure()
    
    def _build_transform(self) -> transforms.Compose:
        """
        Construct image transformation pipeline based on configuration.
        
        Returns:
            Composed transform for preprocessing images
        """
        transform_list = []
        
        # Add center crop if specified
        if self.center_crop:
            # Center crop will be applied first, then resize
            transform_list.append(transforms.CenterCrop(self.image_size))
        
        # Always resize to target size
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Convert to tensor and normalize to [-1, 1]
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def _validate_dataset_structure(self) -> None:
        """Validate that required dataset directories exist."""
        # Check source domains
        for source in config.dataset.source_domains:
            source_name = source.lower().replace("_", "")
            source_path = os.path.join(self.data_root, source_name)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source dataset '{source}' not found at {source_path}")
        
        # Check target domains
        for target in config.dataset.target_domains_10shot:
            target_path = os.path.join(self.data_root, "target", target)
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Target dataset '{target}' not found at {target_path}")
    
    def load_target_dataset(self, name: str, num_shots: int = 10) -> Dataset:
        """
        Load exactly num_shots images from specified target domain.
        
        Args:
            name: Name of target domain (must be in config.dataset.target_domains_10shot)
            num_shots: Number of images to load (default: 10)
            
        Returns:
            Dataset containing limited number of target images
            
        Raises:
            ValueError: If target domain name is invalid or insufficient images available
        """
        if name not in config.dataset.target_domains_10shot:
            raise ValueError(f"Unknown target domain: {name}. Must be one of {config.dataset.target_domains_10shot}")
        
        target_path = os.path.join(self.data_root, "target", name)
        
        return LimitedImageFolder(
            root=target_path,
            transform=self.transform,
            num_images=num_shots
        )
    
    def load_source_dataset(self, name: str) -> Dataset:
        """
        Load full source dataset for pre-training domain.
        
        Args:
            name: Name of source domain (must be in config.dataset.source_domains)
            
        Returns:
            Dataset containing full source dataset
            
        Raises:
            ValueError: If source domain name is invalid
        """
        if name not in config.dataset.source_domains:
            raise ValueError(f"Unknown source domain: {name}. Must be one of {config.dataset.source_domains}")
        
        # Map domain name to directory name (e.g., "FFHQ" -> "ffhq")
        dir_name = name.lower().replace("_", "")
        source_path = os.path.join(self.data_root, dir_name)
        
        return ImageFolder(
            root=source_path,
            transform=self.transform
        )
    
    def get_dataloader(self, 
                      dataset: Dataset, 
                      batch_size: Optional[int] = None, 
                      shuffle: bool = True,
                      drop_last: bool = True) -> DataLoader:
        """
        Create DataLoader from given dataset with specified parameters.
        
        Args:
            dataset: Dataset to wrap in DataLoader
            batch_size: Batch size to use. If None, uses config.training.batch_size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            
        Returns:
            Configured DataLoader object
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,  # Use multiple workers for faster data loading
            pin_memory=True  # Enable faster data transfer to GPU
        )
    
    def get_all_target_loaders(self) -> dict:
        """
        Get DataLoaders for all 10-shot target domains.
        
        Returns:
            Dictionary mapping target domain names to their DataLoaders
        """
        loaders = {}
        for target_domain in config.dataset.target_domains_10shot:
            dataset = self.load_target_dataset(target_domain)
            loaders[target_domain] = self.get_dataloader(dataset)
        return loaders
    
    def get_all_source_loaders(self) -> dict:
        """
        Get DataLoaders for all source domains.
        
        Returns:
            Dictionary mapping source domain names to their DataLoaders
        """
        loaders = {}
        for source_domain in config.dataset.source_domains:
            dataset = self.load_source_dataset(source_domain)
            # Use smaller batch size for source datasets during classifier training
            loaders[source_domain] = self.get_dataloader(dataset, batch_size=self.batch_size)
        return loaders


# Example usage and testing
if __name__ == "__main__":
    # Create dataset loader
    loader = DatasetLoader()
    
    # Test loading a target dataset
    try:
        babies_dataset = loader.load_target_dataset("Babies")
        print(f"Loaded Babies dataset with {len(babies_dataset)} images")
        
        # Create dataloader
        babies_loader = loader.get_dataloader(babies_dataset)
        print(f"Babies DataLoader batch size: {babies_loader.batch_size}")
        
        # Test loading a source dataset
        ffhq_dataset = loader.load_source_dataset("FFHQ")
        print(f"Loaded FFHQ dataset with {len(ffhq_dataset)} images")
        
        # Create dataloader
        ffhq_loader = loader.get_dataloader(ffhq_dataset)
        print(f"FFHQ DataLoader batch size: {ffhq_loader.batch_size}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
