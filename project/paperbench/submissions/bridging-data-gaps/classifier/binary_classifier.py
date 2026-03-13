## classifier/binary_classifier.py
"""
Binary classifier for DPMs-ANT that distinguishes between source and target domains using noised images.

Implements the similarity-guided training component by providing a gradient signal based on whether 
a noised image x_t belongs to the source or target domain. This classifier is trained on pairs of 
source (e.g., FFHQ) and 10-shot target images, where each input is diffused to random timestep t.

Key features:
- Trained on noised versions of images at arbitrary timesteps t ∈ [1, T]
- Computes ∇_{x_t} log p_ϕ(y=τ|x_t) as similarity guidance signal
- Frozen after initial training; only used for inference during main loop
- Lightweight CNN architecture suitable for fast evaluation

All configuration values are sourced from config.yaml to ensure consistency across the pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Import dependencies
from config import config
from utils.noise_scheduler import DDPMNoiseScheduler
from data.dataset_loader import DatasetLoader


class BinaryClassifier(nn.Module):
    """
    Binary classifier that predicts whether a noised image x_t comes from source or target domain.
    
    The model outputs logits for binary classification (source vs target). During training,
    it learns to distinguish between source dataset images and 10-shot target images, both
    transformed via forward diffusion to random timesteps.
    
    After training, this model is frozen and used to compute input gradients for similarity guidance.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 lr: float = 2e-4,
                 device: Optional[torch.device] = None):
        """
        Initialize binary classifier with lightweight CNN architecture.
        
        Args:
            input_channels: Number of input channels (default: 3 for RGB)
            lr: Learning rate for training (default: 2e-4)
            device: Device to place model on. If None, uses CUDA if available
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.lr = lr
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use same image size as configured in dataset
        self.image_size = config.dataset.image_size
        
        # Build network architecture
        self.features = self._build_feature_extractor()
        self.classifier = self._build_classifier_head()
        
        # Move to device
        self.to(self.device)
        
        # Store loss history
        self.train_losses = []
    
    def _build_feature_extractor(self) -> nn.Sequential:
        """
        Build convolutional feature extractor with progressive downsampling.
        
        Architecture:
          Input: (B, 3, 256, 256)
          → ConvBlock: 64 filters, stride 2 → (B, 64, 128, 128)
          → ConvBlock: 128 filters, stride 2 → (B, 128, 64, 64)
          → ConvBlock: 256 filters, stride 2 → (B, 256, 32, 32)
          → ConvBlock: 512 filters, stride 2 → (B, 512, 16, 16)
          → ConvBlock: 512 filters, stride 2 → (B, 512, 8, 8)
          
        Returns:
            Sequential feature extractor
        """
        return nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(self.input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            # Block 2: 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            # Block 3: 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            # Block 4: 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            
            # Block 5: 16 -> 8
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )
    
    def _build_classifier_head(self) -> nn.Linear:
        """
        Build final classification head.
        
        Takes flattened features from last conv layer and produces single logit.
        
        Returns:
            Linear layer mapping to scalar logit
        """
        # After 5 downsampling blocks: 256 / (2^5) = 8
        # Feature map size: (B, 512, 8, 8) → flattened: 512 * 8 * 8 = 32768
        feature_dim = 512 * 8 * 8
        return nn.Linear(feature_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classifier.
        
        Args:
            x: Input tensor of shape (B, C, H, W), assumed normalized to [-1, 1]
            
        Returns:
            Logits of shape (B,) representing unnormalized scores for 'target' class
        """
        B = x.shape[0]
        
        # Extract features
        features = self.features(x)  # (B, 512, 8, 8)
        
        # Flatten
        flat_features = features.view(B, -1)  # (B, 32768)
        
        # Classify
        logits = self.classifier(flat_features).squeeze(-1)  # (B,)
        
        return logits
    
    def predict_log_prob_gradient(self, xt: torch.Tensor, label: str = 'target') -> torch.Tensor:
        """
        Compute gradient of log probability w.r.t. input xt.
        
        This implements the similarity signal used in Equation (5):
        ∇_{x_t} log p_ϕ(y = τ | x_t)
        
        Args:
            xt: Noised input image of shape (B, C, H, W)
            label: Which class probability to differentiate ('target' or 'source')
            
        Returns:
            Gradient tensor of same shape as xt
            
        Raises:
            ValueError: If label is not 'target' or 'source'
        """
        # Ensure model is in eval mode
        self.eval()
        
        # Detach input to avoid unwanted backprop
        xt = xt.detach().clone()
        xt.requires_grad_(True)
        
        # Forward pass
        logits = self.forward(xt)  # (B,)
        
        # Compute log probability for specified class
        if label == 'target':
            # log p(y=τ|x_t) = log sigmoid(logits)
            log_prob = -F.softplus(-logits)  # numerically stable log(sigmoid(x))
        elif label == 'source':
            # log p(y=s|x_t) = log (1 - sigmoid(logits)) = log sigmoid(-logits)
            log_prob = -F.softplus(logits)
        else:
            raise ValueError(f"Label must be 'target' or 'source', got {label}")
        
        # Compute gradient w.r.t. input
        grad = torch.autograd.grad(
            outputs=log_prob.sum(),      # Sum over batch
            inputs=xt,
            create_graph=False,           # Not needed for outer optimization
            retain_graph=False,           # Don't need to reuse computation
            allow_unused=False
        )[0]
        
        return grad
    
    def train_classifier(self,
                         source_domain: str,
                         target_domain: str,
                         num_epochs: int = 50,
                         val_interval: int = 5) -> None:
        """
        Train the binary classifier on source vs target domain discrimination task.
        
        Uses dynamic noise scheduling to generate x_t from clean images at random timesteps.
        
        Args:
            source_domain: Name of source domain (e.g., 'FFHQ')
            target_domain: Name of target domain (e.g., 'Sunglasses')
            num_epochs: Number of training epochs
            val_interval: How often to print loss summary
        """
        # Create dataset loader
        loader = DatasetLoader()
        
        # Load datasets
        source_dataset = loader.load_source_dataset(source_domain)
        target_dataset = loader.load_target_dataset(target_domain, num_shots=config.classifier.num_shots)
        
        # Create data loaders
        source_loader = loader.get_dataloader(source_dataset, shuffle=True)
        target_loader = loader.get_dataloader(target_dataset, shuffle=True)
        
        # Create noise scheduler
        scheduler = DDPMNoiseScheduler()
        scheduler.to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.train()
        total_steps = min(len(source_loader), len(target_loader)) * num_epochs
        step_count = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # Zip both loaders together
            for (source_batch, _), (target_batch, _) in zip(source_loader, target_loader):
                # Move to device
                source_batch = source_batch.to(self.device)
                target_batch = source_batch.to(self.device)  # Already normalized
                
                # Sample random timesteps
                t_source = torch.randint(1, scheduler.T + 1, (source_batch.size(0),), device=self.device)
                t_target = torch.randint(1, scheduler.T + 1, (target_batch.size(0),), device=self.device)
                
                # Generate noise
                noise_source = torch.randn_like(source_batch)
                noise_target = torch.randn_like(target_batch)
                
                # Apply forward diffusion
                xt_source = scheduler.add_noise(source_batch, t_source, noise_source)
                xt_target = scheduler.add_noise(target_batch, t_target, noise_target)
                
                # Concatenate batches: first half source, second half target
                combined_xt = torch.cat([xt_source, xt_target], dim=0)
                labels = torch.cat([
                    torch.zeros(xt_source.size(0), device=self.device),  # source = 0
                    torch.ones(xt_target.size(0), device=self.device)     # target = 1
                ], dim=0)
                
                # Shuffle combined batch
                shuffle_idx = torch.randperm(combined_xt.size(0))
                combined_xt = combined_xt[shuffle_idx]
                labels = labels[shuffle_idx]
                
                # Forward pass
                logits = self.forward(combined_xt)
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update stats
                epoch_loss += loss.item()
                batch_count += 1
                step_count += 1
                
                # Print progress
                if step_count % val_interval == 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"Step [{step_count}/{total_steps}], Loss: {avg_loss:.4f}")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            self.train_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
        
        # Final evaluation mode
        self.eval()
        print("Binary classifier training completed.")
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save classifier state dictionary.
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'train_losses': self.train_losses,
            'config': {
                'input_channels': self.input_channels,
                'lr': self.lr,
                'image_size': self.image_size
            }
        }, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load classifier from checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.eval()  # Set to evaluation mode
        print(f"Loaded classifier checkpoint from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Create classifier
    classifier = BinaryClassifier(lr=2e-4)
    print(f"Initialized BinaryClassifier on {classifier.device}")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(4, 3, 256, 256).to(classifier.device)
    logits = classifier(dummy_input)
    print(f"Forward pass successful: logits shape {logits.shape}")
    
    # Test gradient computation
    grad = classifier.predict_log_prob_gradient(dummy_input, label='target')
    print(f"Gradient computation successful: grad shape {grad.shape}")
    
    # Optionally train classifier (requires actual data)
    # WARNING: This will fail unless data directories exist
    try:
        print("\nStarting classifier training...")
        classifier.train_classifier(
            source_domain="FFHQ",
            target_domain="Sunglasses",
            num_epochs=2,  # Short test run
            val_interval=1
        )
        
        # Save checkpoint
        classifier.save_checkpoint("checkpoints/classifier_sunglasses.pth")
        
    except FileNotFoundError as e:
        print(f"Skipping training test: Data not found - {e}")
        print("Please ensure data directories are set up correctly before training.")
