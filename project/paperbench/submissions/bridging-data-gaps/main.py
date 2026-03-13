## main.py
"""
Main entry point for DPMs-ANT (Diffusion Probabilistic Models with Adversarial Noise-based Transfer Learning).

This script orchestrates the entire pipeline:
1. Loads configuration from config.yaml
2. Initializes dataset loader and prepares 10-shot target data
3. Builds pre-trained U-Net model with adaptor layers
4. Trains binary classifier on noised source/target images
5. Sets up adversarial noise generator
6. Runs training loop (Algorithm 1) updating only adaptor parameters
7. Evaluates performance using Intra-LPIPS and FID

All components are initialized using values from config.yaml to ensure reproducibility.
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Any

# Import all required modules
from config import config
from data.dataset_loader import DatasetLoader
from model.unet_with_adaptor import UNetWithAdaptor
from classifier.binary_classifier import BinaryClassifier
from utils.noise_scheduler import DDPMNoiseScheduler
from utils.adversarial_noise_generator import AdversarialNoiseGenerator
from train.trainer import DPMsANTTrainer
from eval.evaluator import Evaluator


def load_config_from_yaml(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file not found
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories():
    """Create necessary directories for checkpoints and results."""
    Path(config.logging.checkpoint_dir).mkdir(exist_ok=True)
    Path(config.logging.results_dir).mkdir(exist_ok=True)


def main():
    """Main execution function."""
    print("Starting DPMs-ANT training pipeline...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup directories
    setup_directories()
    
    # Initialize dataset loader
    print("Initializing dataset loader...")
    dataset_loader = DatasetLoader()
    
    # Load target dataset (use first target domain for now)
    target_domain = config.dataset.target_domains_10shot[0]
    print(f"Loading 10-shot target dataset: {target_domain}")
    target_dataset = dataset_loader.load_target_dataset(target_domain, num_shots=10)
    target_dataloader = dataset_loader.get_dataloader(target_dataset)
    
    # Load source dataset
    source_domain = config.dataset.source_domains[0]
    print(f"Loading source dataset: {source_domain}")
    source_dataset = dataset_loader.load_source_dataset(source_domain)
    source_dataloader = dataset_loader.get_dataloader(source_dataset, batch_size=32)
    
    # Build base U-Net model (this would normally be loaded from checkpoint)
    print("Building U-Net with adaptor layers...")
    # Note: In practice, this would load a pre-trained U-Net from disk
    # For demonstration, we'll create a dummy structure
    class DummyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        
        def forward(self, x, timesteps=None):
            return self.conv(x)
    
    base_unet = DummyUNet()
    model = UNetWithAdaptor(base_unet, model_type='ddpm')
    model.to(device)
    
    # Freeze base U-Net weights
    model.freeze_base_unet()
    print(f"Model parameter efficiency: {model.get_parameter_efficiency():.4f}")
    
    # Train binary classifier
    print("Training binary classifier on noised images...")
    classifier = BinaryClassifier(device=device)
    
    # Use subset of source data for classifier training
    classifier.train_classifier(
        source_domain=source_domain,
        target_domain=target_domain,
        num_epochs=10,
        val_interval=5
    )
    
    # Save classifier checkpoint
    classifier.save_checkpoint(f"{config.logging.checkpoint_dir}/classifier_{target_domain}.pth")
    
    # Initialize noise scheduler
    print("Initializing noise scheduler...")
    noise_scheduler = DDPMNoiseScheduler().to(device)
    
    # Initialize adversarial noise generator
    print("Initializing adversarial noise generator...")
    adv_noise_gen = AdversarialNoiseGenerator(
        model=model.base_unet,  # Use frozen base model for noise generation
        noise_scheduler=noise_scheduler,
        device=device
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = DPMsANTTrainer(
        model=model,
        classifier=classifier,
        noise_gen=adv_noise_gen,
        noise_scheduler=noise_scheduler,
        device=device
    )
    
    # Run training
    print("Starting training...")
    trainer.train(target_dataloader)
    
    # Evaluate model
    print("Evaluating trained model...")
    evaluator = Evaluator(model, device=device)
    
    # Generate images
    gen_images = evaluator.generate_images(num_samples=config.evaluation.intra_lpips_num_samples)
    
    # Compute Intra-LPIPS
    real_images = [img for img in target_dataset]
    intra_lpips_score = evaluator.compute_intra_lpips(gen_images, real_images)
    print(f"Intra-LPIPS score: {intra_lpips_score:.6f}")
    
    # Save generated images
    output_dir = f"{config.logging.results_dir}/{target_domain}"
    evaluator.save_images(gen_images, output_dir)
    
    # Save final metrics
    metrics_file = f"{output_dir}/metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Target Domain: {target_domain}\n")
        f.write(f"Intra-LPIPS: {intra_lpips_score:.6f}\n")
        f.write(f"Parameter Efficiency: {model.get_parameter_efficiency():.6f}\n")
    
    print(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
