"""
Compatibility layer for loading pre-trained models from HuggingFace diffusers.

This allows us to use google/ddpm-ema-celebahq-256 and google/ddpm-ema-church-256
pre-trained models with our adaptor and ANT training pipeline.

The diffusers UNet2DModel has a compatible interface: forward(sample, timestep) -> sample.
We wrap it to match our codebase's expectations.
"""

import torch
import torch.nn as nn
from typing import Optional


class DiffusersUNetWrapper(nn.Module):
    """
    Wraps a diffusers UNet2DModel to match our UNetModel interface.

    The diffusers model outputs UNet2DOutput with a .sample attribute,
    while our code expects a raw tensor output.
    """

    def __init__(self, diffusers_unet):
        super().__init__()
        self.unet = diffusers_unet
        # Store model_channels for time embedding compatibility
        self.model_channels = diffusers_unet.config.block_out_channels[0]
        self.image_size = diffusers_unet.config.sample_size

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching our UNetModel interface.

        Args:
            x: Noisy image (B, C, H, W)
            timesteps: Timestep indices (B,)

        Returns:
            Model output (B, C, H, W) - noise prediction
        """
        output = self.unet(x, timesteps)
        return output.sample

    @property
    def in_channels(self):
        return self.unet.config.in_channels

    @property
    def out_channels(self):
        return self.unet.config.out_channels


class DiffusersAdaptorWrapper(nn.Module):
    """
    Adaptor wrapper for diffusers UNet2DModel.

    Injects adaptor layers at the output of each ResNet block in the
    diffusers UNet. Since diffusers uses a different internal structure,
    we use hooks to inject adaptors.

    Args:
        diffusers_unet: Pre-trained diffusers UNet2DModel
        bottleneck_dim: Adaptor bottleneck dimension (d=8)
    """

    def __init__(
        self,
        diffusers_unet,
        bottleneck_dim: int = 8,
    ):
        super().__init__()
        self.unet = diffusers_unet
        self.model_channels = diffusers_unet.config.block_out_channels[0]
        self.image_size = diffusers_unet.config.sample_size

        # Freeze the pre-trained model
        for param in self.unet.parameters():
            param.requires_grad = False

        # Create adaptor layers for each block
        self.adaptors = nn.ModuleDict()
        self._register_adaptors(bottleneck_dim)

        # Storage for hook outputs
        self._adaptor_outputs = {}
        self._hooks = []
        self._register_hooks()

    def _register_adaptors(self, bottleneck_dim: int):
        """Create adaptor layers for ResNet blocks in the diffusers UNet."""
        block_channels = self.unet.config.block_out_channels

        # Down blocks
        for i, ch in enumerate(block_channels):
            for j in range(self.unet.config.layers_per_block):
                name = f"down_{i}_{j}"
                self.adaptors[name] = SimpleAdaptor(ch, bottleneck_dim)

        # Mid block
        self.adaptors["mid"] = SimpleAdaptor(block_channels[-1], bottleneck_dim)

        # Up blocks
        for i, ch in enumerate(reversed(block_channels)):
            for j in range(self.unet.config.layers_per_block + 1):
                name = f"up_{i}_{j}"
                self.adaptors[name] = SimpleAdaptor(ch, bottleneck_dim)

    def _register_hooks(self):
        """Register forward hooks on ResNet blocks."""
        block_channels = self.unet.config.block_out_channels

        # Down blocks
        for i, down_block in enumerate(self.unet.down_blocks):
            if hasattr(down_block, 'resnets'):
                for j, resnet in enumerate(down_block.resnets):
                    name = f"down_{i}_{j}"
                    hook = resnet.register_forward_hook(
                        self._make_hook(name)
                    )
                    self._hooks.append(hook)

        # Mid block
        if hasattr(self.unet.mid_block, 'resnets'):
            hook = self.unet.mid_block.resnets[0].register_forward_hook(
                self._make_hook("mid")
            )
            self._hooks.append(hook)

        # Up blocks
        for i, up_block in enumerate(self.unet.up_blocks):
            if hasattr(up_block, 'resnets'):
                for j, resnet in enumerate(up_block.resnets):
                    name = f"up_{i}_{j}"
                    hook = resnet.register_forward_hook(
                        self._make_hook(name)
                    )
                    self._hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a hook that adds adaptor output to the block output."""
        def hook_fn(module, input, output):
            if name in self.adaptors:
                adaptor_out = self.adaptors[name](output)
                return output + adaptor_out
        return hook_fn

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptor injection via hooks."""
        output = self.unet(x, timesteps)
        return output.sample

    def get_adaptor_parameters(self):
        """Get only adaptor parameters for optimization."""
        return list(self.adaptors.parameters())

    def count_adaptor_parameters(self) -> int:
        return sum(p.numel() for p in self.get_adaptor_parameters())

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_rate(self) -> float:
        return self.count_adaptor_parameters() / self.count_total_parameters()


class SimpleAdaptor(nn.Module):
    """Simple adaptor: down-project, GELU, up-project. Zero-initialized."""

    def __init__(self, channels: int, bottleneck_dim: int = 8):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck_dim, 1, bias=True)
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck_dim, channels, 1, bias=True)

        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


def load_pretrained_ddpm(model_id: str, device: torch.device = torch.device("cpu")):
    """
    Load a pre-trained DDPM from HuggingFace.

    Args:
        model_id: HuggingFace model ID, e.g.
            "google/ddpm-ema-celebahq-256" (face domain)
            "google/ddpm-ema-church-256" (church domain)
        device: Device to load to

    Returns:
        Tuple of (model_wrapper, noise_scheduler)
    """
    from diffusers import DDPMPipeline, DDPMScheduler

    print(f"Loading pre-trained model: {model_id}")
    pipeline = DDPMPipeline.from_pretrained(model_id)

    unet = pipeline.unet.to(device)
    scheduler = pipeline.scheduler

    wrapper = DiffusersUNetWrapper(unet)
    return wrapper, scheduler


def load_pretrained_with_adaptor(
    model_id: str,
    bottleneck_dim: int = 8,
    device: torch.device = torch.device("cpu"),
):
    """
    Load a pre-trained DDPM and wrap it with adaptors.

    Args:
        model_id: HuggingFace model ID
        bottleneck_dim: Adaptor bottleneck dim
        device: Device

    Returns:
        Tuple of (adaptor_model, noise_scheduler)
    """
    from diffusers import DDPMPipeline

    print(f"Loading pre-trained model: {model_id}")
    pipeline = DDPMPipeline.from_pretrained(model_id)

    unet = pipeline.unet.to(device)
    scheduler = pipeline.scheduler

    adaptor_model = DiffusersAdaptorWrapper(unet, bottleneck_dim)
    adaptor_model = adaptor_model.to(device)

    print(f"  Adaptor params: {adaptor_model.count_adaptor_parameters():,}")
    print(f"  Total params: {adaptor_model.count_total_parameters():,}")
    print(f"  Parameter rate: {adaptor_model.parameter_rate():.2%}")

    return adaptor_model, scheduler
