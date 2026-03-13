## model.py
"""
DiffusionWrapper with Adaptor for ANT-style transfer learning.

This module provides a lightweight, plug-in adaptor mechanism for pre-trained
diffusion models (DDPMs and Latent Diffusion Models). The adaptor is implemented
as a small bottleneck module (per-channel 1x1 convolutions) attached to a set of
target convolutional layers inside the base model via forward hooks. The base
diffusion backbone (θ) remains frozen during adaptor training; only the adaptor
parameters (ψ) are updated.

Two core ideas from the DPMs-ANT approach are supported:
- similarity-guided training: the adaptor participates in the diffusion denoising
  process and is guided by a fixed binary classifier φ that differentiates between
  source and target domains on a mid-noise input x_t.
- adversarial noise selection (AN): a lightweight inner loop identifies the
  “worst-case” Gaussian noise for the current adaptor and model, focusing training
  on the most challenging perturbations.

This implementation emphasizes reproducibility and a principled API surface while
remaining implementation-friendly for a variety of diffusion backbones. It is designed
to be used as a drop-in wrapper around a pre-trained diffusion model.

Notes:
- The adaptor uses a bottleneck of dimension `d` with a down-projection to `d`,
  GELU activation, and an up-projection back to the original channel dimension.
- Adaptor blocks are attached to a subset of Conv2d modules found inside the base model
  (default: first N Conv2d modules). The number of adaptor blocks can be controlled by
  adaptor_config (e.g., "num_blocks"). If not specified, a reasonable default is used.
- All adaptor parameters are initialized to zero so that the base model behavior is preserved
  at initialization.
- The forward hooks modify the outputs in-place by adding the adaptor delta. This preserves
  the original computation graph and allows gradients to flow to adaptor parameters ψ.

Usage overview
- Instantiate with a pre-trained base model and adaptor_config.
- Call load_pretrained(path) to load θ if needed.
- Call freeze_base() to ensure only ψ is trained.
- Forward-pass through the wrapper will automatically apply adaptor-equipped activations.

This module does not depend on external files beyond PyTorch and standard libraries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptorModule(nn.Module):
    """
    Bottleneck adaptor module.

    Given an input with `in_channels`, this module projects down to
    `bottleneck_channels` via a 1x1 conv, applies GELU, then projects back up to
    `out_channels` via another 1x1 conv. All weights are zero-initialized to ensure
    the adaptor is a no-op at initialization.

    The adaptor delta produced by this module has shape [N, out_channels, H, W],
    suitable for addition to the corresponding layer's output.
    """

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, padding=0, bias=True)
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, padding=0, bias=True)

        # Initialize to zero to ensure zero-effect at start
        self._zero_init(self.down)
        self._zero_init(self.up)

    @staticmethod
    def _zero_init(conv: nn.Conv2d) -> None:
        if isinstance(conv, nn.Conv2d):
            nn.init.constant_(conv.weight, 0.0)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptor delta given input feature map x.

        Args:
            x: Tensor of shape [N, in_channels, H, W]

        Returns:
            delta: Tensor of shape [N, out_channels, H, W]
        """
        z = self.down(x)
        z = self.act(z)
        delta = self.up(z)
        return delta


class AdaptorHook:
    """
    Forward-hook wrapper to inject adaptor deltas into a target Conv2d module.

    This class registers a forward hook on the target module. The hook computes a
    delta from the module's input using a small AdaptorModule and adds it to the
    module's output in-place. The adaptor's parameters are trainable; the base
    module's parameters are frozen by the surrounding DiffusionWrapper.

    The hook is designed to be robust to shape differences (via optional interpolation)
    and always preserves the computational graph for gradient flow to the adaptor.
    """

    def __init__(self, target_module: nn.Module, bottleneck: int) -> None:
        if not isinstance(target_module, nn.Conv2d):
            raise TypeError("AdaptorHook currently supports Conv2d target modules.")
        self.target = target_module
        in_ch = getattr(target_module, "in_channels", None)
        out_ch = getattr(target_module, "out_channels", None)
        if in_ch is None or out_ch is None:
            raise ValueError("Target Conv2d module missing channel information.")
        # Adaptor attached to this module
        self.adaptor = AdaptorModule(in_channels=in_ch, bottleneck_channels=bottleneck, out_channels=out_ch)
        # Forward hook
        self.hook_handle = target_module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module: nn.Module, input: List[torch.Tensor], output: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Forward hook that adds adaptor delta to the module's output.

        Args:
            module: target module (Conv2d)
            input: list containing the input tensor to the module
            output: the forward result tensor from the module

        Returns:
            Optional[Tensor]: If you return a tensor, PyTorch will replace output.
                              We perform in-place addition to avoid replacing hooks.
        """
        if not input:
            return None
        x = input[0]  # input feature map to the Conv2d layer
        delta = self.adaptor(x)  # shape [N, C_out, H, W]

        # Ensure shapes match; most Conv2d layers preserve spatial dims (stride=1).
        if delta.shape != output.shape:
            # Align by resizing delta to match output's spatial dimensions
            delta = F.interpolate(delta, size=output.shape[-2:], mode="nearest")

        # In-place addition to modify the module's output
        output.add_(delta)
        return None

    def detach(self) -> None:
        """Remove the registered forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class DiffusionWrapper(nn.Module):
    """
    Wrapper for a pre-trained diffusion backbone with a small trainable adaptor.

    Core responsibilities:
    - Hold a pre-trained base diffusion model (θ) which is typically frozen during ANT training.
    - Build and attach a small number of adaptor blocks ψ to a subset of Conv2d modules inside θ.
      Adaptor insertion follows a bottleneck design with zero initialization, so the adaptor is a
      controlled, low-capacity modification to the backbone.
    - Expose a forward(x_t, t) interface compatible with diffusion model usage (θ(x_t, t) with adaptors).
    - Provide utilities to load pre-trained weights, freeze the base model, and detach adaptors for
      ablations or resets.

    Important design notes:
    - Adaptor blocks are attached via forward hooks to Conv2d modules discovered inside the base model.
    - The adaptor delta has the same spatial dimensions as the corresponding layer's output (via 1x1 bottlenecks).
    - The adaptor's down projection uses a dimension defined by the config (d); the top-level
      channel dimension remains the same as the target Conv2d layer's out_channels.
    """

    def __init__(self, base_model: nn.Module, adaptor_config: Dict[str, Any]) -> None:
        """
        Initialize a diffusion wrapper with an adaptor.

        Args:
            base_model: Pre-trained diffusion backbone (DDPM or LDM U-Net, etc.).
            adaptor_config: Configuration dictionary with at least:
                - "d" or adaptor_params.*.d: bottleneck channel size for adaptor (int)
                - optional "num_blocks": max number of adaptor blocks to attach (int)
                - optional "c": unused by this implementation but kept for compatibility
        """
        super().__init__()
        self.base_model = base_model
        self._adaptor_config = adaptor_config or {}

        self._adaptor_hooks: List[AdaptorHook] = []
        self._attach_adaptors(self._adaptor_config)

    # ----------------------------
    # Public API surface
    # ----------------------------
    def load_pretrained(self, path: str) -> None:
        """
        Load pre-trained weights into the base diffusion backbone.

        This method only touches the base_model's weights; adaptors remain unaffected.

        Args:
            path: Path to the pre-trained state_dict (or a wrapper containing it).
        """
        try:
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state" in state:
                state = state["state"]
            self.base_model.load_state_dict(state, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained weights from {path}: {e}") from e

    def freeze_base(self) -> None:
        """Freeze all parameters of the base diffusion backbone θ; only adaptor parameters stay trainable."""
        for p in self.base_model.parameters():
            p.requires_grad = False
        # Ensure adaptor parameters require grad
        for hook in self._adaptor_hooks:
            for p in hook.adaptor.parameters():
                p.requires_grad = True

    def detach_adaptors(self) -> None:
        """Detach all adaptor hooks (useful for ablation or resetting)."""
        for hook in self._adaptor_hooks:
            hook.detach()
        self._adaptor_hooks = []

    def forward(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Forward pass through the diffusion backbone with adaptors.

        Args:
            x_t: Noised image tensor at timestep t, shape [N, C, H, W]
            t: Diffusion timestep

        Returns:
            The model output (predicted noise) from the base backbone, with adaptor effects
            injected via forward hooks.
        """
        return self.base_model(x_t, t)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _attach_adaptors(self, adaptor_config: Dict[str, Any]) -> None:
        """
        Attach adaptor hooks to a subset of Conv2d modules inside the base model.

        The adaptor uses a bottleneck dimension defined by adaptor_config['d'] and
        is attached to up to `num_blocks` Conv2d modules (default 4). The exact
        modules chosen are discovered in a deterministic way by traversing
        base_model.modules() and selecting Conv2d instances.

        Args:
            adaptor_config: Configuration with keys:
                - "d": bottleneck dimension for adaptor
                - optional "num_blocks": number of adaptor blocks to install
        """
        bottleneck = int(adaptor_config.get("d", 8))
        max_blocks = int(adaptor_config.get("num_blocks", 4))

        # Discover candidate Conv2d modules (excluding the root base_model itself)
        candidates: List[nn.Conv2d] = []
        for m in self.base_model.modules():
            if isinstance(m, nn.Conv2d):
                candidates.append(m)

        # Determine how many adaptors to install
        N = min(len(candidates), max(1, max_blocks))

        # Attach adaptors to the first N Conv2d modules (stable and deterministic)
        self._adaptor_hooks = []
        for idx in range(N):
            target = candidates[idx]
            try:
                hook = AdaptorHook(target, bottleneck=bottleneck)
                self._adaptor_hooks.append(hook)
            except Exception:
                # If a particular target cannot host an adaptor, skip gracefully
                continue

    # Optional: expose state for debugging or saving checkpoints
    def adaptor_param_count(self) -> int:
        """Return the total number of adaptor parameters currently active."""
        total = 0
        for hook in self._adaptor_hooks:
            total += sum(p.numel() for p in hook.adaptor.parameters())
        return total