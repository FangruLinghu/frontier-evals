import torch
import torch.nn as nn
from typing import List

# Import bottleneck configuration helper from adapters_config
try:
    from .adapters_config import bottleneck_configs, get_bottleneck_config  # type: ignore
except Exception:
    # Fallback in case of package import differences during tests
    bottleneck_configs = {
        "ddpm": {"c": 4, "d": 8},
        "ldm": {"c": 2, "d": 8},
    }


class HoulsbyAdaptor(nn.Module):
    """Houlsby-style adaptor module for a diffusion backbone layer.

    Each adaptor performs a bottleneck transformation on the input feature map
    and adds the result residually: x_t^l -> x_t^l + ψ_l(x_t^{l-1}).

    Parameters:
    - in_channels: number of input channels at the layer
    - bottleneck_channels: number of channels to project down to in the bottleneck
    - activation: non-linearity to apply within the bottleneck (class or callable)
    """

    def __init__(self, in_channels: int, bottleneck_channels: int, activation=nn.ReLU):
        super(HoulsbyAdaptor, self).__init__()
        self.activation = activation
        # Bottleneck projection: down-project, nonlinearity, up-project
        # Using 1x1 convolutions to act as channel-wise bottlenecks.
        self.down = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, padding=0, bias=True)
        self.up = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, padding=0, bias=True)

        # Initialize adaptor parameters to zero so that the initial forward pass
        # matches the frozen backbone behavior (no task-specific shift).
        with torch.no_grad():
            if self.down.weight is not None:
                self.down.weight.zero_()
            if self.down.bias is not None:
                self.down.bias.zero_()
            if self.up.weight is not None:
                self.up.weight.zero_()
            if self.up.bias is not None:
                self.up.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the adaptor.

        x: input feature map with shape [B, C, H, W]
        Returns: x + ψ(x) where ψ is the bottleneck transformation.
        """
        y = self.down(x)
        y = self.activation()(y) if callable(self.activation) else self.activation(y)  # ensure callable
        y = self.up(y)
        return x + y


def build_adaptors_for_backbone(
    backbone_name: str,
    in_channels_list: List[int],
    activation=nn.ReLU,
) -> nn.Module:
    """Factory to build a per-layer adaptor module list for a given backbone.

    For each layer channel size in in_channels_list, instantiate a HoulsbyAdaptor
    with bottleneck width derived from the backbone's configuration.

    backbone_name is case-insensitive (e.g., 'ddpm' or 'ldm').
    """
    name_key = backbone_name.lower()
    # Attempt to fetch bottleneck configuration; fall back to defaults if missing
    try:
        c, _ = get_bottleneck_config(name_key)
    except Exception:
        if name_key in bottleneck_configs:
            c = bottleneck_configs[name_key]["c"]
        else:
            # Default to a safe value if config is not found
            c = 4

    adaptors = []
    for ch in in_channels_list:
        adaptors.append(HoulsbyAdaptor(in_channels=ch, bottleneck_channels=int(c), activation=activation))
    return nn.ModuleList(adaptors)


__all__ = ["HoulsbyAdaptor", "build_adaptors_for_backbone"]
