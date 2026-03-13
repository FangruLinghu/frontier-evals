from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def collapse_features(x: torch.Tensor) -> torch.Tensor:
    """
    Collapse spatial or sequence dimensions of a feature tensor into a 2D [B, C] representation.

    Supported input shapes:
    - [B, C, H, W] -> global average pooling over HxW -> [B, C]
    - [B, C, L]    -> mean over L -> [B, C]
    - [B, C]       -> [B, C] (no change)

    Args:
        x: Input tensor of shape described above.

    Returns:
        Tensor of shape [B, C].
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("collapse_features expects a torch.Tensor input")

    if x.dim() == 4:
        # [B, C, H, W]
        out = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), x.size(1))
        return out
    elif x.dim() == 3:
        # [B, C, L]
        return x.mean(dim=-1)
    elif x.dim() == 2:
        # [B, C]
        return x
    else:
        raise ValueError(f"Unsupported input tensor shape for collapse_features: {x.shape}")


def build_binary_head(in_features: int, hidden_dim: int = 128, zero_init: bool = True) -> nn.Sequential:
    """
    Build a simple 2-layer MLP head for binary classification on features of size in_features.

    The head maps [B, in_features] -> [B, 2] logits (logits for two classes).

    Args:
        in_features: Number of input feature channels (feature dimension).
        hidden_dim: Number of hidden units in the intermediate layer.
        zero_init: If True, initialize Linear layers' weights and biases to zero for deterministic behavior.

    Returns:
        An nn.Sequential representing the classifier head.
    """
    layers = []
    layers.append(nn.Linear(in_features, hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, 2))
    head = nn.Sequential(*layers)

    if zero_init:
        with torch.no_grad():
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    if m.weight is not None:
                        m.weight.zero_()
                    if m.bias is not None:
                        m.bias.zero_()
    return head


__all__ = ["collapse_features", "build_binary_head"]
