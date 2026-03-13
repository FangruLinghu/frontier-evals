# Domain classifier for p_φ(y|x_t)
# Binary classifier distinguishing source vs target domain at diffusion timestep x_t states.
# Simple, lightweight MLP with spatial pooling to handle 4D feature maps typically produced by CNN backbones.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    """Binary domain classifier over diffusion timestep representations.

    Given an input feature map x_t (e.g., [B, C, H, W]), this module pools spatial
    information to a vector and passes it through a small MLP to produce log-probabilities
    for two classes: [source, target]. The output is log-probabilities (logits normalized with log_softmax).
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = int(hidden_dim)

        # Simple adaptor that reduces spatial information and maps features to a 2-class decision
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # output shape: [B, C, 1, 1]
        self.fc1 = nn.Linear(in_channels, self.hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, 2)  # two domains: source, target

        # Initialize weights a bit more deterministically
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input feature map. Expect shape either [B, C, H, W], or
                              [B, C]. In other cases we flatten spatial dims.

        Returns:
            torch.Tensor: Log-probabilities with shape [B, 2], i.e., log p(y|x_t).
        """
        if x.dim() == 4:
            # [B, C, H, W] -> [B, C, 1, 1] via global average pooling
            x_pooled = self.global_pool(x)  # [B, C, 1, 1]
            x_vec = x_pooled.view(x_pooled.size(0), -1)  # [B, C]
        elif x.dim() == 3:
            # [B, C, L] -> pool spatially by averaging across last dim
            x_pooled = x.mean(dim=-1)  # [B, C]
            x_vec = x_pooled
        elif x.dim() == 2:
            # [B, C]
            x_vec = x
        else:
            # Flatten everything except batch
            x_vec = x.view(x.size(0), -1)

        h = self.act(self.fc1(x_vec))
        logits = self.fc2(h)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


__all__ = ["DomainClassifier"]
