"""
Binary classifier for similarity-guided training in DPMs-ANT.

The classifier pϕ distinguishes between source and target domain images
at arbitrary noise levels (timestep t). It is used to compute the
similarity guidance gradient: ∇_xt log pϕ(y=T|xt).

From the paper:
- Based on the classifier architecture from Dhariwal & Nichol (2021)
- Pre-trained on ImageNet, then fine-tuned with a binary classifier head
- Fine-tuned on 10 target domain images across all timesteps T
- Uses noised images xt at timestep t as input

The classifier architecture follows the guided-diffusion classifier:
- Uses the same U-Net encoder architecture (downsampling path)
- Adds attention pooling and a linear classification head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from dpms_ant.models.unet import (
    timestep_embedding,
    GroupNorm32,
    ResBlock,
    AttentionBlock,
    Downsample,
    TimestepEmbedSequential,
)


class AttentionPool2d(nn.Module):
    """
    Attention pooling as used in the guided-diffusion classifier.
    Adapted from CLIP.
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)  # (B, C, H*W)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # (B, C, H*W+1)
        pos_emb = self.positional_embedding[:, : x.shape[-1]].unsqueeze(0)
        x = x + pos_emb

        x = x.permute(2, 0, 1)  # (H*W+1, B, C)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Scaled dot-product attention
        head_dim = c // self.num_heads
        q = q.reshape(-1, b * self.num_heads, head_dim).transpose(0, 1)
        k = k.reshape(-1, b * self.num_heads, head_dim).transpose(0, 1)
        v = v.reshape(-1, b * self.num_heads, head_dim).transpose(0, 1)

        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.bmm(q * scale, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        out = out.transpose(0, 1).reshape(-1, b, c)
        out = out[0]  # Take the class token

        return self.c_proj(out)


class NoisyImageClassifier(nn.Module):
    """
    Binary classifier that operates on noisy images xt at timestep t.

    Architecture: Simplified encoder (ResBlocks + attention + pooling) + linear head.
    This is a lighter-weight classifier suitable for the binary task.

    Args:
        image_size: Input image resolution
        in_channels: Number of input channels
        model_channels: Base channel count
        channel_mult: Channel multiplier per level
        num_res_blocks: ResBlocks per level
        attention_resolutions: Where to add attention
        num_heads: Attention heads
        num_classes: Number of classes (2 for binary)
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (32, 16, 8),
        num_heads: int = 4,
        dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = model_channels
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout,
                             out_channels=mult * model_channels,
                             use_scale_shift_norm=True)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))

            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, use_conv=True))
                )
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=True),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=True),
        )

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

        self.model_channels = model_channels

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Classify noisy image xt at timestep t.

        Args:
            x: Noisy image (B, C, H, W)
            timesteps: Timestep indices (B,)

        Returns:
            Logits (B, num_classes)
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x
        for module in self.input_blocks:
            h = module(h, emb)

        h = self.middle_block(h, emb)
        return self.out(h)

    def get_target_gradient(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        target_label: int = 1,
    ) -> torch.Tensor:
        """
        Compute ∇_x log pϕ(y=T|x) for similarity guidance.

        Args:
            x: Noisy image (B, C, H, W), requires grad
            timesteps: Timestep indices (B,)
            target_label: Label for target domain (default 1)

        Returns:
            Gradient of log probability w.r.t. x, shape (B, C, H, W)
        """
        x = x.detach().requires_grad_(True)

        logits = self.forward(x, timesteps)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[:, target_label].sum()

        grad = torch.autograd.grad(target_log_prob, x)[0]
        return grad


def train_binary_classifier(
    classifier: NoisyImageClassifier,
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    diffusion,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> NoisyImageClassifier:
    """
    Train the binary classifier to distinguish source vs target noisy images.

    The classifier is trained on noised versions of both source and target images
    at random timesteps, following the diffusion noise schedule.

    Args:
        classifier: The classifier model
        source_images: Source domain images (N_s, C, H, W)
        target_images: Target domain images (N_t, C, H, W) - typically 10 images
        diffusion: GaussianDiffusion instance for adding noise
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to train on

    Returns:
        Trained classifier
    """
    classifier = classifier.to(device)
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    n_source = len(source_images)
    n_target = len(target_images)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Sample balanced batch
        half_batch = batch_size // 2

        for _ in range(max(1, n_source // half_batch)):
            # Sample source images
            src_idx = torch.randint(0, n_source, (half_batch,))
            src_imgs = source_images[src_idx].to(device)

            # Sample target images (with replacement since few-shot)
            tgt_idx = torch.randint(0, n_target, (half_batch,))
            tgt_imgs = target_images[tgt_idx].to(device)

            # Combine
            images = torch.cat([src_imgs, tgt_imgs], dim=0)
            labels = torch.cat([
                torch.zeros(half_batch, dtype=torch.long),
                torch.ones(half_batch, dtype=torch.long),
            ]).to(device)

            # Random timestep
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)

            # Add noise
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(images, t, noise=noise)

            # Forward pass
            logits = classifier(x_t, t)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(1, n_batches)
            print(f"  Classifier epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    classifier.eval()
    return classifier
