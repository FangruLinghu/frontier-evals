#!/usr/bin/env python
"""
Toy 2D experiment from Section 5.1 of the paper.

Reproduces the 2D Gaussian transfer learning experiment:
- Source: 2D Gaussian with mean (1, 1), variance I
- Target: 2D Gaussian with mean (-1, -1), variance I
- Train a simple neural network on source, then transfer to target

This demonstrates:
1. Gradient correction via similarity-guided training
2. Faster convergence via adversarial noise selection
3. Heat map visualization of generated distributions
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm


# --- Simple 2D Denoising Network ---

class Simple2DDenoiser(nn.Module):
    """Simple MLP for 2D denoising."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # Input: (x_t, t_emb) -> predicted noise
        self.net = nn.Sequential(
            nn.Linear(2 + 64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


class Simple2DClassifier(nn.Module):
    """Simple classifier for source vs target 2D data."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # 2 classes: source, target
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


class Simple2DAdaptor(nn.Module):
    """Adaptor for the 2D denoiser."""

    def __init__(self, hidden_dim: int = 128, bottleneck: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_dim)

        # Zero init
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class DenoiserWithAdaptor(nn.Module):
    """Denoiser with adaptor layers."""

    def __init__(self, denoiser: Simple2DDenoiser, bottleneck: int = 8):
        super().__init__()
        self.denoiser = denoiser
        # Freeze denoiser
        for p in self.denoiser.parameters():
            p.requires_grad = False

        # Add adaptor after each hidden layer
        hidden_dim = 128
        self.adaptors = nn.ModuleList([
            Simple2DAdaptor(hidden_dim, bottleneck),
            Simple2DAdaptor(hidden_dim, bottleneck),
            Simple2DAdaptor(hidden_dim, bottleneck),
        ])

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t_emb], dim=-1)
        h = self.denoiser.net[0](inp)  # Linear
        h = self.denoiser.net[1](h)    # SiLU
        h = h + self.adaptors[0](h)

        h = self.denoiser.net[2](h)    # Linear
        h = self.denoiser.net[3](h)    # SiLU
        h = h + self.adaptors[1](h)

        h = self.denoiser.net[4](h)    # Linear
        h = self.denoiser.net[5](h)    # SiLU
        h = h + self.adaptors[2](h)

        h = self.denoiser.net[6](h)    # Final linear
        return h


# --- Diffusion Utilities ---

def get_schedule(T: int = 1000):
    """Get linear beta schedule."""
    betas = np.linspace(0.0001, 0.02, T)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)
    return betas, alphas, alphas_cumprod


def time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Simple sinusoidal time embedding."""
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def q_sample(x0, t, alphas_cumprod, noise=None):
    """Forward diffusion."""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha = torch.sqrt(torch.tensor(alphas_cumprod[t.cpu().numpy()], dtype=torch.float32, device=x0.device))
    sqrt_one_minus_alpha = torch.sqrt(1 - torch.tensor(alphas_cumprod[t.cpu().numpy()], dtype=torch.float32, device=x0.device))
    while len(sqrt_alpha.shape) < len(x0.shape):
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
    return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


# --- Training Functions ---

def train_source(model, source_data, T=1000, epochs=200, lr=1e-3):
    """Train denoiser on source domain."""
    _, _, alphas_cumprod = get_schedule(T)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Sample batch
        idx = torch.randint(0, len(source_data), (256,))
        x0 = source_data[idx]
        t = torch.randint(0, T, (256,))
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, alphas_cumprod, noise)
        t_emb = time_embedding(t)

        eps_pred = model(xt, t_emb)
        loss = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Source training epoch {epoch+1}: Loss = {loss.item():.6f}")


def transfer_baseline(model, target_data, T=1000, iterations=300, lr=5e-5):
    """Baseline transfer: direct fine-tuning."""
    _, _, alphas_cumprod = get_schedule(T)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for it in range(iterations):
        idx = torch.randint(0, len(target_data), (256,))
        x0 = target_data[idx]
        t = torch.randint(0, T, (256,))
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, alphas_cumprod, noise)
        t_emb = time_embedding(t)

        eps_pred = model(xt, t_emb)
        loss = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def transfer_ant(
    model_with_adaptor, classifier, target_data,
    T=1000, iterations=300, lr=5e-5, gamma=5.0, J=10, omega=0.02,
):
    """DPMs-ANT transfer with adversarial noise and similarity guidance."""
    _, alphas, alphas_cumprod = get_schedule(T)
    # Only optimize adaptor parameters
    adaptor_params = [p for p in model_with_adaptor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(adaptor_params, lr=lr)
    losses = []

    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    for it in range(iterations):
        idx = torch.randint(0, len(target_data), (256,))
        x0 = target_data[idx]
        t = torch.randint(0, T, (256,))
        t_emb = time_embedding(t)

        # Adversarial noise selection
        noise = torch.randn_like(x0)
        sqrt_ac = torch.sqrt(torch.tensor(alphas_cumprod[t.cpu().numpy()], dtype=torch.float32)).unsqueeze(-1)
        sqrt_1mac = torch.sqrt(1 - torch.tensor(alphas_cumprod[t.cpu().numpy()], dtype=torch.float32)).unsqueeze(-1)

        for j in range(J):
            noise = noise.detach().requires_grad_(True)
            xt = sqrt_ac * x0 + sqrt_1mac * noise

            with torch.no_grad():
                eps_pred = model_with_adaptor.denoiser(xt, t_emb)

            adv_loss = ((noise - eps_pred) ** 2).sum()
            grad = torch.autograd.grad(adv_loss, noise)[0]
            noise = noise.detach() + omega * grad.detach()
            # Normalize
            noise = noise - noise.mean(dim=-1, keepdim=True)
            noise = noise / (noise.std(dim=-1, keepdim=True) + 1e-8)

        noise = noise.detach()

        # Compute noisy image
        xt = sqrt_ac * x0 + sqrt_1mac * noise

        # Model prediction
        eps_pred = model_with_adaptor(xt, t_emb)

        # Similarity guidance
        xt_for_class = xt.detach().requires_grad_(True)
        logits = classifier(xt_for_class, t_emb.detach())
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[:, 1].sum()
        class_grad = torch.autograd.grad(target_log_prob, xt_for_class)[0]

        # σ̂_t
        alpha_t = torch.tensor(alphas[t.cpu().numpy()], dtype=torch.float32).unsqueeze(-1)
        alpha_bar_t = torch.tensor(alphas_cumprod[t.cpu().numpy()], dtype=torch.float32).unsqueeze(-1)
        alpha_bar_t_prev = torch.tensor(
            np.append(1.0, alphas_cumprod[:-1])[t.cpu().numpy()], dtype=torch.float32
        ).unsqueeze(-1)
        sigma_hat = (1 - alpha_bar_t_prev) * torch.sqrt(alpha_t / (1 - alpha_bar_t + 1e-8))

        target = noise - (sigma_hat ** 2 * gamma * class_grad).detach()
        loss = F.mse_loss(eps_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


@torch.no_grad()
def sample_ddpm(model, T=1000, n_samples=5000):
    """DDPM sampling for 2D data."""
    betas, alphas, alphas_cumprod = get_schedule(T)

    x = torch.randn(n_samples, 2)

    for i in reversed(range(T)):
        t = torch.full((n_samples,), i, dtype=torch.long)
        t_emb = time_embedding(t)

        eps_pred = model(x, t_emb)

        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        beta_t = betas[i]

        mean = (1 / np.sqrt(alpha_t)) * (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * eps_pred)

        if i > 0:
            noise = torch.randn_like(x)
            sigma = np.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean

    return x


# --- Main Experiment ---

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = Path("results/toy_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DPMs-ANT Toy 2D Experiment (Section 5.1)")
    print("=" * 60)

    # Generate source and target data
    print("\n1. Generating data...")
    n_source = 10000
    n_target = 10  # Few-shot!

    source_data = torch.randn(n_source, 2) + torch.tensor([1.0, 1.0])
    target_data = torch.randn(n_target, 2) + torch.tensor([-1.0, -1.0])

    print(f"   Source: {n_source} samples, mean ~(1,1)")
    print(f"   Target: {n_target} samples, mean ~(-1,-1)")

    # Train source model
    print("\n2. Training source model...")
    source_model = Simple2DDenoiser()
    train_source(source_model, source_data, epochs=200)

    # Sample from source model
    print("\n3. Sampling from source model...")
    source_samples = sample_ddpm(source_model, n_samples=5000)
    print(f"   Source samples mean: ({source_samples[:, 0].mean():.3f}, {source_samples[:, 1].mean():.3f})")

    # Train binary classifier
    print("\n4. Training binary classifier...")
    classifier = Simple2DClassifier()
    clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    _, _, alphas_cumprod = get_schedule()

    for epoch in range(100):
        # Source batch
        src_idx = torch.randint(0, n_source, (128,))
        src = source_data[src_idx]
        # Target batch (with replacement)
        tgt_idx = torch.randint(0, n_target, (128,))
        tgt = target_data[tgt_idx]

        data = torch.cat([src, tgt], dim=0)
        labels = torch.cat([torch.zeros(128), torch.ones(128)]).long()

        t = torch.randint(0, 1000, (256,))
        noise = torch.randn_like(data)
        xt = q_sample(data, t, alphas_cumprod, noise)
        t_emb = time_embedding(t)

        logits = classifier(xt, t_emb)
        loss = F.cross_entropy(logits, labels)

        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()

    # Method 1: Baseline (direct fine-tuning)
    print("\n5. Baseline transfer (direct fine-tuning)...")
    import copy
    baseline_model = copy.deepcopy(source_model)
    baseline_losses = transfer_baseline(baseline_model, target_data, iterations=300)
    baseline_samples = sample_ddpm(baseline_model, n_samples=5000)
    print(f"   Baseline samples mean: ({baseline_samples[:, 0].mean():.3f}, {baseline_samples[:, 1].mean():.3f})")

    # Method 2: Similarity-guided only (ANT w/o AN)
    # Adaptor with similarity guidance, gamma tuned for 2D
    print("\n6. Similarity-guided transfer (ANT w/o AN)...")
    sg_model = DenoiserWithAdaptor(copy.deepcopy(source_model))
    sg_losses = transfer_ant(sg_model, classifier, target_data,
                             J=0, iterations=500, lr=5e-4, gamma=1.0)
    sg_samples = sample_ddpm(sg_model, n_samples=5000)
    print(f"   SG samples mean: ({sg_samples[:, 0].mean():.3f}, {sg_samples[:, 1].mean():.3f})")

    # Method 3: Full DPMs-ANT
    print("\n7. Full DPMs-ANT transfer...")
    ant_model = DenoiserWithAdaptor(copy.deepcopy(source_model))
    ant_losses = transfer_ant(ant_model, classifier, target_data,
                              J=5, iterations=500, lr=5e-4, gamma=1.0, omega=0.01)
    ant_samples = sample_ddpm(ant_model, n_samples=5000)
    print(f"   ANT samples mean: ({ant_samples[:, 0].mean():.3f}, {ant_samples[:, 1].mean():.3f})")

    # Plot results
    print("\n8. Creating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Scatter plots
    lim = 4
    for ax in axes[0]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    axes[0, 0].scatter(source_samples[:, 0], source_samples[:, 1], alpha=0.1, s=1, c="blue")
    axes[0, 0].scatter(source_data[:100, 0], source_data[:100, 1], alpha=0.3, s=10, c="blue", marker="x")
    axes[0, 0].scatter(target_data[:, 0], target_data[:, 1], alpha=1.0, s=50, c="red", marker="*")
    axes[0, 0].set_title("Source Model")

    axes[0, 1].scatter(baseline_samples[:, 0], baseline_samples[:, 1], alpha=0.1, s=1, c="blue")
    axes[0, 1].scatter(target_data[:, 0], target_data[:, 1], alpha=1.0, s=50, c="red", marker="*")
    axes[0, 1].set_title("Baseline Transfer")

    axes[0, 2].scatter(ant_samples[:, 0], ant_samples[:, 1], alpha=0.1, s=1, c="orange")
    axes[0, 2].scatter(target_data[:, 0], target_data[:, 1], alpha=1.0, s=50, c="red", marker="*")
    axes[0, 2].set_title("DPMs-ANT Transfer")

    # Row 2: Heat maps (1D marginal)
    bins = np.linspace(-lim, lim, 100)

    for i, (samples, title, color) in enumerate([
        (source_samples[:, 0].numpy(), "Source Model (1D)", "blue"),
        (baseline_samples[:, 0].numpy(), "Baseline (1D)", "cyan"),
        (ant_samples[:, 0].numpy(), "DPMs-ANT (1D)", "orange"),
    ]):
        axes[1, i].hist(samples, bins=bins, density=True, alpha=0.6, color=color)
        # True target distribution
        x = np.linspace(-lim, lim, 200)
        true_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x + 1) ** 2)
        axes[1, i].plot(x, true_pdf, "r--", label="True target")
        axes[1, i].set_title(title)
        axes[1, i].legend()

    plt.suptitle("DPMs-ANT Toy 2D Experiment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "toy_experiment.png", dpi=150)
    print(f"   Saved: {output_dir / 'toy_experiment.png'}")

    # Loss curves
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(baseline_losses, label="Baseline", alpha=0.7)
    ax2.plot(sg_losses, label="ANT w/o AN", alpha=0.7)
    ax2.plot(ant_losses, label="DPMs-ANT", alpha=0.7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "toy_losses.png", dpi=150)
    print(f"   Saved: {output_dir / 'toy_losses.png'}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    true_mean = np.array([-1.0, -1.0])
    for name, samples in [
        ("Source model", source_samples),
        ("Baseline", baseline_samples),
        ("ANT w/o AN", sg_samples),
        ("DPMs-ANT", ant_samples),
    ]:
        mean = samples.mean(dim=0).numpy()
        dist = np.linalg.norm(mean - true_mean)
        print(f"  {name:20s}: mean=({mean[0]:+.3f}, {mean[1]:+.3f}), "
              f"dist to target={dist:.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
