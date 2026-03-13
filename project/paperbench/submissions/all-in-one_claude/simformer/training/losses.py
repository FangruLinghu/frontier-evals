"""
Loss functions for training Simformer.

Implements denoising score matching loss as described in the paper.

The loss is:
    L(φ) = E_{MC, t, x̂_0, x̂_t} [ ||ℓ(φ, MC, t, x̂_0, x̂_t)||^2 ]

where:
    ℓ(φ, MC, t, x̂_0, x̂_t) = (1 - MC) · (s_φ^{ME}(x̂_t^{MC}, t) - ∇_{x̂_t} log p_t(x̂_t|x̂_0))

This objective requires samples from the original distribution x̂_0 ~ p(x̂).
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from simformer.diffusion.sde import SDE


def denoising_score_matching_loss(
    score_pred: torch.Tensor,
    noise: torch.Tensor,
    std: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute denoising score matching loss.

    The true conditional score is:
        ∇_{x_t} log p(x_t | x_0) = -noise / σ_t

    The loss is:
        || score_pred - (-noise / σ_t) ||^2

    Args:
        score_pred: Predicted score of shape (batch_size, n_variables)
        noise: The noise that was added, shape (batch_size, n_variables)
        std: Standard deviation σ_t, shape (batch_size,) or broadcastable
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Expand std for broadcasting
    while std.dim() < noise.dim():
        std = std.unsqueeze(-1)

    # True score: -noise / σ_t
    true_score = -noise / (std + 1e-8)

    # MSE loss
    loss = (score_pred - true_score) ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def conditional_score_matching_loss(
    score_pred: torch.Tensor,
    noise: torch.Tensor,
    std: torch.Tensor,
    condition_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute conditional score matching loss.

    Only computes loss for latent (unconditioned) variables.

    Args:
        score_pred: Predicted score
        noise: Added noise
        std: Standard deviation σ_t
        condition_mask: Binary mask (1 = conditioned, 0 = latent)
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Expand std for broadcasting
    while std.dim() < noise.dim():
        std = std.unsqueeze(-1)

    # True score for latent variables
    true_score = -noise / (std + 1e-8)

    # Mask: only compute loss for latent variables
    latent_mask = 1 - condition_mask

    # Squared error
    squared_error = (score_pred - true_score) ** 2

    # Masked squared error
    masked_error = squared_error * latent_mask

    if reduction == "mean":
        # Mean over latent variables
        num_latent = latent_mask.sum() + 1e-8
        return masked_error.sum() / num_latent
    elif reduction == "sum":
        return masked_error.sum()
    else:
        return masked_error


class SimformerLoss(nn.Module):
    """
    Complete loss function for Simformer training.

    Handles:
    - Sampling diffusion time
    - Perturbing data
    - Sampling condition masks
    - Computing denoising score matching loss

    Args:
        sde: The SDE for diffusion
        n_params: Number of parameter variables
        n_data: Number of data variables
        weighting: Loss weighting scheme ("uniform", "likelihood", etc.)
    """

    def __init__(
        self,
        sde: SDE,
        n_params: int,
        n_data: int,
        weighting: str = "uniform",
    ):
        super().__init__()
        self.sde = sde
        self.n_params = n_params
        self.n_data = n_data
        self.n_variables = n_params + n_data
        self.weighting = weighting

    def sample_condition_mask(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample condition masks for training.

        As described in the paper, we uniformly sample:
        - Joint mask (all zeros)
        - Posterior mask (data conditioned)
        - Likelihood mask (parameters conditioned)
        - Random masks with p=0.3 or p=0.7
        """
        masks = torch.zeros(batch_size, self.n_variables, device=device)

        for i in range(batch_size):
            choice = torch.randint(0, 4, (1,)).item()

            if choice == 0:
                # Joint: all latent
                pass
            elif choice == 1:
                # Posterior: condition on data
                masks[i, self.n_params:] = 1.0
            elif choice == 2:
                # Likelihood: condition on parameters
                masks[i, :self.n_params] = 1.0
            else:
                # Random mask
                p = 0.3 if torch.rand(1).item() < 0.5 else 0.7
                masks[i] = (torch.rand(self.n_variables, device=device) < p).float()

        return masks

    def get_weighting(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get loss weighting based on time.

        Args:
            t: Time values

        Returns:
            Weighting values
        """
        if self.weighting == "uniform":
            return torch.ones_like(t)
        elif self.weighting == "likelihood":
            # Weight by g(t)^2 for likelihood weighting
            g = self.sde.diffusion(t)
            return g ** 2
        elif self.weighting == "snr":
            # Signal-to-noise ratio weighting
            mean, std = self.sde.marginal_prob(torch.zeros_like(t), t)
            snr = (mean / (std + 1e-8)) ** 2
            return snr / (snr + 1)
        else:
            return torch.ones_like(t)

    def forward(
        self,
        model: nn.Module,
        theta: torch.Tensor,
        x: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the training loss.

        Args:
            model: Simformer model
            theta: Parameter values of shape (batch_size, n_params)
            x: Data values of shape (batch_size, n_data)
            condition_mask: Optional pre-specified condition mask

        Returns:
            Dictionary containing loss and optional metrics
        """
        batch_size = theta.shape[0]
        device = theta.device

        # Concatenate theta and x
        x_hat = torch.cat([theta, x], dim=-1)  # (batch_size, n_variables)

        # Sample condition mask if not provided
        if condition_mask is None:
            condition_mask = self.sample_condition_mask(batch_size, device)

        # Sample time
        t = self.sde.sample_time(batch_size, device)

        # Perturb data (only latent variables)
        mean, std = self.sde.marginal_prob(x_hat, t)
        noise = torch.randn_like(x_hat)

        # Create partially noisy sample: conditioned variables stay clean
        x_hat_t = mean + std * noise
        x_hat_t = x_hat_t * (1 - condition_mask) + x_hat * condition_mask

        # Split back to theta and x
        theta_t = x_hat_t[:, :self.n_params]
        x_t = x_hat_t[:, self.n_params:]

        # Forward pass through model
        score_pred = model(theta_t, x_t, t, condition_mask)

        # Compute loss only for latent variables
        loss = conditional_score_matching_loss(
            score_pred, noise, std, condition_mask, reduction="none"
        )

        # Apply weighting
        weight = self.get_weighting(t)
        while weight.dim() < loss.dim():
            weight = weight.unsqueeze(-1)
        loss = loss * weight

        # Mean loss
        latent_mask = 1 - condition_mask
        num_latent = latent_mask.sum() + 1e-8
        loss_value = loss.sum() / num_latent

        return {
            "loss": loss_value,
            "mse": (score_pred - (-noise / (std + 1e-8))) ** 2 * latent_mask,
        }


class JointLoss(nn.Module):
    """
    Loss function specifically for joint distribution training.
    """

    def __init__(self, sde: SDE, n_params: int, n_data: int):
        super().__init__()
        self.sde = sde
        self.n_params = n_params
        self.n_data = n_data

    def forward(
        self,
        model: nn.Module,
        theta: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = theta.shape[0]
        device = theta.device

        # No conditioning
        condition_mask = torch.zeros(
            batch_size, self.n_params + self.n_data, device=device
        )

        # Sample time and perturb
        t = self.sde.sample_time(batch_size, device)
        x_hat = torch.cat([theta, x], dim=-1)
        mean, std = self.sde.marginal_prob(x_hat, t)
        noise = torch.randn_like(x_hat)
        x_hat_t = mean + std * noise

        # Forward
        theta_t = x_hat_t[:, :self.n_params]
        x_t = x_hat_t[:, self.n_params:]
        score_pred = model(theta_t, x_t, t, condition_mask)

        # Loss
        return denoising_score_matching_loss(score_pred, noise, std)


class PosteriorLoss(nn.Module):
    """
    Loss function specifically for posterior training.
    """

    def __init__(self, sde: SDE, n_params: int, n_data: int):
        super().__init__()
        self.sde = sde
        self.n_params = n_params
        self.n_data = n_data

    def forward(
        self,
        model: nn.Module,
        theta: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = theta.shape[0]
        device = theta.device

        # Condition on data
        condition_mask = torch.zeros(
            batch_size, self.n_params + self.n_data, device=device
        )
        condition_mask[:, self.n_params:] = 1.0

        # Sample time and perturb only theta
        t = self.sde.sample_time(batch_size, device)
        mean, std = self.sde.marginal_prob(theta, t)
        noise = torch.randn_like(theta)
        theta_t = mean + std * noise

        # Forward
        score_pred = model(theta_t, x, t, condition_mask)

        # Loss only for theta
        score_theta = score_pred[:, :self.n_params]
        return denoising_score_matching_loss(score_theta, noise, std)
