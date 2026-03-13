"""
Gaussian Diffusion process for DDPM/DDIM.

Implements the forward (noising) and reverse (denoising) processes.

From the paper (Section 3):
    q(xt|x0) = N(xt; sqrt(αbar_t) * x0, (1 - αbar_t) * I)
    xt = sqrt(αbar_t) * x0 + sqrt(1 - αbar_t) * ε

Reverse process:
    xt-1 = sqrt(αbar_{t-1}) * predicted_x0
            + sqrt(1 - αbar_{t-1} - σ²_t) * ε_θ(xt,t)
            + σ_t * ε_t
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from tqdm import tqdm


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    """
    Get a pre-defined beta schedule.

    Args:
        schedule_name: "linear" or "cosine"
        num_diffusion_timesteps: T, number of diffusion steps

    Returns:
        Array of beta values
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        max_beta = 0.999
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            alpha_bar_t1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar_t2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
            betas.append(min(1 - alpha_bar_t2 / alpha_bar_t1, max_beta))
        return np.array(betas, dtype=np.float64)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_name}")


class GaussianDiffusion:
    """
    Gaussian diffusion process.

    Handles both the forward noising process and the reverse denoising process.

    Args:
        betas: Array of beta values for each timestep
    """

    def __init__(self, betas: np.ndarray):
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Calculations for diffusion q(xt | x0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | xt, x0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

        # For similarity guidance: σ̂_t = (1 - ᾱ_{t-1}) * sqrt(α_t / (1 - ᾱ_t))
        self.sigma_hat = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas / (1.0 - self.alphas_cumprod))
        )

    def _extract(self, arr: np.ndarray, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from array at timestep t and reshape for broadcasting."""
        res = torch.from_numpy(arr).to(device=t.device, dtype=torch.float32)[t]
        while len(res.shape) < len(x_shape):
            res = res[..., None]
        return res

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(xt | x0).

        Args:
            x_start: Clean image x0
            t: Timestep
            noise: Optional pre-sampled noise

        Returns:
            Noisy image xt
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x0 from noise prediction."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mean and variance of q(x_{t-1} | xt, x0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the mean and variance of p_θ(x_{t-1} | xt).

        Args:
            model: Denoising model
            x: Current noisy image
            t: Current timestep
            clip_denoised: Whether to clip predicted x0 to [-1, 1]

        Returns:
            Dict with "mean", "variance", "log_variance", "pred_xstart"
        """
        model_output = model(x, t)

        # If model outputs both mean and variance
        if model_output.shape[1] == x.shape[1] * 2:
            eps, model_var_values = torch.split(model_output, x.shape[1], dim=1)
            # Learned variance interpolation
            min_log = self._extract(self.posterior_log_variance_clipped, t, x.shape)
            max_log = self._extract(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2  # model_var_values in [-1, 1]
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            eps = model_output
            model_variance = self._extract(self.posterior_variance, t, x.shape)
            model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)

        pred_xstart = self.predict_x0_from_eps(x, t, eps)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Sample from p_θ(x_{t-1} | xt).

        Args:
            model: Denoising model
            x: Current noisy image
            t: Current timestep
            clip_denoised: Whether to clip predicted x0

        Returns:
            Sampled x_{t-1}
        """
        out = self.p_mean_variance(model, x, t, clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        clip_denoised: bool = True,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Full reverse sampling loop (DDPM).

        Args:
            model: Denoising model
            shape: Shape of the output (B, C, H, W)
            device: Device
            clip_denoised: Whether to clip
            progress: Whether to show progress bar

        Returns:
            Generated samples
        """
        x = torch.randn(shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices, desc="Sampling")

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            x = self.p_sample(model, x, t, clip_denoised)

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        ddim_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        DDIM sampling (Song et al., 2020).

        Args:
            model: Denoising model
            shape: Output shape
            device: Device
            ddim_steps: Number of DDIM steps
            eta: DDIM stochasticity parameter (0 = deterministic)
            clip_denoised: Whether to clip
            progress: Show progress

        Returns:
            Generated samples
        """
        # Create DDIM timestep subsequence
        c = self.num_timesteps // ddim_steps
        ddim_timesteps = list(range(0, self.num_timesteps, c))

        x = torch.randn(shape, device=device)

        indices = list(range(len(ddim_timesteps)))[::-1]
        if progress:
            indices = tqdm(indices, desc="DDIM Sampling")

        for i in indices:
            t = torch.tensor([ddim_timesteps[i]] * shape[0], device=device)

            model_output = model(x, t)
            if model_output.shape[1] == x.shape[1] * 2:
                eps, _ = torch.split(model_output, x.shape[1], dim=1)
            else:
                eps = model_output

            # Predict x0
            pred_xstart = self.predict_x0_from_eps(x, t, eps)
            if clip_denoised:
                pred_xstart = pred_xstart.clamp(-1, 1)

            alpha_bar = self._extract(self.alphas_cumprod, t, x.shape)

            if i > 0:
                t_prev = torch.tensor([ddim_timesteps[i - 1]] * shape[0], device=device)
                alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar)

            # DDIM step
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )

            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * pred_xstart + dir_xt + sigma * noise

        return x

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for DDPM.

        Standard denoising loss: ||ε - ε_θ(xt, t)||²

        Args:
            model: Denoising model
            x_start: Clean images
            t: Timesteps
            noise: Optional noise

        Returns:
            Dict with "loss" and "mse" keys
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)
        model_output = model(x_t, t)

        if model_output.shape[1] == x_start.shape[1] * 2:
            eps_pred, _ = torch.split(model_output, x_start.shape[1], dim=1)
        else:
            eps_pred = model_output

        mse = (noise - eps_pred) ** 2
        loss = mse.mean()

        return {"loss": loss, "mse": mse.mean(dim=[1, 2, 3])}


def create_diffusion(
    timestep_respacing: str = "",
    noise_schedule: str = "linear",
    num_diffusion_timesteps: int = 1000,
) -> GaussianDiffusion:
    """
    Create a GaussianDiffusion instance.

    Args:
        timestep_respacing: Optional respacing string
        noise_schedule: Beta schedule type
        num_diffusion_timesteps: Number of timesteps T

    Returns:
        GaussianDiffusion instance
    """
    betas = get_named_beta_schedule(noise_schedule, num_diffusion_timesteps)
    return GaussianDiffusion(betas)
