"""
Sampling methods for score-based diffusion models.

Implements the reverse SDE sampling using Euler-Maruyama discretization,
as well as convenience functions for sampling different conditionals.
"""

from typing import Optional, Callable, Tuple, Union
import torch
import torch.nn as nn

from simformer.diffusion.sde import SDE, VESDE, VPSDE


class EulerMaruyamaSampler:
    """
    Euler-Maruyama sampler for reverse SDE.

    The reverse SDE is:
        dx_t = [f(x_t, t) - g(t)^2 * s(x_t, t)]dt + g(t)dw̃

    Args:
        sde: The forward SDE
        score_fn: Score function s(x_t, t)
        num_steps: Number of discretization steps
        denoise: Whether to apply final denoising step
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_steps: int = 500,
        denoise: bool = True,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.num_steps = num_steps
        self.denoise = denoise

    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        condition_mask: Optional[torch.Tensor] = None,
        condition_values: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from the reverse SDE.

        Args:
            shape: Shape of samples to generate (batch_size, n_variables)
            device: Device to generate samples on
            condition_mask: Optional mask for conditioned variables (1 = conditioned)
            condition_values: Values for conditioned variables
            return_trajectory: Whether to return the full trajectory

        Returns:
            Samples of shape (batch_size, n_variables)
            If return_trajectory, also returns trajectory of shape (num_steps+1, batch_size, n_variables)
        """
        batch_size = shape[0]

        # Initialize from prior
        x_t = self.sde.prior_sampling(shape, device)

        # Apply conditioning mask if provided
        if condition_mask is not None and condition_values is not None:
            x_t = x_t * (1 - condition_mask) + condition_values * condition_mask

        # Time discretization
        dt = (self.sde.t_max - self.sde.t_min) / self.num_steps
        timesteps = torch.linspace(self.sde.t_max, self.sde.t_min, self.num_steps + 1, device=device)

        # Store trajectory if requested
        if return_trajectory:
            trajectory = [x_t.clone()]

        # Reverse diffusion
        for i in range(self.num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Batch time
            t_batch = torch.full((batch_size,), t, device=device)

            # Compute drift and diffusion
            drift = self.sde.drift(x_t, t_batch)
            diffusion = self.sde.diffusion(t_batch)

            # Compute score
            score = self.score_fn(x_t, t_batch)

            # Euler-Maruyama step
            # dx = [f - g^2 * s] dt + g dW
            if condition_mask is not None:
                # Only update latent variables
                score = score * (1 - condition_mask)

            # Expand diffusion for broadcasting
            while diffusion.dim() < x_t.dim():
                diffusion = diffusion.unsqueeze(-1)

            # Deterministic drift
            drift_term = (drift - diffusion ** 2 * score) * (-dt)

            # Stochastic term
            if i < self.num_steps - 1 or not self.denoise:
                noise = torch.randn_like(x_t)
                diffusion_term = diffusion * math.sqrt(abs(dt)) * noise
            else:
                diffusion_term = 0

            # Update
            x_t = x_t + drift_term + diffusion_term

            # Re-apply conditioning
            if condition_mask is not None and condition_values is not None:
                x_t = x_t * (1 - condition_mask) + condition_values * condition_mask

            if return_trajectory:
                trajectory.append(x_t.clone())

        if return_trajectory:
            return x_t, torch.stack(trajectory)
        return x_t


import math


class ReverseSDE:
    """
    Reverse SDE sampler with more control over the sampling process.

    Supports:
    - Euler-Maruyama sampling
    - Predictor-corrector methods
    - Self-recurrence for improved conditioning
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        num_steps: int = 500,
    ):
        """
        Args:
            sde: The forward SDE
            score_fn: Score function with signature (x_t, t, condition_mask) -> score
            num_steps: Number of discretization steps
        """
        self.sde = sde
        self.score_fn = score_fn
        self.num_steps = num_steps

    def euler_maruyama_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        score: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Single Euler-Maruyama step for the reverse SDE.

        Args:
            x_t: Current state
            t: Current time
            dt: Time step (negative for reverse)
            score: Score at current state
            add_noise: Whether to add stochastic noise

        Returns:
            Updated state
        """
        drift = self.sde.drift(x_t, t)
        diffusion = self.sde.diffusion(t)

        # Expand dimensions
        while diffusion.dim() < x_t.dim():
            diffusion = diffusion.unsqueeze(-1)

        # Reverse drift: f - g^2 * s
        reverse_drift = drift - diffusion ** 2 * score

        # Deterministic update
        x_new = x_t + reverse_drift * dt

        # Stochastic update
        if add_noise:
            noise = torch.randn_like(x_t)
            x_new = x_new + diffusion * math.sqrt(abs(dt)) * noise

        return x_new

    def sample(
        self,
        batch_size: int,
        n_variables: int,
        device: torch.device,
        condition_mask: Optional[torch.Tensor] = None,
        condition_values: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        self_recurrence: int = 0,
    ) -> torch.Tensor:
        """
        Sample from the model using reverse SDE.

        Args:
            batch_size: Number of samples
            n_variables: Number of variables
            device: Device to sample on
            condition_mask: Binary mask (1 = conditioned, 0 = latent)
            condition_values: Values for conditioned variables
            guidance_fn: Optional guidance function for constraints
            self_recurrence: Number of self-recurrence steps (0 = disabled)

        Returns:
            Samples of shape (batch_size, n_variables)
        """
        shape = (batch_size, n_variables)

        # Initialize from prior
        x_t = self.sde.prior_sampling(shape, device)

        # Setup condition mask
        if condition_mask is None:
            condition_mask = torch.zeros(batch_size, n_variables, device=device)

        # Apply initial conditioning
        if condition_values is not None:
            x_t = x_t * (1 - condition_mask) + condition_values * condition_mask

        # Time discretization
        dt = -(self.sde.t_max - self.sde.t_min) / self.num_steps  # Negative for reverse
        timesteps = torch.linspace(self.sde.t_max, self.sde.t_min, self.num_steps + 1, device=device)

        # Sampling loop
        for i in range(self.num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = torch.full((batch_size,), t, device=device)

            # Self-recurrence loop
            for r in range(max(1, self_recurrence)):
                # Compute score
                score = self.score_fn(x_t, t_batch, condition_mask)

                # Apply guidance if provided
                if guidance_fn is not None:
                    guidance_score = guidance_fn(x_t, t_batch)
                    score = score + guidance_score

                # Mask score for conditioned variables
                score = score * (1 - condition_mask)

                # Euler-Maruyama step
                add_noise = (i < self.num_steps - 1) or (r < self_recurrence - 1)
                x_t = self.euler_maruyama_step(x_t, t_batch, dt, score, add_noise)

                # Re-apply conditioning
                if condition_values is not None:
                    x_t = x_t * (1 - condition_mask) + condition_values * condition_mask

                # If self-recurrence, resample forward
                if r < self_recurrence - 1:
                    # Add noise back (forward step)
                    noise = torch.randn_like(x_t)
                    diffusion = self.sde.diffusion(t_batch)
                    while diffusion.dim() < x_t.dim():
                        diffusion = diffusion.unsqueeze(-1)
                    x_t = x_t + diffusion * math.sqrt(abs(dt)) * noise

                    # Re-apply conditioning
                    if condition_values is not None:
                        x_t = x_t * (1 - condition_mask) + condition_values * condition_mask

        return x_t


def sample_posterior(
    model: nn.Module,
    sde: SDE,
    x_observed: torch.Tensor,
    num_samples: int = 1000,
    num_steps: int = 500,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample from the posterior p(θ|x).

    Args:
        model: Simformer model
        sde: SDE for diffusion
        x_observed: Observed data of shape (batch_size, n_data) or (n_data,)
        num_samples: Number of posterior samples to generate
        num_steps: Number of SDE steps
        device: Device to sample on

    Returns:
        Posterior samples of shape (num_samples, n_params)
    """
    if device is None:
        device = next(model.parameters()).device

    if x_observed.dim() == 1:
        x_observed = x_observed.unsqueeze(0)

    batch_size = x_observed.shape[0]

    # Expand x_observed if num_samples > batch_size
    if num_samples > batch_size:
        repeat_factor = (num_samples + batch_size - 1) // batch_size
        x_observed = x_observed.repeat(repeat_factor, 1)[:num_samples]

    # Create condition mask (condition on data)
    n_params = model.n_params
    n_data = model.n_data
    condition_mask = torch.zeros(num_samples, n_params + n_data, device=device)
    condition_mask[:, n_params:] = 1.0

    # Condition values
    condition_values = torch.zeros(num_samples, n_params + n_data, device=device)
    condition_values[:, n_params:] = x_observed

    # Define score function
    def score_fn(x_t, t, cond_mask):
        theta_t = x_t[:, :n_params]
        x_data = condition_values[:, n_params:]
        return model.forward(theta_t, x_data, t, cond_mask)

    # Create sampler and sample
    sampler = ReverseSDE(sde, score_fn, num_steps)
    samples = sampler.sample(
        num_samples, n_params + n_data, device,
        condition_mask, condition_values
    )

    # Return only parameter samples
    return samples[:, :n_params]


def sample_likelihood(
    model: nn.Module,
    sde: SDE,
    theta_observed: torch.Tensor,
    num_samples: int = 1000,
    num_steps: int = 500,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample from the likelihood p(x|θ).

    Args:
        model: Simformer model
        sde: SDE for diffusion
        theta_observed: Observed parameters of shape (batch_size, n_params) or (n_params,)
        num_samples: Number of likelihood samples to generate
        num_steps: Number of SDE steps
        device: Device to sample on

    Returns:
        Likelihood samples of shape (num_samples, n_data)
    """
    if device is None:
        device = next(model.parameters()).device

    if theta_observed.dim() == 1:
        theta_observed = theta_observed.unsqueeze(0)

    batch_size = theta_observed.shape[0]

    # Expand theta_observed if num_samples > batch_size
    if num_samples > batch_size:
        repeat_factor = (num_samples + batch_size - 1) // batch_size
        theta_observed = theta_observed.repeat(repeat_factor, 1)[:num_samples]

    # Create condition mask (condition on parameters)
    n_params = model.n_params
    n_data = model.n_data
    condition_mask = torch.zeros(num_samples, n_params + n_data, device=device)
    condition_mask[:, :n_params] = 1.0

    # Condition values
    condition_values = torch.zeros(num_samples, n_params + n_data, device=device)
    condition_values[:, :n_params] = theta_observed

    # Define score function
    def score_fn(x_t, t, cond_mask):
        theta_data = condition_values[:, :n_params]
        x_t_data = x_t[:, n_params:]
        return model.forward(theta_data, x_t_data, t, cond_mask)

    # Create sampler and sample
    sampler = ReverseSDE(sde, score_fn, num_steps)
    samples = sampler.sample(
        num_samples, n_params + n_data, device,
        condition_mask, condition_values
    )

    # Return only data samples
    return samples[:, n_params:]


def sample_joint(
    model: nn.Module,
    sde: SDE,
    num_samples: int = 1000,
    num_steps: int = 500,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from the joint distribution p(θ, x).

    Args:
        model: Simformer model
        sde: SDE for diffusion
        num_samples: Number of joint samples to generate
        num_steps: Number of SDE steps
        device: Device to sample on

    Returns:
        Tuple of (theta_samples, x_samples)
    """
    if device is None:
        device = next(model.parameters()).device

    n_params = model.n_params
    n_data = model.n_data

    # No conditioning
    condition_mask = torch.zeros(num_samples, n_params + n_data, device=device)

    # Define score function
    def score_fn(x_t, t, cond_mask):
        theta_t = x_t[:, :n_params]
        x_t_data = x_t[:, n_params:]
        return model.forward(theta_t, x_t_data, t, cond_mask)

    # Create sampler and sample
    sampler = ReverseSDE(sde, score_fn, num_steps)
    samples = sampler.sample(
        num_samples, n_params + n_data, device,
        condition_mask, None
    )

    return samples[:, :n_params], samples[:, n_params:]


def sample_arbitrary_conditional(
    model: nn.Module,
    sde: SDE,
    condition_mask: torch.Tensor,
    condition_values: torch.Tensor,
    num_samples: int = 1000,
    num_steps: int = 500,
    device: Optional[torch.device] = None,
    guidance_fn: Optional[Callable] = None,
    self_recurrence: int = 0,
) -> torch.Tensor:
    """
    Sample from an arbitrary conditional distribution.

    This is the most general sampling function that can sample any
    conditional of the joint p(θ, x).

    Args:
        model: Simformer model
        sde: SDE for diffusion
        condition_mask: Binary mask of shape (n_variables,) or (num_samples, n_variables)
                       1 = conditioned, 0 = latent
        condition_values: Values for conditioned variables
        num_samples: Number of samples to generate
        num_steps: Number of SDE steps
        device: Device to sample on
        guidance_fn: Optional guidance function for constraints
        self_recurrence: Number of self-recurrence steps

    Returns:
        Samples of shape (num_samples, n_variables)
    """
    if device is None:
        device = next(model.parameters()).device

    n_params = model.n_params
    n_data = model.n_data
    n_variables = n_params + n_data

    # Expand condition_mask if needed
    if condition_mask.dim() == 1:
        condition_mask = condition_mask.unsqueeze(0).expand(num_samples, -1)
    condition_mask = condition_mask.to(device)

    # Expand condition_values if needed
    if condition_values.dim() == 1:
        condition_values = condition_values.unsqueeze(0).expand(num_samples, -1)
    condition_values = condition_values.to(device)

    # Define score function
    def score_fn(x_t, t, cond_mask):
        theta_t = x_t[:, :n_params]
        x_t_data = x_t[:, n_params:]
        return model.forward(theta_t, x_t_data, t, cond_mask)

    # Create sampler and sample
    sampler = ReverseSDE(sde, score_fn, num_steps)
    samples = sampler.sample(
        num_samples, n_variables, device,
        condition_mask, condition_values,
        guidance_fn, self_recurrence
    )

    return samples
