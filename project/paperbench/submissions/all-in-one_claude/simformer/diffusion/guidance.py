"""
Diffusion Guidance for Simformer.

Implements guided diffusion to condition on intervals and arbitrary constraints,
as described in the paper (Section 3.4).

The guided score is:
    s(x_t, t | c) ≈ s_φ(x_t, t) + ∇_{x_t} log σ(-s(t) * c(x_t))

where:
- c(x) is a constraint function with c(x) ≤ 0
- s(t) is a scaling function
- σ is the sigmoid function
"""

from typing import Callable, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from simformer.diffusion.sde import SDE, VESDE, VPSDE


class DiffusionGuidance:
    """
    Base class for diffusion guidance.

    Provides methods to modify the score function to guide the diffusion
    process towards satisfying certain constraints.
    """

    def __init__(self, sde: SDE, scale_fn: Optional[Callable] = None):
        """
        Args:
            sde: The SDE being used
            scale_fn: Function s(t) that scales the guidance strength.
                     If None, uses default scaling based on SDE type.
        """
        self.sde = sde
        self.scale_fn = scale_fn or self._default_scale_fn

    def _default_scale_fn(self, t: torch.Tensor) -> torch.Tensor:
        """
        Default scaling function s(t) = 1/σ(t)^2.

        This is inversely proportional to the variance of the marginal score.
        """
        if isinstance(self.sde, VESDE):
            sigma_t = self.sde.sigma(t)
            return 1.0 / (sigma_t ** 2 + 1e-8)
        elif isinstance(self.sde, VPSDE):
            _, std = self.sde.marginal_prob(torch.zeros(1), t)
            return 1.0 / (std ** 2 + 1e-8)
        else:
            # Fallback: use 1/(1-t) which increases as t -> 0
            return 1.0 / (1 - t + 1e-3)

    def compute_guidance_score(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        constraint_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the guidance score for a constraint function.

        Args:
            x_t: Current noisy state
            t: Current time
            constraint_fn: Function c(x) where the constraint is c(x) ≤ 0

        Returns:
            Guidance score ∇_{x_t} log σ(-s(t) * c(x_t))
        """
        x_t = x_t.requires_grad_(True)

        # Denoise x_t to get approximate x_0
        mean, std = self.sde.marginal_prob(torch.zeros_like(x_t), t)
        x_denoised = x_t / (mean + 1e-8)  # Simple denoising estimate

        # Compute constraint
        constraint = constraint_fn(x_denoised)

        # Scale and apply sigmoid
        scale = self.scale_fn(t)
        if scale.dim() < constraint.dim():
            scale = scale.unsqueeze(-1)

        # log σ(-s(t) * c(x))
        log_prob = F.logsigmoid(-scale * constraint)

        # Sum over dimensions if needed
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)

        # Compute gradient
        guidance_score = torch.autograd.grad(
            log_prob.sum(), x_t, create_graph=False
        )[0]

        x_t.requires_grad_(False)

        return guidance_score


class IntervalGuidance(DiffusionGuidance):
    """
    Guidance for interval constraints.

    Constrains variables to lie within specified intervals [lower, upper].

    Example: Constrain energy consumption to be below a threshold.
    """

    def __init__(
        self,
        sde: SDE,
        lower_bounds: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None,
        variable_indices: Optional[List[int]] = None,
        scale_fn: Optional[Callable] = None,
    ):
        """
        Args:
            sde: The SDE being used
            lower_bounds: Lower bounds for each constrained variable
            upper_bounds: Upper bounds for each constrained variable
            variable_indices: Indices of variables to constrain (None = all)
            scale_fn: Scaling function for guidance strength
        """
        super().__init__(sde, scale_fn)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.variable_indices = variable_indices

    def __call__(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute guidance score for interval constraints.

        Args:
            x_t: Current noisy state
            t: Current time

        Returns:
            Guidance score
        """
        guidance_score = torch.zeros_like(x_t)
        device = x_t.device

        # Get variables to constrain
        if self.variable_indices is not None:
            indices = self.variable_indices
        else:
            indices = list(range(x_t.shape[-1]))

        # Denoise estimate
        x_t_req = x_t.clone().requires_grad_(True)
        mean, std = self.sde.marginal_prob(torch.zeros_like(x_t_req), t)

        # For VESDE, mean = x_0, so x_denoised ≈ x_t
        # For VPSDE, need to rescale
        if isinstance(self.sde, VESDE):
            x_denoised = x_t_req
        else:
            while mean.dim() < x_t_req.dim():
                mean = mean.unsqueeze(-1)
            x_denoised = x_t_req / (mean + 1e-8)

        # Compute constraint violations
        log_prob = torch.zeros(x_t_req.shape[0], device=device)
        scale = self.scale_fn(t)

        for idx in indices:
            x_var = x_denoised[:, idx]

            # Upper bound constraint: x - upper ≤ 0
            if self.upper_bounds is not None:
                upper = self.upper_bounds[idx].to(device) if isinstance(self.upper_bounds, torch.Tensor) else self.upper_bounds
                constraint_upper = x_var - upper
                log_prob = log_prob + F.logsigmoid(-scale * constraint_upper)

            # Lower bound constraint: lower - x ≤ 0
            if self.lower_bounds is not None:
                lower = self.lower_bounds[idx].to(device) if isinstance(self.lower_bounds, torch.Tensor) else self.lower_bounds
                constraint_lower = lower - x_var
                log_prob = log_prob + F.logsigmoid(-scale * constraint_lower)

        # Compute gradient
        if log_prob.requires_grad:
            guidance_score = torch.autograd.grad(
                log_prob.sum(), x_t_req, create_graph=False
            )[0]
        else:
            guidance_score = torch.zeros_like(x_t)

        return guidance_score


class ConstraintGuidance(DiffusionGuidance):
    """
    General constraint guidance.

    Supports arbitrary constraint functions c(x) ≤ 0.
    """

    def __init__(
        self,
        sde: SDE,
        constraint_fns: List[Callable[[torch.Tensor], torch.Tensor]],
        scale_fn: Optional[Callable] = None,
    ):
        """
        Args:
            sde: The SDE being used
            constraint_fns: List of constraint functions c_i(x) where c_i(x) ≤ 0
            scale_fn: Scaling function for guidance strength
        """
        super().__init__(sde, scale_fn)
        self.constraint_fns = constraint_fns

    def __call__(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute guidance score for all constraints.

        Args:
            x_t: Current noisy state
            t: Current time

        Returns:
            Combined guidance score
        """
        x_t_req = x_t.clone().requires_grad_(True)
        device = x_t.device

        # Denoise estimate
        mean, std = self.sde.marginal_prob(torch.zeros_like(x_t_req), t)
        if isinstance(self.sde, VESDE):
            x_denoised = x_t_req
        else:
            while mean.dim() < x_t_req.dim():
                mean = mean.unsqueeze(-1)
            x_denoised = x_t_req / (mean + 1e-8)

        # Compute log probability for all constraints
        log_prob = torch.zeros(x_t_req.shape[0], device=device)
        scale = self.scale_fn(t)

        for constraint_fn in self.constraint_fns:
            constraint = constraint_fn(x_denoised)

            # Handle both scalar and vector constraints
            if constraint.dim() > 1:
                constraint = constraint.sum(dim=-1)

            log_prob = log_prob + F.logsigmoid(-scale * constraint)

        # Compute gradient
        if log_prob.requires_grad:
            guidance_score = torch.autograd.grad(
                log_prob.sum(), x_t_req, create_graph=False
            )[0]
        else:
            guidance_score = torch.zeros_like(x_t)

        return guidance_score


class CompositeGuidance:
    """
    Combine multiple guidance functions.
    """

    def __init__(self, guidances: List[DiffusionGuidance]):
        """
        Args:
            guidances: List of guidance functions
        """
        self.guidances = guidances

    def __call__(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined guidance score.

        Args:
            x_t: Current noisy state
            t: Current time

        Returns:
            Sum of all guidance scores
        """
        total_guidance = torch.zeros_like(x_t)

        for guidance in self.guidances:
            total_guidance = total_guidance + guidance(x_t, t)

        return total_guidance


def create_interval_constraint(
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    variable_idx: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a constraint function for an interval bound.

    Args:
        lower: Lower bound (None for no lower bound)
        upper: Upper bound (None for no upper bound)
        variable_idx: Index of variable to constrain

    Returns:
        Constraint function
    """
    def constraint(x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(x.shape[0], device=x.device)

        if upper is not None:
            # x - upper ≤ 0
            total = total + F.relu(x[:, variable_idx] - upper)

        if lower is not None:
            # lower - x ≤ 0
            total = total + F.relu(lower - x[:, variable_idx])

        return total

    return constraint


def create_linear_constraint(
    coefficients: torch.Tensor,
    bound: float,
    constraint_type: str = "leq",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a linear constraint function: a^T x ≤ b or a^T x ≥ b.

    Args:
        coefficients: Coefficients a
        bound: Bound value b
        constraint_type: "leq" for ≤, "geq" for ≥

    Returns:
        Constraint function
    """
    def constraint(x: torch.Tensor) -> torch.Tensor:
        # a^T x
        linear_term = (x * coefficients.to(x.device)).sum(dim=-1)

        if constraint_type == "leq":
            # a^T x - b ≤ 0
            return linear_term - bound
        else:
            # b - a^T x ≤ 0
            return bound - linear_term

    return constraint


def create_equality_constraint(
    variable_idx: int,
    value: float,
    tolerance: float = 0.01,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create an approximate equality constraint using a small interval.

    Args:
        variable_idx: Index of variable
        value: Target value
        tolerance: Tolerance around the value

    Returns:
        Constraint function (actually two inequality constraints)
    """
    def constraint(x: torch.Tensor) -> torch.Tensor:
        diff = x[:, variable_idx] - value
        # |x - value| - tolerance ≤ 0
        return torch.abs(diff) - tolerance

    return constraint
