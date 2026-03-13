import numpy as np
from typing import Callable, Optional

__all__ = ["DiffusionGuidance"]

class DiffusionGuidance:
    """
    Diffusion guidance module implementing Algorithm 1 style constraint guidance.

    Given a constraint function c(xt) that encodes desired properties (e.g., interval
    constraints), this module modifies the score s_phi(xt, t) by a gradient-based term
    derived from log σ(-s(t) * c(xt)).

    - c_fn: function xt -> scalar constraint value
    - grad_c_fn: optional function xt -> gradient of c with respect to xt. If None, a
      simple finite-difference gradient is used.
    - s_schedule: optional function t -> scalar s(t). If None, s(t) defaults to 1.0.
    - strength: scaling factor for the guidance gradient term.
    - eps: finite-difference step for gradient approximation when grad_c_fn is not provided.
    """

    def __init__(
        self,
        c_fn: Optional[Callable[[np.ndarray], float]] = None,
        grad_c_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        s_schedule: Optional[Callable[[float], float]] = None,
        strength: float = 1.0,
        eps: float = 1e-6,
    ):
        self.c_fn = c_fn
        self.grad_c_fn = grad_c_fn
        self.s_schedule = s_schedule if s_schedule is not None else (lambda t: 1.0)
        self.strength = float(strength)
        self.eps = float(eps)

    def _sigmoid(self, z: float) -> float:
        # stable sigmoid for scalar inputs; supports numpy arrays as well
        z = np.asarray(z)
        # clip to avoid overflow in exp
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _grad_c(self, xt: np.ndarray) -> np.ndarray:
        if self.grad_c_fn is not None:
            return np.asarray(self.grad_c_fn(xt))
        # Finite-difference gradient: d/dx c(xt)
        xt = np.asarray(xt, dtype=float)
        grad = np.zeros_like(xt, dtype=float)
        # iterate per-dimension
        for i in range(xt.size):
            dx = np.zeros_like(xt)
            dx[i] = self.eps
            c_plus = self.c_fn(xt + dx) if self.c_fn is not None else 0.0
            c_minus = self.c_fn(xt - dx) if self.c_fn is not None else 0.0
            grad[i] = (c_plus - c_minus) / (2.0 * self.eps)
        return grad

    def guided_score(self, xt: np.ndarray, t: float, s_phi: np.ndarray) -> np.ndarray:
        """Return the guided score s_guided(xt, t) = s_phi(xt, t) + strength * grad_term.

        grad_term = ∇x log σ(-s(t) * c(xt)) where c(xt) encodes the constraint.
        If no constraint is provided (c_fn is None), returns s_phi unchanged.
        """
        if self.c_fn is None:
            return np.asarray(s_phi)

        xt = np.asarray(xt, dtype=float)
        s_t = float(self.s_schedule(t))
        c_xt = float(self.c_fn(xt))
        # Compute gradient of c with respect to xt
        grad_c = self._grad_c(xt)
        # Compute logistic term
        y = - s_t * c_xt
        sigma_y = self._sigmoid(y)
        # dy/dx = - s_t * grad_c
        dy_dx = - s_t * grad_c
        # d/dx log sigma(y) = (1 - sigma(y)) * dy/dx
        grad_log_sigma = (1.0 - sigma_y) * dy_dx
        # Guided score
        s_guided = s_phi + self.strength * grad_log_sigma
        return np.asarray(s_guided)
