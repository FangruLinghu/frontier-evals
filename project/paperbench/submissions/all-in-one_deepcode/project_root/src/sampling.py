import numpy as np
from typing import Optional, Callable, Dict, Any

class ReverseDiffusionSampler:
    """A lightweight reverse diffusion sampler for p(θ, x).

    This implementation uses Euler–Maruyama discretization over a simple
    forward diffusion process (provided via a VPSDE/VESDE-like interface)
    together with a score network sφ(x, t) that estimates ∇_x log p_t(x | x0).

    Conditioning is supported by clamping observed components at every step.
    """

    def __init__(self, score_function: Callable[[np.ndarray, float], np.ndarray],
                 sde_forward: Any,
                 steps: int = 500,
                 rng_seed: Optional[int] = 0):
        """Initialize the sampler.

        Args:
            score_function: Callable that takes (x, t) and returns sφ(x, t).
            sde_forward: An object implementing methods f(x, t) and g(t).
            steps: Number of Euler–Maruyama steps for the reverse process.
            rng_seed: Seed for deterministic randomness.
        """
        self.score_function = score_function
        self.sde_forward = sde_forward
        self.steps = int(steps)
        self._rng = np.random.default_rng(rng_seed)

    def _to_1d(self, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim == 0:
            return arr.reshape(-1)
        return arr.reshape(-1)

    def sample(self,
               dim: int,
               conditioning: Optional[Dict[str, Any]] = None,
               x_T: Optional[np.ndarray] = None,
               steps: Optional[int] = None) -> np.ndarray:
        """Draw a sample from p(θ, x) by reversing the forward SDE.

        Args:
            dim: Dimensionality of the joint variable vector (θ concatenated with x).
            conditioning: Optional dict with keys:
                - 'mask': boolean array of shape (dim,) indicating which indices are observed
                - 'values': array of shape (dim,) with observed values to clamp for masked indices
            x_T: Optional initial state at time t=1. If None, initializes from standard normal N(0, I).
            steps: Optional override for number of steps; if None uses self.steps.

        Returns:
            x_0: The sample at time t=0 with conditioning applied.
        """
        N = int(dim)
        n_steps = int(self.steps if steps is None else steps)

        # Initialize x at t=1 (the forward process final time) from prior if not supplied
        if x_T is None:
            x = self._to_1d(self._rng.normal(size=(N,)))
        else:
            x = self._to_1d(x_T)
            if x.shape[0] != N:
                x = x.reshape(-1)
                if x.shape[0] != N:
                    raise ValueError("Provided x_T has incompatible dimension with dim")

        # Conditioning helpers
        if conditioning is not None:
            mask = np.asarray(conditioning.get('mask', np.zeros(N, dtype=bool))).astype(bool)
            values = np.asarray(conditioning.get('values', np.zeros(N)))
            if mask.shape[0] != N:
                raise ValueError("conditioning mask must have shape (dim,)")
            if values.shape[0] != N:
                raise ValueError("conditioning values must have shape (dim,)")
        else:
            mask = None
            values = None

        # Time grid: t goes from 1 down to 0 in n_steps steps
        t_values = np.linspace(1.0, 0.0, n_steps + 1)
        for i in range(n_steps):
            t = t_values[i]
            t_next = t_values[i + 1]
            dt = t_next - t  # negative or zero

            # Score and forward drift/diffusion at this time
            s = self.score_function(x, t)
            f = getattr(self.sde_forward, 'f')(x, t)
            g = getattr(self.sde_forward, 'g')(t)

            # Euler–Maruyama update: dx = f dt - g^2 s dt + g dW
            # Use dW ~ N(0, dt); implement with sqrt(-dt)
            dW = self._rng.normal(size=x.shape) * np.sqrt(-dt)
            x = x + (f - (g ** 2) * s) * dt + g * dW

            # Apply conditioning by clamping observed components
            if mask is not None and mask.any():
                x = x.copy()
                x[mask] = values[mask]

        return x

__all__ = ["ReverseDiffusionSampler"]
