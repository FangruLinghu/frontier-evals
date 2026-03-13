import numpy as np
from typing import Optional, Any, Tuple, Dict, Callable
import pickle
from pathlib import Path


class TrainingLoop:
    """
    Lightweight training loop scaffold for joint diffusion score modeling.

    This is a minimal, deterministic runner intended for unit tests and quick
    demonstrations. It wires together a data sampler (generates (theta, x0)),
    a forward diffusion process (provides x_t and analytical score terms), and a
    score network (callable accepting (x, t) and returning a score vector).

    The loop does not perform full optimizer-based training; instead it computes a
    simple loss based on a score-matching-like objective and returns it. It can be
    extended with an actual optimizer if needed in user projects.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray, float], np.ndarray],
        data_sampler: Any,
        sde_forward: Any = None,
        loss_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        optimizer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.model = model
        self.data_sampler = data_sampler
        # Forward diffusion core (VPSDE/VESDE) – default to VPSDE if not supplied
        from .diffusion_core import VPSDE
        self.sde_forward = sde_forward if sde_forward is not None else VPSDE()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = {
            "batch_size": 16,
            "epochs": 1,
            "steps_forward": 50,
            "mc_prob": 0.5,  # probability of a given dimension being masked/observed
            "device": "cpu",
            "seed": 0,
        }
        if config is not None:
            self.config.update(config)
        self.batch_size = int(self.config.get("batch_size", 16))
        self.epochs = int(self.config.get("epochs", 1))
        self.steps_forward = int(self.config.get("steps_forward", 50))
        self.mc_p = float(self.config.get("mc_prob", 0.5))
        self.rng = rng if rng is not None else np.random.default_rng(self.config.get("seed", 0))
        # Ensure reproducibility for stdlib rand when needed
        self._seed = int(self.config.get("seed", 0))

    def _to_batch(self, x: Any) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def generate_batch(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of (theta, x0) samples and corresponding diffusion times.

        Returns a tuple (theta_batch, x0_batch, t_batch, x_t_batch).
        - theta_batch: shape (B, dim_theta)
        - x0_batch: shape (B, dim_x)
        - t_batch: shape (B,)
        - x_t_batch: shape (B, dim_x)
        """
        B = batch_size if batch_size is not None else self.batch_size
        # Sample theta in batch
        theta_batch = self.data_sampler.sample_prior(n=B)
        theta_batch = np.asarray(theta_batch)
        if theta_batch.ndim == 1:
            theta_batch = theta_batch.reshape(1, -1)
        # Generate x0 given theta; try batched simulate API first
        x0_list = []
        try:
            sim_out = self.data_sampler.simulate(theta_batch)
            sim_out = np.asarray(sim_out)
            if sim_out.ndim == 2 and sim_out.shape[0] == B:
                x0_batch = sim_out
            elif sim_out.ndim == 1:
                x0_batch = sim_out.reshape(1, -1)
            else:
                # Fallback to per-sample simulation
                raise ValueError("Unexpected simulate output shape for batched input")
        except Exception:
            for i in range(B):
                xi = self.data_sampler.simulate(theta_batch[i])
                xi = np.asarray(xi).reshape(-1)
                x0_list.append(xi)
            x0_batch = np.vstack(x0_list)
        # Time samples
        t_batch = self.rng.uniform(low=0.0, high=1.0, size=(B,))
        # Forward diffusion to obtain x_t
        x_t_list = []
        for i in range(B):
            x0_i = x0_batch[i]
            t_i = float(t_batch[i])
            # Forward sample; ensure 1D input to sde
            x_t_i = self.sde_forward.sample_forward(x0_i, t_i, steps=self.steps_forward, rng=self.rng)
            x_t_list.append(np.asarray(x_t_i).reshape(-1))
        x_t_batch = np.vstack(x_t_list)
        return theta_batch, x0_batch, t_batch, x_t_batch

    def train_step(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
        theta_batch, x0_batch, t_batch, x_t_batch = batch
        B, dim_x = x_t_batch.shape
        # Compute analytical score gradient for Gaussian p_t(x_t|x0)
        grads = []
        scores = []
        losses = []
        for i in range(B):
            x_t = x_t_batch[i]
            t = float(t_batch[i])
            x0 = x0_batch[i]
            mean, var = self.sde_forward.marginal_mean_and_var(x0, t)
            # gradient of log p_t(x_t | x0)
            grad_log_p = -(x_t - mean) / (var + 1e-12)
            # score network output
            s_phi = self.model(x_t, t)
            scores.append(s_phi)
            # simple loss: L = || s_phi - grad_log_p ||^2 elementwise averaged over dim_x
            diff = s_phi - grad_log_p
            # MC masking: randomly zero out a subset of dimensions per sample
            mask = self.rng.binomial(1, float(self.mc_p))
            # If mask is scalar due to numpy broadcasting, expand to dim_x
            if np.ndim(mask) == 0:
                mask_vec = np.full(dim_x, mask, dtype=float)
            else:
                mask_vec = mask.astype(float)
            loss_i = np.mean(((diff) * (1.0 - mask_vec)) ** 2)
            losses.append(loss_i)
        loss = float(np.mean(losses)) if losses else 0.0
        return loss

    def train(self, num_epochs: int = 1, batch_size: Optional[int] = None) -> float:
        """Run a lightweight training loop for a given number of epochs.

        Returns the final loss value after the last epoch.
        """
        final_loss = 0.0
        for _ in range(max(1, int(num_epochs))):
            batch = self.generate_batch(batch_size=batch_size)
            loss = self.train_step(batch)
            final_loss = loss
        return final_loss

    def generate_and_log(self, batch_size: Optional[int] = None) -> float:
        batch = self.generate_batch(batch_size=batch_size)
        loss = self.train_step(batch)
        return float(loss)

    def save_checkpoint(self, path: str) -> None:
        path_p = Path(path)
        path_p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "config": self.config,
        }
        with open(path_p, "wb") as f:
            pickle.dump(payload, f)

    def load_checkpoint(self, path: str) -> None:
        path_p = Path(path)
        if not path_p.exists():
            raise FileNotFoundError(str(path))
        with open(path_p, "rb") as f:
            payload = pickle.load(f)
        # Restore
        self.model = payload.get("model", self.model)
        self.config = payload.get("config", self.config)
        self.batch_size = int(self.config.get("batch_size", self.batch_size))


__all__ = ["TrainingLoop"]
