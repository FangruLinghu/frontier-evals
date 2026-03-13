import numpy as np

try:
    from .embeddings import Embeddings
except Exception:
    # Fallback for environments without package import semantics during tests
    class Embeddings:
        def __init__(self, d_model=64, max_vars=1024, max_funcs=256, seed=0):
            self.d_model = d_model
            self.max_vars = max_vars
            self.max_funcs = max_funcs
            rng = np.random.default_rng(seed)
            self.id_embeddings = rng.normal(scale=0.02, size=(max_vars, d_model))
            self.mc_embeddings = rng.normal(scale=0.02, size=(2, d_model))
            self.func_embeddings = rng.normal(scale=0.02, size=(max_funcs, d_model))
            self.val_W = rng.normal(scale=0.02, size=(1, d_model))  # scalar to d_model
            self.val_b = rng.normal(scale=0.02, size=(d_model,))
            self.time_W = rng.normal(scale=0.02, size=(16, d_model))  # Fourier features to d_model
            self.time_b = rng.normal(scale=0.02, size=(d_model,))

        def _ensure_id(self, idx):
            if idx < self.id_embeddings.shape[0]:
                return
            # extend if needed
            extra = idx - self.id_embeddings.shape[0] + 1
            rng = np.random.default_rng(0)
            new_emb = rng.normal(scale=0.02, size=(extra, self.d_model))
            self.id_embeddings = np.vstack([self.id_embeddings, new_emb])

        def get_id_embedding(self, idx):
            self._ensure_id(idx)
            return self.id_embeddings[idx]

        def get_mc_embedding(self, mc):
            mc = int(mc)
            if mc not in (0, 1):
                mc = 0
            return self.mc_embeddings[mc]

        def get_val_embedding(self, val):
            # val is scalar
            val = float(val)
            v = np.array([[val]])  # shape (1,1)
            return v.dot(self.val_W) + self.val_b  # shape (1, d_model)

        def get_func_embedding(self, func_idx):
            func_idx = int(func_idx)
            if func_idx >= self.func_embeddings.shape[0]:
                # extend if needed
                extra = func_idx - self.func_embeddings.shape[0] + 1
                rng = np.random.default_rng(0)
                new_emb = rng.normal(scale=0.02, size=(extra, self.d_model))
                self.func_embeddings = np.vstack([self.func_embeddings, new_emb])
            return self.func_embeddings[func_idx]

        def get_time_embedding(self, t):
            # Fourier features of time t in [0, 1] (or general range)
            t = float(t)
            freqs = np.arange(1, 9)  # 8 frequencies
            feats = []
            for f in freqs:
                feats.append(np.sin(2 * np.pi * f * t))
                feats.append(np.cos(2 * np.pi * f * t))
            feats = np.asarray(feats)  # shape (16,)
            return feats.dot(self.time_W) + self.time_b  # shape (d_model,)

        def zero_embedding(self):
            return np.zeros(self.d_model)


class JointTokenizer:
    """Joint tokenization for θ and x with MC flags.

    Each variable (θ_i or x_j) is represented as a token consisting of:
      - id embedding for the variable index
      - value embedding for the scalar value (unless function-valued, in which case function index embedding is used)
      - MC embedding indicating conditioning state
      - optional time embedding (positional / temporal conditioning)

    The resulting token sequence is suitable for feeding into a Transformer-like encoder.
    """

    def __init__(self, d_model: int = 64, max_vars: int = 1024, max_funcs: int = 256, include_time: bool = True, seed: int = 0):
        self.d_model = int(d_model)
        self.include_time = bool(include_time)
        self._rng_seed = int(seed)
        np.random.seed(self._rng_seed)
        # Embedding providers (modular)
        self.embeddings = Embeddings(d_model=self.d_model, max_vars=max_vars, max_funcs=max_funcs, seed=self._rng_seed)

    def encode(self, theta, x, mc_flags=None, is_function_flags=None, func_indices=None, times=None):
        """Encode variables into a sequence of token embeddings.

        Args:
            theta: list or array-like of scalar values for theta variables, length n_theta
            x: list or array-like of scalar values for x variables, length n_x
            mc_flags: list/array-like of 0/1 flags for conditioning for each variable (total length n_theta+n_x)
            is_function_flags: list/array-like of booleans indicating function-valued inputs (length n_theta+n_x)
            func_indices: list/array-like with function index for function-valued inputs (length n_theta+n_x)
            times: scalar or list-like of time values for time embeddings (length n_theta+n_x) or single scalar

        Returns:
            tokens: numpy array of shape (n_tokens, d_model)
        """
        theta = [] if theta is None else list(theta)
        x = [] if x is None else list(x)
        n_theta = len(theta)
        n_x = len(x)
        n_tokens = n_theta + n_x

        if mc_flags is None:
            mc_flags = [0] * n_tokens
        else:
            mc_flags = list(mc_flags)
            if len(mc_flags) != n_tokens:
                raise ValueError("mc_flags length must equal number of variables (theta + x)")

        if is_function_flags is None:
            is_function_flags = [False] * n_tokens
        else:
            is_function_flags = list(is_function_flags)
            if len(is_function_flags) != n_tokens:
                raise ValueError("is_function_flags length must equal number of variables")

        if func_indices is None:
            func_indices = [0] * n_tokens
        else:
            func_indices = list(func_indices)
            if len(func_indices) != n_tokens:
                raise ValueError("func_indices length must equal number of variables")

        # Times handling
        times_provided = times is not None
        if times_provided:
            if np.isscalar(times):
                times = [times] * n_tokens
            else:
                times = list(times)
                if len(times) != n_tokens:
                    raise ValueError("times length must equal number of variables if provided as list")
        else:
            times = [None] * n_tokens

        tokens = []
        # Build each token embedding
        for idx in range(n_tokens):
            var_id = idx  # unique per variable across theta/x
            if idx < n_theta:
                val = theta[idx]
            else:
                val = x[idx - n_theta]

            mc = mc_flags[idx]
            is_func = bool(is_function_flags[idx])
            func_idx = int(func_indices[idx]) if is_func else None

            id_emb = self.embeddings.get_id_embedding(var_id)

            if is_func and func_idx is not None:
                val_emb = self.embeddings.get_func_embedding(func_idx)
            else:
                val_emb = self.embeddings.get_val_embedding(val)

            mc_emb = self.embeddings.get_mc_embedding(mc)

            if self.include_time:
                t = times[idx] if times[idx] is not None else 0.0
                t_emb = self.embeddings.get_time_embedding(t)
            else:
                t_emb = self.embeddings.zero_embedding()

            token = id_emb + val_emb + mc_emb + t_emb
            tokens.append(token)

        tokens = np.stack(tokens, axis=0) if len(tokens) > 0 else np.zeros((0, self.d_model))
        return tokens


__all__ = ["JointTokenizer"]
