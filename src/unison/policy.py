"""A small neural-network policy in plain NumPy.

Pulled out of the scikit-learn model so its weights can be mutated directly
(evolution strategies) and so it runs fast, with no heavy libraries.
"""

from __future__ import annotations

import numpy as np

from .controllers import Controller


class NumpyPolicy(Controller):
    """observation -> (standardize) -> ReLU layers -> field vector (length <= 1)."""

    def __init__(self, mean, scale, shapes, params):
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.shapes = [tuple(int(v) for v in s) for s in shapes]   # (n_in, n_out) per layer
        self.set_params(params)

    @property
    def n_params(self) -> int:
        return sum(i * o + o for i, o in self.shapes)

    def set_params(self, params) -> None:
        params = np.asarray(params, dtype=float).copy()
        if params.size != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {params.size}")
        self.params = params
        self._layers = []
        k = 0
        for n_in, n_out in self.shapes:
            w = params[k:k + n_in * n_out].reshape(n_in, n_out)
            k += n_in * n_out
            b = params[k:k + n_out]
            k += n_out
            self._layers.append((w, b))

    def with_params(self, params) -> "NumpyPolicy":
        return NumpyPolicy(self.mean, self.scale, self.shapes, params)

    @classmethod
    def from_sklearn(cls, pipeline) -> "NumpyPolicy":
        """Build from the StandardScaler + MLPRegressor pipeline that train_imitation saves."""
        scaler = pipeline.named_steps["standardscaler"]
        mlp = pipeline.named_steps["mlpregressor"]
        shapes = [w.shape for w in mlp.coefs_]
        flat = np.concatenate([np.concatenate([w.ravel(), b])
                               for w, b in zip(mlp.coefs_, mlp.intercepts_)])
        return cls(scaler.mean_, scaler.scale_, shapes, flat)

    def act(self, obs, rng):
        x = (np.asarray(obs, dtype=float) - self.mean) / self.scale
        for w, b in self._layers[:-1]:
            x = np.maximum(x @ w + b, 0.0)
        w, b = self._layers[-1]
        f = x @ w + b
        n = np.linalg.norm(f)
        return f / n if n > 1.0 else f

    def save(self, path) -> None:
        np.savez(path, mean=self.mean, scale=self.scale,
                 shapes=np.array(self.shapes), params=self.params)

    @classmethod
    def load(cls, path) -> "NumpyPolicy":
        d = np.load(path)
        return cls(d["mean"], d["scale"], d["shapes"], d["params"])
