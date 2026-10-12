"""Controllers: whoever decides the single field direction at each moment.

Random and PushTowardTarget are the baselines any learned controller must beat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the field vector (length 2) for this step."""


class RandomController(Controller):
    """Picks a random direction every step."""

    def act(self, obs, rng):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)])


class PushTowardTarget(Controller):
    """Always pushes straight toward the target (+x). The hand-tuned baseline."""

    def act(self, obs, rng):
        return np.array([1.0, 0.0])
