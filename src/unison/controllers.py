"""Controllers: whoever decides the single field direction at each moment.

Controllers see an observation vector. Its LAST THREE entries are always
[episode progress 0..1, sin(heartbeat phase), cos(heartbeat phase)].

Random, PushTowardTarget, WallHugging and SwitchController are the baselines any
learned controller must beat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

PROGRESS = -3   # index of "how far through the episode are we" in the observation


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
    """Always pushes straight toward +x."""

    def act(self, obs, rng):
        return np.array([1.0, 0.0])


class WallHugging(Controller):
    """Pushes diagonally up-right, so bots slide up onto the top wall."""

    def act(self, obs, rng):
        return np.array([0.7, 0.7])


class SwitchController(Controller):
    """Pushes up-right until a chosen moment, then down-right.

    switch_at is a fraction of the episode (0..1). Bots that are ahead at the
    switch are already inside the upper arm and stay there; bots still lagging in
    the main channel turn toward the lower arm. This sorts the swarm by speed
    using nothing but timing.
    """

    def __init__(self, switch_at: float = 0.5):
        self.switch_at = switch_at

    def act(self, obs, rng):
        up = obs[PROGRESS] < self.switch_at
        return np.array([0.7, 0.7 if up else -0.7])
