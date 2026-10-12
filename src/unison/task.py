"""Tasks: what counts as success."""

from __future__ import annotations

import numpy as np


class ReachTargetTask:
    """Get as many bots as possible past a target line.

    Score = fraction of bots with x >= target_x at the end of the episode.
    """

    def __init__(self, target_x: float):
        self.target_x = target_x

    def fraction_reached(self, pos: np.ndarray) -> float:
        return float(np.mean(pos[:, 0] >= self.target_x))

    def observe(self, pos: np.ndarray) -> np.ndarray:
        """A small, fixed-size summary of the swarm (what a controller sees)."""
        return np.array([
            pos[:, 0].mean(), pos[:, 1].mean(),
            pos[:, 0].std(), pos[:, 1].std(),
            self.fraction_reached(pos),
        ])
