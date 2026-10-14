"""Tasks: what counts as success."""

from __future__ import annotations

import numpy as np


class ReachTargetTask:
    """Get as many bots as possible past a target line.

    Score = fraction of bots with x >= target_x at the end of the episode.
    """

    def __init__(self, target_x: float, y_range: tuple[float, float] = (-1.0, 1.0)):
        self.target_x = target_x
        self.y_range = y_range

    def fraction_reached(self, pos: np.ndarray) -> float:
        return float(np.mean(pos[:, 0] >= self.target_x))

    def score(self, pos: np.ndarray) -> float:
        return self.fraction_reached(pos)

    def observe(self, pos: np.ndarray) -> np.ndarray:
        """A small, fixed-size summary of the swarm (what a controller sees)."""
        return np.array([
            pos[:, 0].mean(), pos[:, 1].mean(),
            pos[:, 0].std(), pos[:, 1].std(),
            self.fraction_reached(pos),
        ])

    def target_regions(self):
        return [(self.target_x, self.target_x + 50.0, *self.y_range)]

    def status(self, pos: np.ndarray) -> str:
        return f"Reached target: {self.score(pos) * 100:.0f}% of bots"


class SplitTask:
    """Deliver half the swarm to the upper target and half to the lower target.

    Score = 2 * min(fraction in upper target, fraction in lower target).
    1.0 means a perfect 50/50 split with every bot delivered; 0.0 means one of the
    two targets got nothing, so sending everything to one arm scores zero.
    """

    def __init__(self, target_x: float, y_top: float = 2.5, y_bottom: float = -2.5,
                 junction_x: float = 6.0):
        self.target_x = target_x
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.junction_x = junction_x

    def fractions(self, pos: np.ndarray) -> tuple[float, float]:
        at_end = pos[:, 0] >= self.target_x
        return (float(np.mean(at_end & (pos[:, 1] > 0))),
                float(np.mean(at_end & (pos[:, 1] < 0))))

    def score(self, pos: np.ndarray) -> float:
        up, low = self.fractions(pos)
        return 2.0 * min(up, low)

    def committed(self, pos: np.ndarray) -> tuple[float, float, float]:
        """Fractions of bots (in main channel, inside upper arm, inside lower arm).

        This is the "picture" a human sees: where the bots have already gone,
        long before they reach the targets.
        """
        past = pos[:, 0] >= self.junction_x
        return (float(np.mean(~past)),
                float(np.mean(past & (pos[:, 1] > 0))),
                float(np.mean(past & (pos[:, 1] < 0))))

    def observe(self, pos: np.ndarray) -> np.ndarray:
        up, low = self.fractions(pos)
        main, c_up, c_low = self.committed(pos)
        return np.array([pos[:, 0].mean(), pos[:, 1].mean(),
                         pos[:, 0].std(), pos[:, 1].std(), up, low,
                         main, c_up, c_low])

    def target_regions(self):
        return [(self.target_x, self.target_x + 50.0, 0.0, self.y_top),
                (self.target_x, self.target_x + 50.0, self.y_bottom, 0.0)]

    def status(self, pos: np.ndarray) -> str:
        up, low = self.fractions(pos)
        return (f"Upper: {up * 100:.0f}%   Lower: {low * 100:.0f}%   "
                f"Split score: {self.score(pos):.2f}")
