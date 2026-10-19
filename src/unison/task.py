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

    def delivered_mask(self, pos: np.ndarray) -> np.ndarray:
        return pos[:, 0] >= self.target_x

    def in_main_mask(self, pos: np.ndarray) -> np.ndarray:
        return pos[:, 0] < self.junction_x

    def junction_point(self, arena=None):
        return (self.junction_x, 0.0)

    def target_rings(self, arena):
        """Rings marking where each target zone begins: (centre, radius, direction)."""
        rings = []
        for r in arena.rects:
            if r.xmax > self.target_x:
                rings.append(((self.target_x, (r.ymin + r.ymax) / 2), (r.ymax - r.ymin) / 2, (1.0, 0.0)))
        return rings

    def status(self, pos: np.ndarray) -> str:
        up, low = self.fractions(pos)
        return (f"Upper: {up * 100:.0f}%   Lower: {low * 100:.0f}%   "
                f"Split score: {self.score(pos):.2f}")


class BifurcationTask:
    """Split the swarm between the two daughter vessels of a VesselArena.

    A bot counts as delivered to a daughter when it is within `target_length` of that
    daughter's far end. Score = 2 * min(upper fraction, lower fraction), exactly as
    in SplitTask, so scores are comparable across scenarios.
    """

    def __init__(self, arena, target_length: float = 1.4):
        self.arena = arena
        self.target_length = target_length
        ends = {}
        for tb in arena.tubes:
            if tb.branch > 0:
                ends[tb.branch] = tb                      # last tube of each daughter wins
        self._ends = sorted(ends.values(), key=lambda tb: -tb.b[1])   # upper daughter first
        if len(self._ends) != 2:
            raise ValueError("BifurcationTask needs exactly two daughter branches")
        self._centres = np.array([tb.b for tb in self._ends])
        self._branches = [tb.branch for tb in self._ends]
        root_last = max(i for i, tb in enumerate(arena.tubes) if tb.branch == 0)
        self.junction = arena.tubes[root_last].b

    def _near(self, pos):
        d = [np.linalg.norm(pos - c, axis=1) <= self.target_length for c in self._centres]
        return d[0], d[1]

    def fractions(self, pos: np.ndarray) -> tuple[float, float]:
        up, low = self._near(pos)
        return float(np.mean(up)), float(np.mean(low))

    def score(self, pos: np.ndarray) -> float:
        up, low = self.fractions(pos)
        return 2.0 * min(up, low)

    def committed(self, pos: np.ndarray) -> tuple[float, float, float]:
        br = self.arena.branch_of(pos)
        return (float(np.mean(br == 0)), float(np.mean(br == self._branches[0])),
                float(np.mean(br == self._branches[1])))

    def observe(self, pos: np.ndarray) -> np.ndarray:
        up, low = self.fractions(pos)
        main, c_up, c_low = self.committed(pos)
        return np.array([pos[:, 0].mean(), pos[:, 1].mean(), pos[:, 0].std(), pos[:, 1].std(),
                         up, low, main, c_up, c_low])

    def delivered_mask(self, pos: np.ndarray) -> np.ndarray:
        up, low = self._near(pos)
        return up | low

    def in_main_mask(self, pos: np.ndarray) -> np.ndarray:
        return self.arena.branch_of(pos) == 0

    def junction_point(self, arena=None):
        return tuple(self.junction)

    def target_circles(self):
        return [(float(c[0]), float(c[1]), self.target_length) for c in self._centres]

    def target_rings(self, arena):
        rings = []
        for tb in self._ends:
            a, b = np.array(tb.a), np.array(tb.b)
            tan = (b - a) / np.linalg.norm(b - a)
            c = b - tan * self.target_length
            t = float(np.clip(np.dot(c - a, b - a) / np.dot(b - a, b - a), 0, 1))
            rings.append(((float(c[0]), float(c[1])), tb.ra + t * (tb.rb - tb.ra), (float(tan[0]), float(tan[1]))))
        return rings

    def status(self, pos: np.ndarray) -> str:
        up, low = self.fractions(pos)
        return (f"Upper: {up * 100:.0f}%   Lower: {low * 100:.0f}%   "
                f"Split score: {self.score(pos):.2f}")
