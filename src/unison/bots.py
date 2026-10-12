"""Bots: every bot is just a few numbers. No two are identical."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bots:
    pos: np.ndarray            # (n, 2) positions
    speed_factor: np.ndarray   # (n,) how strongly each bot reacts to the field

    @property
    def n(self) -> int:
        return len(self.speed_factor)


def make_bots(rng: np.random.Generator, n: int = 300, spread: float = 0.15,
              start_center=(0.0, 0.0), start_radius: float = 0.3) -> Bots:
    """Create n bots with random size and magnetism.

    spread controls how different the bots are (0.01 = nearly identical).
    Bigger bots feel more drag; stronger magnetism means a stronger push.
    """
    size = rng.normal(1.0, spread, n).clip(0.5, 1.5)
    magnetism = rng.normal(1.0, spread, n).clip(0.5, 1.5)
    drag = 1.0 + 0.5 * (size - 1.0)
    mobility = 1.0 / drag
    speed_factor = mobility * magnetism
    pos = np.asarray(start_center) + rng.normal(0.0, start_radius, (n, 2))
    return Bots(pos=pos, speed_factor=speed_factor)
