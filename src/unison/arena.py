"""Arenas: the shape the bots live in, and the fluid flowing through it.

Every arena answers a few questions for the simulator:
  contain(pos)        -> where do bots end up after hitting a wall?
  flow_at(pos, t)     -> how fast does the fluid push a bot here, right now?
  drag_factor_at(pos) -> how much of its own push does a bot keep here?
                         (1.0 = all of it; lower near sticky walls)
  pulse_phase(t)      -> (sin, cos) of the heartbeat phase, so a controller can see it
  shapes()            -> rectangles to draw

New shapes (a vessel loaded from a file) only need to implement these methods,
so the rest of the project never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Arena(ABC):
    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax), used for drawing and for starting positions."""

    @abstractmethod
    def contain(self, pos: np.ndarray) -> np.ndarray:
        """Return positions pushed back inside the walls. pos has shape (n, 2)."""

    @abstractmethod
    def flow_at(self, pos: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Fluid velocity at each position at time t. Shape (n, 2)."""

    def drag_factor_at(self, pos: np.ndarray) -> np.ndarray:
        """Fraction of its own push each bot keeps (1.0 = no wall drag)."""
        return np.ones(len(pos))

    def pulse_phase(self, t: float) -> tuple[float, float]:
        """(sin, cos) of the heartbeat phase. (0, 0) when the flow does not pulse."""
        return (0.0, 0.0)

    def shapes(self) -> list[tuple[float, float, float, float]]:
        xmin, xmax, ymin, ymax = self.bounds
        return [(xmin, xmax, ymin, ymax)]


def inside_mask(arena: Arena, extent, res: int = 400) -> np.ndarray:
    """Boolean image of where the arena is (True = fluid). Works for every arena,
    because a point is inside exactly when contain() leaves it unchanged."""
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, max(2, int(res * (y1 - y0) / (x1 - x0))))
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.abs(arena.contain(pts) - pts).max(axis=1) < 1e-9
    return inside.reshape(gx.shape)


def _pulse(amp: float, period: float, t: float) -> float:
    return 1.0 + amp * np.sin(2.0 * np.pi * t / period) if amp else 1.0


def _phase(amp: float, period: float, t: float) -> tuple[float, float]:
    if not amp:
        return (0.0, 0.0)
    a = 2.0 * np.pi * t / period
    return (float(np.sin(a)), float(np.cos(a)))


class BoxArena(Arena):
    """Square box with still fluid. This is the Phase 1 arena."""

    def __init__(self, half_size: float = 5.0):
        self.half_size = half_size

    @property
    def bounds(self):
        h = self.half_size
        return (-h, h, -h, h)

    def contain(self, pos):
        return np.clip(pos, -self.half_size, self.half_size)

    def flow_at(self, pos, t=0.0):
        return np.zeros_like(pos)


class TubeArena(Arena):
    """Straight tube with parabolic (Poiseuille) flow.

    Flow is fastest in the middle and zero at the walls. flow_direction = -1
    pushes bots toward the left end, so a bot must swim AGAINST the flow to reach
    a target at the right end.

    pulse_amp / pulse_period: heartbeat-like pulsing of the flow (0 = steady).
    wall_drag: 0 = walls are free; 0.6 = a bot touching the wall keeps only 40%
    of its push.
    """

    def __init__(self, length: float = 10.0, half_width: float = 1.0,
                 flow_speed: float = 0.6, flow_direction: int = -1,
                 pulse_amp: float = 0.0, pulse_period: float = 4.0,
                 wall_drag: float = 0.0):
        self.length = length
        self.half_width = half_width
        self.flow_speed = flow_speed
        self.flow_direction = flow_direction
        self.pulse_amp = pulse_amp
        self.pulse_period = pulse_period
        self.wall_drag = wall_drag

    @property
    def bounds(self):
        return (0.0, self.length, -self.half_width, self.half_width)

    def contain(self, pos):
        out = pos.copy()
        out[:, 0] = np.clip(out[:, 0], 0.0, self.length)
        out[:, 1] = np.clip(out[:, 1], -self.half_width, self.half_width)
        return out

    def flow_at(self, pos, t=0.0):
        r = pos[:, 1] / self.half_width
        u = (self.flow_direction * self.flow_speed * (1.0 - r**2)
             * _pulse(self.pulse_amp, self.pulse_period, t))
        flow = np.zeros_like(pos)
        flow[:, 0] = u
        return flow

    def drag_factor_at(self, pos):
        r = np.clip(np.abs(pos[:, 1]) / self.half_width, 0.0, 1.0)
        return 1.0 - self.wall_drag * r**4

    def pulse_phase(self, t):
        return _phase(self.pulse_amp, self.pulse_period, t)


@dataclass(frozen=True)
class Rect:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    flow_scale: float = 1.0   # this channel's flow relative to flow_speed


class RectUnionArena(Arena):
    """An arena made of overlapping rectangles (channels).

    A bot is inside if it is inside ANY rectangle. A bot outside every rectangle
    is moved to the nearest point of the nearest rectangle. Each channel has its
    own parabolic flow profile across its width.
    """

    def __init__(self, rects: list[Rect], flow_speed: float = 0.6,
                 flow_direction: int = -1, pulse_amp: float = 0.0,
                 pulse_period: float = 4.0, wall_drag: float = 0.0):
        self.rects = list(rects)
        self.flow_speed = flow_speed
        self.flow_direction = flow_direction
        self.pulse_amp = pulse_amp
        self.pulse_period = pulse_period
        self.wall_drag = wall_drag
        self._lo = np.array([[r.xmin, r.ymin] for r in self.rects])   # (R, 2)
        self._hi = np.array([[r.xmax, r.ymax] for r in self.rects])   # (R, 2)

    @property
    def bounds(self):
        return (float(self._lo[:, 0].min()), float(self._hi[:, 0].max()),
                float(self._lo[:, 1].min()), float(self._hi[:, 1].max()))

    def shapes(self):
        return [(r.xmin, r.xmax, r.ymin, r.ymax) for r in self.rects]

    def _nearest(self, pos):
        """For each bot: the clipped position and the index of the nearest rectangle."""
        cands = np.clip(pos[None, :, :], self._lo[:, None, :], self._hi[:, None, :])
        d2 = ((cands - pos[None, :, :]) ** 2).sum(axis=2)       # (R, n)
        idx = d2.argmin(axis=0)
        return cands[idx, np.arange(len(pos))], idx

    def contain(self, pos):
        return self._nearest(pos)[0]

    def _local_r(self, pos, idx):
        yc = (self._lo[idx, 1] + self._hi[idx, 1]) / 2.0
        w = (self._hi[idx, 1] - self._lo[idx, 1]) / 2.0
        return np.clip((pos[:, 1] - yc) / w, -1.0, 1.0)

    def flow_at(self, pos, t=0.0):
        _, idx = self._nearest(pos)
        r = self._local_r(pos, idx)
        scale = np.array([self.rects[i].flow_scale for i in range(len(self.rects))])[idx]
        u = (self.flow_direction * self.flow_speed * scale * (1.0 - r**2)
             * _pulse(self.pulse_amp, self.pulse_period, t))
        flow = np.zeros_like(pos)
        flow[:, 0] = u
        return flow

    def drag_factor_at(self, pos):
        _, idx = self._nearest(pos)
        r = np.abs(self._local_r(pos, idx))
        return 1.0 - self.wall_drag * r**4

    def pulse_phase(self, t):
        return _phase(self.pulse_amp, self.pulse_period, t)

    def tube_layout(self):
        """Channels in the plain tube form the 3D viewer reads."""
        out = []
        for k, r in enumerate(self.rects):
            yc, w = (r.ymin + r.ymax) / 2.0, (r.ymax - r.ymin) / 2.0
            parent = next((j for j, o in enumerate(self.rects)
                           if j != k and abs(o.xmax - r.xmin) < 1e-9), -1)
            u = self.flow_speed * r.flow_scale
            out.append(dict(a=[r.xmin, yc], b=[r.xmax, yc], ra=w, rb=w,
                            parent=parent, umaxA=u, umaxB=u))
        return out


def y_branch(flow_speed: float = 0.6, pulse_amp: float = 0.3,
             pulse_period: float = 4.0, wall_drag: float = 0.6) -> RectUnionArena:
    """A main channel that splits into an upper and a lower arm.

        main: x 0..6, y -1..1      upper arm: x 6..12, y 0.5..2.5
                                   lower arm: x 6..12, y -2.5..-0.5

    Flow runs from the arms back into the main channel (toward -x), so bots must
    swim upstream. Each arm carries half the flow of the main channel.
    """
    return RectUnionArena(
        [Rect(0.0, 6.0, -1.0, 1.0, 1.0),
         Rect(6.0, 12.0, 0.5, 2.5, 0.5),
         Rect(6.0, 12.0, -2.5, -0.5, 0.5)],
        flow_speed=flow_speed, pulse_amp=pulse_amp,
        pulse_period=pulse_period, wall_drag=wall_drag)
