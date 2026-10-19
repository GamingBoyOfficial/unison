"""A general vessel engine: any branching tree of tapered tubes.

A vessel is described by centerlines and radii. Each straight piece (a Tube) is a
tapered capsule; the arena is the union of all tubes, so curves, tapering and
angled bifurcations with a sharp carina all come for free.

Flow follows the geometry: every tube carries a share of the root flow rate, shares
split at each branch by Murray's law (proportional to radius cubed), and the
velocity profile in each tube is parabolic and speeds up where the vessel narrows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arena import Arena, _phase, _pulse


@dataclass(frozen=True)
class Tube:
    a: tuple[float, float]   # end nearer the root
    b: tuple[float, float]   # end farther from the root
    ra: float                # radius at a
    rb: float                # radius at b
    parent: int              # index of the tube this one continues from (-1 = root)
    flow: float              # share of the root flow rate carried (0..1)
    branch: int              # 0 = root branch, 1, 2, ... = daughters


@dataclass(frozen=True)
class Branch:
    """A smooth vessel segment: a polyline of centerline points with a radius at each."""
    points: list
    radii: list
    parent: int | None = None   # index of the branch it leaves (None = root)


def tubes_from_branches(branches: list[Branch]) -> list[Tube]:
    """Turn branches into tubes, computing parent links and Murray's-law flow shares."""
    n_br = len(branches)
    children = {i: [j for j, b in enumerate(branches) if b.parent == i] for i in range(n_br)}
    share = {0: 1.0}
    stack = [0]
    while stack:                                        # Murray: flow splits as radius^3
        i = stack.pop()
        kids = children[i]
        if kids:
            w = np.array([branches[k].radii[0] ** 3 for k in kids])
            for k, f in zip(kids, w / w.sum()):
                share[k] = share[i] * float(f)
                stack.append(k)
    tubes, last_of = [], {}
    order = [0]
    for i in order:                                     # parents before children
        order += children[i]
    for i in order:
        br = branches[i]
        prev = -1 if br.parent is None else last_of[br.parent]
        for k in range(len(br.points) - 1):
            tubes.append(Tube(tuple(br.points[k]), tuple(br.points[k + 1]),
                              float(br.radii[k]), float(br.radii[k + 1]),
                              prev, share[i], i))
            prev = len(tubes) - 1
        last_of[i] = len(tubes) - 1
    return tubes


class VesselArena(Arena):
    """Union of tapered tubes, with flow that follows the vessel.

    flow_speed: centre-line speed in the root tube.
    flow_direction: -1 = toward the root (a bot starting at the root must swim
    upstream), +1 = away from the root.
    """

    def __init__(self, tubes: list[Tube], flow_speed: float = 0.6, flow_direction: int = -1,
                 pulse_amp: float = 0.0, pulse_period: float = 4.0, wall_drag: float = 0.0):
        self.tubes = list(tubes)
        self.flow_speed = flow_speed
        self.flow_direction = flow_direction
        self.pulse_amp = pulse_amp
        self.pulse_period = pulse_period
        self.wall_drag = wall_drag
        self._a = np.array([t.a for t in tubes], dtype=float)
        self._b = np.array([t.b for t in tubes], dtype=float)
        self._ra = np.array([t.ra for t in tubes], dtype=float)
        self._rb = np.array([t.rb for t in tubes], dtype=float)
        self._flow = np.array([t.flow for t in tubes], dtype=float)
        self._branch = np.array([t.branch for t in tubes], dtype=int)
        ab = self._b - self._a
        self._len = np.linalg.norm(ab, axis=1)
        self._len2 = self._len ** 2
        self._tan = ab / self._len[:, None]
        r0 = 0.5 * (self._ra[0] + self._rb[0])
        self._q0 = (flow_speed / 1.5) * 2.0 * r0        # root flow rate (2D channel)

    # ---- geometry -----------------------------------------------------------
    @property
    def bounds(self):
        lo = np.minimum(self._a - self._ra[:, None], self._b - self._rb[:, None]).min(axis=0)
        hi = np.maximum(self._a + self._ra[:, None], self._b + self._rb[:, None]).max(axis=0)
        return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))

    def _locate(self, pos):
        """For every tube and bot: axis parameter t, closest axis point, distance, radius."""
        p = pos[None, :, :]
        ap = p - self._a[:, None, :]
        t = np.clip((ap * (self._b - self._a)[:, None, :]).sum(axis=2) / self._len2[:, None], 0.0, 1.0)
        closest = self._a[:, None, :] + t[:, :, None] * (self._b - self._a)[:, None, :]
        d = np.linalg.norm(p - closest, axis=2)
        r = self._ra[:, None] + t * (self._rb - self._ra)[:, None]
        return t, closest, d, r

    def contain(self, pos):
        t, closest, d, r = self._locate(pos)
        sdf = d - r
        i = sdf.argmin(axis=0)
        n = np.arange(len(pos))
        out = pos.copy()
        outside = sdf[i, n] > 0.0
        if outside.any():
            c, dd, rr = closest[i, n][outside], d[i, n][outside], r[i, n][outside]
            out[outside] = c + (pos[outside] - c) * (rr / dd)[:, None]
        return out

    def _owner(self, pos):
        """The tube each bot is most central in, with its local geometry."""
        t, closest, d, r = self._locate(pos)
        s = d / r
        i = s.argmin(axis=0)
        n = np.arange(len(pos))
        return i, t[i, n], np.minimum(s[i, n], 1.0), r[i, n]

    # ---- physics ------------------------------------------------------------
    def flow_at(self, pos, t=0.0):
        i, _, s, r = self._owner(pos)
        umax = 1.5 * self._q0 * self._flow[i] / (2.0 * r)
        u = self.flow_direction * umax * (1.0 - s ** 2) * _pulse(self.pulse_amp, self.pulse_period, t)
        return u[:, None] * self._tan[i]

    def drag_factor_at(self, pos):
        _, _, s, _ = self._owner(pos)
        return 1.0 - self.wall_drag * s ** 4

    def pulse_phase(self, t):
        return _phase(self.pulse_amp, self.pulse_period, t)

    # ---- helpers for tasks and drawing ----------------------------------------
    def branch_of(self, pos):
        """Branch index (0 = root branch) of the tube each bot is most central in."""
        i, _, _, _ = self._owner(pos)
        return self._branch[i]

    def tube_layout(self):
        """Tubes in a plain form for the 3D viewer."""
        out = []
        for k, tb in enumerate(self.tubes):
            qa = 1.5 * self._q0 * tb.flow / (2.0 * tb.ra)
            qb = 1.5 * self._q0 * tb.flow / (2.0 * tb.rb)
            out.append(dict(a=list(tb.a), b=list(tb.b), ra=tb.ra, rb=tb.rb,
                            parent=tb.parent, umaxA=qa, umaxB=qb))
        return out


def bifurcation(angle_deg: float = 35.0, flow_speed: float = 0.6, pulse_amp: float = 0.3,
                pulse_period: float = 4.0, wall_drag: float = 0.6) -> VesselArena:
    """A parent vessel splitting into two curved, tapering daughters.

    Each daughter leaves at +/- angle_deg from the parent's axis (so the full
    bifurcation angle is twice that) and gently bends back toward the parent's
    direction, like a real branching artery. The inner walls meet at a sharp carina.
    Radii follow Murray's law at the split (daughter r = parent r / 2^(1/3)).
    """
    th = np.radians(angle_deg)
    r_p = 1.0
    r_d = r_p / 2 ** (1 / 3)
    carina = (6.0, 0.0)
    root = Branch([(0.0, 0.0), (3.0, 0.0), carina], [r_p, r_p, r_p * 0.98])
    ups, lows = [carina], [carina]
    bend = [th, th * 0.6, th * 0.25]                 # heading of each piece, easing toward 0
    seg = [2.6, 2.4, 2.2]
    for k in range(3):
        x, y = ups[-1]
        ups.append((x + seg[k] * np.cos(bend[k]), y + seg[k] * np.sin(bend[k])))
        x, y = lows[-1]
        lows.append((x + seg[k] * np.cos(bend[k]), y - seg[k] * np.sin(bend[k])))
    radii = [r_d, r_d * 0.95, r_d * 0.88, r_d * 0.8]
    upper = Branch([tuple(p) for p in ups], radii, parent=0)
    lower = Branch([tuple(p) for p in lows], radii, parent=0)
    return VesselArena(tubes_from_branches([root, upper, lower]), flow_speed=flow_speed,
                       pulse_amp=pulse_amp, pulse_period=pulse_period, wall_drag=wall_drag)
