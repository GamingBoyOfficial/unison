"""Compare the rolling-swarm model with Fig. 5 of Ahmed et al. (arXiv 2005.12703).

A single swarm rolls upstream along the wall of a straight capillary (half-width 1 =
about 75 micrometres) against a steady flow. We sweep the flow and record the upstream
speed, then check three criteria that were written down BEFORE this script was run
(see docs/GROUNDING.md):

  1. Upstream speed falls with flow, roughly linearly.
  2. The flow at which upstream motion stops is about 5-12 times the still-water speed;
     accept 2.5-22.
  3. At 0.2 mm/s of flow, a 60 micrometre swarm is 2-6 times faster than a 15 micrometre
     one (the paper's data show roughly 4-5).

What is calibrated and what is a test is reported honestly at the end.

Units: length 1 = about 75 micrometres, speed 1 = about 100 micrometres/s
(so a flow of 2 units = 0.2 mm/s).

Run:  python experiments/validate_rolling.py
"""

import numpy as np

from unison.bots import Bots
from unison.geometry import straight_vessel
from unison.rolling import RollingParams, RollingSimulator

PARAMS = RollingParams()
V_SAT, A_HALF = 1.6, 0.04
UM = 75.0                       # micrometres per length unit


def upstream_speed(radius_um: float, flow: float, steps: int = 160) -> float:
    """Steady upstream speed (units of 100 um/s) of one swarm against a centre-line flow."""
    a = radius_um / UM
    arena = straight_vessel(length=400.0, radius=1.0, flow_speed=max(flow, 1e-9))
    bots = Bots(pos=np.array([[300.0, 0.3]]), speed_factor=np.ones(1), radius=np.array([a]),
                v0=np.array([V_SAT * a / (a + A_HALF)]))
    sim = RollingSimulator(arena, bots, np.random.default_rng(0), noise=0.0, params=PARAMS)
    xs = []
    for _ in range(steps):
        sim.step(np.array([1.0, 0.0]))
        xs.append(sim.pos[0, 0])
    lo, hi = 60, steps - 1
    return (xs[hi] - xs[lo]) / ((hi - lo) * sim.dt)      # positive = upstream (toward the leaf)


flows = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20.0])   # units; 10 = 1 mm/s
sizes = {"15 um": 15.0, "40 um": 40.0, "60 um": 60.0}
curves = {k: np.array([upstream_speed(r, f) for f in flows]) for k, r in sizes.items()}

print("Upstream speed (micrometres/s) against centre-line flow (mm/s):\n")
print(f"{'flow mm/s':>10}" + "".join(f"{k:>10}" for k in sizes))
for j, f in enumerate(flows):
    print(f"{f / 10:>10.2f}" + "".join(f"{curves[k][j] * 100:>10.0f}" for k in sizes))

print("\nPer-size summary:")
summary = {}
for k, v in curves.items():
    v0 = v[0]
    pos = np.where(v > 0)[0]
    last = pos[-1] if len(pos) else 0
    if last + 1 < len(flows):                           # interpolate the zero crossing
        f0, f1, v_a, v_b = flows[last], flows[last + 1], v[last], v[last + 1]
        stop = f0 + (f1 - f0) * v_a / (v_a - v_b)
    else:
        stop = float("inf")
    seg = slice(0, last + 1)
    slope, icpt = np.polyfit(flows[seg], v[seg], 1)
    resid = v[seg] - (slope * flows[seg] + icpt)
    r2 = 1.0 - resid.var() / v[seg].var() if v[seg].var() > 0 else 1.0
    mono = bool(np.all(np.diff(v[seg]) <= 1e-6))
    summary[k] = dict(v0=v0, stop=stop, slope=slope, r2=r2, mono=mono)
    print(f"  {k}: still-water {v0 * 100:.0f} um/s, slope {slope:+.3f}, linear fit R^2 {r2:.3f}, "
          f"stops at {stop / 10:.2f} mm/s (ratio to still-water speed {stop / v0:.1f})")

c1 = all(s["mono"] and s["r2"] >= 0.9 for s in summary.values())
ratios = [s["stop"] / s["v0"] for s in summary.values()]
c2 = all(2.5 <= r <= 22 for r in ratios)
j2 = int(np.argmin(np.abs(flows - 2.0)))
r3 = curves["60 um"][j2] / curves["15 um"][j2]
c3 = 2.0 <= r3 <= 6.0

print("\nCriteria (fixed before running):")
print(f"  1. Falls with flow, roughly linear (R^2 >= 0.9, monotone): {'PASS' if c1 else 'FAIL'}")
print(f"  2. Stopping flow / still-water speed within 2.5-22 "
      f"(got {min(ratios):.1f}-{max(ratios):.1f}): {'PASS' if c2 else 'FAIL'}")
print(f"  3. Speed(60 um) / speed(15 um) at 0.2 mm/s within 2-6 (got {r3:.2f}): {'PASS' if c3 else 'FAIL'}")
print("\nWhat these mean:")
print("  The slope (kappa_plot) was set from the paper's Fig. 5b fits, so criterion 1 and the")
print("  size of the ratio in criterion 2 follow largely BY CONSTRUCTION. Criterion 3 depends on")
print("  how rolling speed varies with swarm size and is the more independent check.")
