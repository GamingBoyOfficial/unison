"""Record runs on one swarm so the 3D viewer (demo/index.html) can play them back.

Each controller plays the SAME swarm (same bots, same random jitter), so the viewer
can show them side by side.

Run:
  python experiments/export_replay.py
  python experiments/export_replay.py --seed 20003 --model results/es_split.npz
  python experiments/export_replay.py --scenario split_long --stride 8

Then open the viewer:
  cd demo
  python -m http.server 8000      # and visit http://localhost:8000
"""

import argparse
import json
from pathlib import Path

import numpy as np

from unison.arena import RectUnionArena
from unison.bots import make_bots
from unison.controllers import PushTowardTarget
from unison.evaluate import build_obs
from unison.policy import NumpyPolicy
from unison.scenarios import SCENARIOS
from unison.simulator import Simulator

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="split", choices=["split", "split_long"])
ap.add_argument("--model", default="results/es_split.npz", help="trained .npz policy")
ap.add_argument("--seed", type=int, default=20003)
ap.add_argument("--stride", type=int, default=4, help="record every Nth step")
ap.add_argument("--out", default="demo/replay.js")
args = ap.parse_args()

scenario = SCENARIOS[args.scenario]()
arena = scenario.make_arena()
if not isinstance(arena, RectUnionArena):
    raise SystemExit("The viewer currently supports the Y-branch scenarios only.")
task = scenario.make_task(arena)

drivers = [("Straight push", PushTowardTarget(), "#ff9f43")]
if Path(args.model).exists():
    drivers.append(("Trained AI", NumpyPolicy.load(args.model), "#35e0ff"))
else:
    print(f"Model {args.model} not found, exporting the baseline only.")


def q(a):
    """Positions as integers in hundredths: small files, plenty precise."""
    return np.round(np.asarray(a) * 100).astype(int).ravel().tolist()


bot_rng = np.random.default_rng(args.seed + 999)
runs, speed, hfrac = [], None, None
for name, ctrl, color in drivers:
    rng = np.random.default_rng(args.seed)
    bots = make_bots(rng, start_center=scenario.start_center)
    sim = Simulator(arena, bots, rng)
    if speed is None:
        speed = np.round(bots.speed_factor, 3).tolist()
        hfrac = np.round(bot_rng.uniform(-1, 1, bots.n), 3).tolist()
    frames, ups, lows, scores, field = [q(sim.pos)], [0.0], [0.0], [0.0], [[0.0, 0.0]]
    for i in range(scenario.steps):
        f = ctrl.act(build_obs(task, sim, i / scenario.steps), rng)
        sim.step(f)
        if (i + 1) % args.stride == 0:
            up, low = task.fractions(sim.pos)
            frames.append(q(sim.pos))
            ups.append(round(up, 4))
            lows.append(round(low, 4))
            scores.append(round(task.score(sim.pos), 4))
            field.append([round(float(f[0]), 3), round(float(f[1]), 3)])
    runs.append(dict(name=name, color=color, frames=frames, up=ups, low=lows,
                     score=scores, field=field))
    print(f"{name}: final score {scores[-1]:.3f}")

data = dict(
    scenario=args.scenario, seed=args.seed, steps=scenario.steps, dt=sim.dt,
    stride=args.stride, nBots=len(speed), speed=speed, hfrac=hfrac,
    arena=dict(
        rects=[dict(x0=r.xmin, x1=r.xmax, y0=r.ymin, y1=r.ymax, flow=r.flow_scale)
               for r in arena.rects],
        flowSpeed=arena.flow_speed, flowDir=arena.flow_direction,
        pulseAmp=arena.pulse_amp, pulsePeriod=arena.pulse_period),
    targetX=task.target_x, runs=runs)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("window.UNISON_REPLAY = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(runs[0]['frames'])} frames per run)")
