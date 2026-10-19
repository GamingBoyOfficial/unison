"""Where does the lost score go? Splits the loss into two causes.

  score = delivered - imbalance
  loss  = 1 - score = (bots never delivered) + (unevenness between the two arms)

If almost all the loss is 'never delivered', the time budget is the limit (a real
ceiling). If it is mostly 'unevenness', a better controller has room to improve.

Run:
  python experiments/diagnose_loss.py                              # the trained AI
  python experiments/diagnose_loss.py --model push                 # the straight push
  python experiments/diagnose_loss.py --scenario split_long
  python experiments/diagnose_loss.py --scenario bifurcation --model push
"""

import argparse

import numpy as np

from unison.bots import make_bots
from unison.controllers import PushTowardTarget
from unison.evaluate import build_obs
from unison.imitation import LearnedController
from unison.policy import NumpyPolicy
from unison.scenarios import SCENARIOS
from unison.simulator import Simulator

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="results/es_split.npz", help="a .npz/.joblib model, or 'push'")
ap.add_argument("--scenario", default="split")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--seed", type=int, default=20000)
args = ap.parse_args()

if args.model == "push":
    ctrl = PushTowardTarget()
elif args.model.endswith(".npz"):
    ctrl = NumpyPolicy.load(args.model)
else:
    import joblib
    ctrl = LearnedController(joblib.load(args.model))

scenario = SCENARIOS[args.scenario]()
rows, sp_ok, sp_bad = [], [], []
for seed in range(args.seed, args.seed + args.episodes):
    rng = np.random.default_rng(seed)
    arena = scenario.make_arena()
    task = scenario.make_task(arena)
    sim = Simulator(arena, make_bots(rng, start_center=scenario.start_center), rng)
    for i in range(scenario.steps):
        sim.step(ctrl.act(build_obs(task, sim, i / scenario.steps), rng))
    pos, speed = sim.pos, sim.bots.speed_factor
    at_end = task.delivered_mask(pos)
    up, low = task.fractions(pos)
    in_main = task.in_main_mask(pos)
    rows.append([task.score(pos), 1 - (up + low), abs(up - low),
                 float(np.mean(in_main)), float(np.mean(~in_main & ~at_end))])
    if at_end.any():
        sp_ok.append(speed[at_end].mean())
    if (~at_end).any():
        sp_bad.append(speed[~at_end].mean())

r = np.array(rows)
m = r.mean(axis=0)
print(f"Model: {args.model}   scenario: {args.scenario} ({scenario.steps} steps)   {args.episodes} swarms\n")
print(f"Mean score                                  {m[0]:.3f}")
print(f"Mean loss (1 - score)                       {1 - m[0]:.3f}   made of:")
print(f"  bots never delivered                      {m[1]:.3f}   ({100 * m[1] / (1 - m[0]):.0f}% of the loss)")
print(f"      still in the main channel             {m[3]:.3f}")
print(f"      in an arm but short of the target     {m[4]:.3f}")
print(f"  unevenness between the two arms           {m[2]:.3f}   ({100 * m[2] / (1 - m[0]):.0f}% of the loss)")
if sp_ok and sp_bad:
    print(f"\nAverage bot speed factor: delivered {np.mean(sp_ok):.3f}, undelivered {np.mean(sp_bad):.3f}")
    print("(a clear gap means the slow bots are the ones that miss out)")
