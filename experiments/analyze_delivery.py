"""Where does a controller's score go, and how much could any controller still gain?

For a controller (default: the straight push) on a scenario, this reports

  * the score, the fraction of bots delivered to either target, and the score a PERFECT
    50/50 split of the same delivered bots would give (so the most that balancing the
    arms could add);
  * delivery by swarm size, with each size's stalling flow (the flow beyond which it
    loses grip) next to the flow it actually meets.

Run:
  python experiments/analyze_delivery.py                       # straight push, rolling scenario
  python experiments/analyze_delivery.py --model results/es_rolling.npz
"""

import argparse

import numpy as np

from unison.controllers import PushTowardTarget
from unison.evaluate import build_obs
from unison.imitation import LearnedController
from unison.policy import NumpyPolicy
from unison.scenarios import SCENARIOS

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="rolling")
ap.add_argument("--model", default="push", help="'push', a .npz or a .joblib model")
ap.add_argument("--episodes", type=int, default=12)
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
scores, delivered, rows = [], [], []
for seed in range(args.seed, args.seed + args.episodes):
    rng = np.random.default_rng(seed)
    arena = scenario.make_arena()
    task = scenario.make_task(arena)
    sim = scenario.new_sim(rng, arena)
    for i in range(scenario.steps):
        sim.step(ctrl.act(build_obs(task, sim, i / scenario.steps), rng))
    up, low = task.fractions(sim.pos)
    scores.append(task.score(sim.pos))
    delivered.append(up + low)
    b = sim.bots
    if b.radius is not None:
        rows.append(np.column_stack([b.radius, task.delivered_mask(sim.pos), b.v0 / sim.kappa]))

print(f"{args.model} on '{args.scenario}' ({scenario.steps} steps), {args.episodes} swarms\n")
print(f"  mean score                           {np.mean(scores):.3f}")
print(f"  mean delivered (either target)       {np.mean(delivered):.3f}")
gain = np.mean(delivered) - np.mean(scores)
print(f"  score if the split were PERFECT      {np.mean(delivered):.3f}   "
      f"(balancing could add at most {gain:+.3f})")

if rows:
    R = np.vstack(rows)
    flow = arena.flow_at(np.array([[3.0, 0.55]]), 1.0)
    print(f"\nDelivery by swarm size (flow in the parent's wall lane at the systolic peak: "
          f"about {np.linalg.norm(flow):.1f})\n")
    print(f"{'swarm radius (um)':<20}{'bots':>6}{'delivered':>11}{'avg stalling flow':>19}")
    for lo, hi in [(0, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0)]:
        m = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if m.any():
            print(f"{lo * 75:3.0f} to {min(hi, 0.9) * 75:3.0f}{'':<10}{m.sum():>6}{R[m, 1].mean():>11.2f}"
                  f"{R[m, 2].mean():>19.2f}")
    print("\nA swarm whose stalling flow is below the flow it meets cannot hold the wall at every")
    print("heartbeat peak; no controller can change that, so it caps what learning can add.")
