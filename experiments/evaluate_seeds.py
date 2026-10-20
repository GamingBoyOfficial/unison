"""How much does the result depend on the training run's luck?

Scores every practiced model for a scenario (results/es_<scenario>.npz is seed 0,
results/es_<scenario>_seed1.npz, _seed2.npz, ...) on the SAME fresh swarms, then
reports the spread across training runs.

Run:
  python experiments/evaluate_seeds.py --scenario bifurcation
  python experiments/evaluate_seeds.py --scenario bifurcation --episodes 100
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np

from unison.controllers import PushTowardTarget
from unison.evaluate import evaluate
from unison.policy import NumpyPolicy
from unison.scenarios import SCENARIOS
from unison.stats import mean_ci

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="bifurcation")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--seed", type=int, default=20000, help="first evaluation swarm seed")
args = ap.parse_args()

files = sorted(glob.glob(f"results/es_{args.scenario}.npz") + glob.glob(f"results/es_{args.scenario}_seed*.npz"),
               key=lambda f: int(re.search(r"_seed(\d+)", f).group(1)) if "_seed" in f else 0)
if len(files) < 2:
    raise SystemExit(f"Found {len(files)} trained model(s) for '{args.scenario}'. Train more with:\n"
                     f"  python experiments/train_es.py --scenario {args.scenario} --seed 1")

scenario = SCENARIOS[args.scenario]()
print(f"Scoring {len(files)} independently trained models on the same {args.episodes} swarms "
      f"(seeds {args.seed}..{args.seed + args.episodes - 1}) ...\n", flush=True)
push = evaluate(PushTowardTarget(), scenario, n_episodes=args.episodes, base_seed=args.seed)
means = []
print(f"{'Training run':<34}{'Mean':>8}{'  95% CI':>12}{'  Worst':>9}")
print("-" * 63)
for f in files:
    r = evaluate(NumpyPolicy.load(f), scenario, n_episodes=args.episodes, base_seed=args.seed)
    means.append(r.mean)
    print(f"{Path(f).name:<34}{r.mean:>8.3f}{'  +/- ' + format(r.ci95, '.3f'):>12}{r.scores.min():>9.3f}", flush=True)
print(f"{'Straight push (reference)':<34}{push.mean:>8.3f}{'  +/- ' + format(push.ci95, '.3f'):>12}{push.scores.min():>9.3f}")

m = np.array(means)
print(f"\nAcross {len(m)} training runs: mean {m.mean():.3f}, std {m.std(ddof=1):.3f}, "
      f"range {m.min():.3f} to {m.max():.3f}")
print(f"Every run beat the straight push by at least {m.min() - push.mean:+.3f}"
      if m.min() > push.mean else "At least one run did NOT beat the straight push.")
