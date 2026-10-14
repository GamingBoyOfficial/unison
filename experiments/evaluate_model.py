"""Stress-test a trained model on many fresh swarms, with a paired comparison.

Run:
  python experiments/evaluate_model.py
  python experiments/evaluate_model.py --model results/es_split.npz
  python experiments/evaluate_model.py --scenario split_long   # a different time budget
  python experiments/evaluate_model.py --episodes 200 --seed 30000

The model and the straight-push baseline play the SAME swarms, so the difference
between them is measured swarm by swarm (a paired comparison), which is much
more sensitive than comparing two separate averages.
"""

import argparse

import numpy as np

from unison.controllers import PushTowardTarget
from unison.evaluate import evaluate
from unison.imitation import LearnedController
from unison.policy import NumpyPolicy
from unison.scenarios import SCENARIOS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="results/imitation_split.joblib")
ap.add_argument("--scenario", default="split")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--seed", type=int, default=20000,
                help="first swarm seed; keep it away from training seeds (5000s) "
                     "and earlier test seeds (1000s)")
args = ap.parse_args()

scenario = SCENARIOS[args.scenario]()
if args.model.endswith(".npz"):
    ai = NumpyPolicy.load(args.model)                    # trained by train_es.py
else:
    import joblib
    ai = LearnedController(joblib.load(args.model))      # trained by train_imitation.py

print(f"Scoring on {args.episodes} fresh swarms (seeds {args.seed}..{args.seed + args.episodes - 1}), "
      f"scenario '{args.scenario}' ({scenario.steps} steps) ...", flush=True)
r_ai = evaluate(ai, scenario, n_episodes=args.episodes, base_seed=args.seed)
r_push = evaluate(PushTowardTarget(), scenario, n_episodes=args.episodes, base_seed=args.seed)

diff = r_ai.scores - r_push.scores
ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
print(f"\n{'Driver':<24}{'Mean':>8}{'  95% CI':>12}{'  Worst':>9}{'  Best':>8}")
print("-" * 61)
for name, r in [("Push toward target", r_push), ("Trained model", r_ai)]:
    print(f"{name:<24}{r.mean:>8.3f}{'  +/- ' + format(r.ci95, '.3f'):>12}"
          f"{r.scores.min():>9.3f}{r.scores.max():>8.3f}")
print(f"\nPaired difference (model - push): {diff.mean():+.3f} +/- {ci:.3f}")
print(f"Model beat the push on {int((diff > 0).sum())} of {len(diff)} swarms.")
