"""Train an AI to copy your play (behavior cloning), then score it fairly.

Run:
  python experiments/train_imitation.py                  # split task, all your episodes
  python experiments/train_imitation.py --min-score 0.8  # learn only from your good episodes

Needs:  python -m pip install scikit-learn
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unison.controllers import PushTowardTarget
from unison.evaluate import evaluate
from unison.imitation import LearnedController, load_demos
from unison.scenarios import SCENARIOS

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="split")
ap.add_argument("--min-score", type=float, default=0.0,
                help="only learn from episodes scoring at least this much")
ap.add_argument("--hold-out", type=float, default=0.25,
                help="fraction of EPISODES kept aside to check generalization")
args = ap.parse_args()

scenario = SCENARIOS[args.scenario]()
demos, skipped = load_demos(args.scenario, scenario)
for seed, why in skipped:
    print(f"  skipped episode {seed}: {why}")
print(f"Loaded {len(demos)} replayable episodes "
      f"(human mean score {np.mean([d.score for d in demos]):.3f}).")

demos = [d for d in demos if d.score >= args.min_score]
if len(demos) < 4:
    raise SystemExit("Need at least 4 usable episodes. Play more with experiments/play.py.")
human = np.array([d.score for d in demos])
print(f"Using {len(demos)} episodes with score >= {args.min_score} "
      f"(mean {human.mean():.3f}).\n")


def new_model():
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                      random_state=0))


def stack(ds):
    return np.vstack([d.obs for d in ds]), np.vstack([d.actions for d in ds])


# 1) Can the model predict what you would do in episodes it has never seen?
order = np.random.default_rng(0).permutation(len(demos))
n_val = max(1, int(round(len(demos) * args.hold_out)))
val = [demos[i] for i in order[:n_val]]
train = [demos[i] for i in order[n_val:]]
Xtr, ytr = stack(train)
Xva, yva = stack(val)
check = new_model().fit(Xtr, ytr)
print(f"Held-out check ({len(val)} unseen episodes): R^2 = {check.score(Xva, yva):.3f}")
print("  (1.0 = predicts your key presses perfectly, 0.0 = no better than guessing the average)\n")

# 2) Train on everything, then score the AI the same way as every other controller.
X, y = stack(demos)
model = new_model().fit(X, y)
ai = LearnedController(model)
print("Scoring on 30 fresh random swarms ...", flush=True)
r_ai = evaluate(ai, scenario, n_episodes=30)
r_push = evaluate(PushTowardTarget(), scenario, n_episodes=30)

print(f"\n{'Driver':<30}{'Mean score':>12}{'  95% CI':>14}")
print("-" * 56)
print(f"{'Push toward target':<30}{r_push.mean:>12.3f}{'  +/- ' + format(r_push.ci95, '.3f'):>14}")
print(f"{'Imitation AI (copies you)':<30}{r_ai.mean:>12.3f}{'  +/- ' + format(r_ai.ci95, '.3f'):>14}")
print(f"{'You (the demos it learned)':<30}{human.mean():>12.3f}")

out = Path("results")
out.mkdir(exist_ok=True)
joblib.dump(model, out / f"imitation_{args.scenario}.joblib")
print(f"\nSaved model to results/imitation_{args.scenario}.joblib")
