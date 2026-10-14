"""Head-to-head: you vs the trained AI vs the straight push, on the SAME swarms.

First play the benchmark swarms yourself:
    python experiments/play.py split --benchmark      (about 30 episodes)
Then run:
    python experiments/compare_human_ai.py
    python experiments/compare_human_ai.py --model results/es_split.npz

Because everybody plays identical swarms, the comparison is made swarm by swarm
(a paired comparison), which is far more sensitive than comparing two averages.
"""

import argparse
from pathlib import Path

import numpy as np

from unison.controllers import PushTowardTarget
from unison.evaluate import run_episode
from unison.imitation import LearnedController
from unison.playlog import read_rows
from unison.policy import NumpyPolicy
from unison.scenarios import SCENARIOS
from unison.stats import mean_ci

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="split")
ap.add_argument("--model", default="results/es_split.npz")
args = ap.parse_args()

rows = [r for r in read_rows(Path("results/benchmark_scores.csv")) if r["scenario"] == args.scenario]
human = {int(r["seed"]): float(r["score"]) for r in rows}      # latest attempt per swarm
if len(human) < 3:
    raise SystemExit("Play some benchmark swarms first:\n"
                     f"  python experiments/play.py {args.scenario} --benchmark")

if args.model.endswith(".npz"):
    ai = NumpyPolicy.load(args.model)
else:
    import joblib
    ai = LearnedController(joblib.load(args.model))

scenario = SCENARIOS[args.scenario]()
seeds = sorted(human)
print(f"Scoring the AI and the straight push on the {len(seeds)} swarms you played ...", flush=True)
h = np.array([human[s] for s in seeds])
a = np.array([run_episode(ai, s, scenario) for s in seeds])
p = np.array([run_episode(PushTowardTarget(), s, scenario) for s in seeds])

print(f"\nSwarms: {len(seeds)} (seeds {seeds[0]}..{seeds[-1]})\n")
print(f"{'Driver':<22}{'Mean':>8}{'  95% CI':>12}")
print("-" * 42)
for name, v in [("Straight push", p), ("You", h), ("Trained AI", a)]:
    m, half = mean_ci(v)
    print(f"{name:<22}{m:>8.3f}{'  +/- ' + format(half, '.3f'):>12}")


def paired(label, x, y):
    d = x - y
    m, half = mean_ci(d)
    print(f"\n{label}: {m:+.3f} +/- {half:.3f}   (higher on {int((d > 0).sum())} swarms, "
          f"lower on {int((d < 0).sum())}, tied on {int((d == 0).sum())})")
    return m, half


paired("You minus straight push", h, p)
m, half = paired("AI minus you", a, h)
if m - half > 0:
    print("-> The AI scored higher than you; the interval excludes zero.")
elif m + half < 0:
    print("-> You scored higher than the AI; the interval excludes zero.")
else:
    print("-> No clear difference yet: the interval includes zero. "
          "Play more benchmark swarms to sharpen it.")
