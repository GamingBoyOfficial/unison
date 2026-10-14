"""Let the AI practice on its own (evolution strategies), starting from the
model that copied you.

Each generation: make mutated copies of the model, score every copy on the same
fresh swarms, then move the model toward the mutations that scored better.
Uses all your CPU cores. Press Ctrl+C any time: the best model so far is saved.

Run:
  python experiments/train_es.py
  python experiments/train_es.py --generations 100 --workers 8

Needs a model from train_imitation.py first (results/imitation_split.joblib).
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import joblib
import numpy as np

from unison.evaluate import run_episode
from unison.policy import NumpyPolicy
from unison.scenarios import SCENARIOS

TRAIN_SEED = 40000   # swarms used for learning (fresh ones every generation)
VAL_SEED = 50000     # swarms used to pick the best model
# final testing uses seeds 20000+ (evaluate_model.py), never seen in training


def score_task(task):
    """Worker: mean score of one candidate on a list of swarms."""
    arrays, params, scenario_name, seeds = task
    policy = NumpyPolicy(*arrays, params)
    scenario = SCENARIOS[scenario_name]()
    return float(np.mean([run_episode(policy, s, scenario) for s in seeds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/imitation_split.joblib")
    ap.add_argument("--scenario", default="split")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=12, help="mutation pairs per generation")
    ap.add_argument("--swarms", type=int, default=4, help="swarms per candidate")
    ap.add_argument("--sigma", type=float, default=0.02, help="mutation size")
    ap.add_argument("--lr", type=float, default=0.02, help="step size")
    ap.add_argument("--val-swarms", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = NumpyPolicy.from_sklearn(joblib.load(args.model))
    arrays = (base.mean, base.scale, base.shapes)
    out = Path(args.out or f"results/es_{args.scenario}.npz")
    out.parent.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    theta = base.params.copy()
    d = theta.size
    val_seeds = list(range(VAL_SEED, VAL_SEED + args.val_swarms))

    with mp.get_context("spawn").Pool(args.workers) as pool:
        def run(thetas, seeds):
            return np.array(pool.map(
                score_task, [(arrays, t, args.scenario, seeds) for t in thetas]))

        start_val = run([theta], val_seeds)[0]
        best_val, best_theta = start_val, theta.copy()
        print(f"{d} parameters, {args.workers} workers. "
              f"Starting model scores {start_val:.3f} on {args.val_swarms} validation swarms.\n")
        print(f"{'Gen':>4}{'Candidates':>12}{'Validation':>12}{'Best so far':>13}{'Time':>8}")
        t0 = time.time()
        try:
            for gen in range(1, args.generations + 1):
                eps = rng.standard_normal((args.pairs, d))
                cands = np.vstack([theta + args.sigma * eps, theta - args.sigma * eps])
                seeds = [TRAIN_SEED + gen * args.swarms + i for i in range(args.swarms)]
                scores = run(cands, seeds)
                ranks = scores.argsort().argsort() / (len(scores) - 1) - 0.5   # -0.5 .. 0.5
                u_plus, u_minus = ranks[:args.pairs], ranks[args.pairs:]
                direction = ((u_plus - u_minus)[:, None] * eps).sum(axis=0) / (2 * args.pairs)
                theta = theta + args.lr * direction
                line = f"{gen:>4}{scores.mean():>12.3f}"
                if gen % args.eval_every == 0 or gen == args.generations:
                    val = run([theta], val_seeds)[0]
                    if val > best_val:
                        best_val, best_theta = val, theta.copy()
                        base.with_params(best_theta).save(out)
                    line += f"{val:>12.3f}{best_val:>13.3f}{time.time() - t0:>7.0f}s"
                print(line, flush=True)
        except KeyboardInterrupt:
            print("\nStopped early.")

    base.with_params(best_theta).save(out)
    print(f"\nBest validation score {best_val:.3f} (started at {start_val:.3f}). Saved {out}")
    print("Now test it on swarms it has never seen:\n"
          f"  python experiments/evaluate_model.py --model {out}")


if __name__ == "__main__":
    main()
