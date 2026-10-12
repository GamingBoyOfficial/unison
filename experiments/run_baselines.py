"""Score the baseline controllers. This is the bar a learned controller must beat.

Run:  python experiments/run_baselines.py
"""

from unison.controllers import PushTowardTarget, RandomController
from unison.evaluate import evaluate

N_EPISODES = 50

print(f"Tube task: get bots past the target against the flow ({N_EPISODES} random swarms each)\n")
print(f"{'Controller':<22}{'Mean score':>12}{'  95% CI':>12}")
print("-" * 46)
for name, ctrl in [("Random", RandomController()),
                   ("Push toward target", PushTowardTarget())]:
    r = evaluate(ctrl, n_episodes=N_EPISODES)
    print(f"{name:<22}{r.mean:>12.3f}{'  +/- ' + format(r.ci95, '.3f'):>12}")
print("\nScore = fraction of bots past the target line at the end of the episode.")
