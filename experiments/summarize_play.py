"""Summarize the episodes logged by play.py: mean score +/- 95% CI per driver.

Run:  python experiments/summarize_play.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

log = Path("results/play_scores.csv")
if not log.exists():
    raise SystemExit("No results yet. Play some episodes with experiments/play.py first.")

groups = defaultdict(list)
with log.open() as fh:
    for row in csv.DictReader(fh):
        groups[(row["scenario"], row["driver"])].append(float(row["score"]))

print(f"{'Scenario':<10}{'Driver':<24}{'Episodes':>9}{'Mean':>9}{'  95% CI':>12}")
print("-" * 64)
for (scenario, driver), scores in sorted(groups.items()):
    s = np.array(scores)
    ci = f"+/- {1.96 * s.std(ddof=1) / np.sqrt(len(s)):.3f}" if len(s) > 1 else "(need 2+)"
    print(f"{scenario:<10}{driver:<24}{len(s):>9}{s.mean():>9.3f}{ci:>12}")
print("\nEach episode is on a different random swarm, so this is a fair comparison.")
