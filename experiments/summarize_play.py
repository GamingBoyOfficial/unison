"""Summarize the episodes logged by play.py: mean score +/- 95% CI per driver.

Each swarm (seed) counts once. If a swarm was played more than once, only the
latest attempt is used, because repeats of a swarm you have already seen are not
independent samples.

Run:  python experiments/summarize_play.py
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from unison.playlog import latest_per_seed, read_rows

rows = read_rows(Path("results/play_scores.csv"))
if not rows:
    raise SystemExit("No results yet. Play some episodes with experiments/play.py first.")

kept = latest_per_seed(rows)
if len(kept) < len(rows):
    print(f"Note: {len(rows) - len(kept)} episode(s) were repeats of a swarm already "
          f"played; only the latest attempt per swarm is counted.\n")

groups = defaultdict(list)
for r in kept:
    groups[(r["scenario"], r["driver"])].append(float(r["score"]))

print(f"{'Scenario':<12}{'Driver':<24}{'Swarms':>8}{'Mean':>9}{'  95% CI':>12}")
print("-" * 65)
for (scenario, driver), scores in sorted(groups.items()):
    s = np.array(scores)
    ci = f"+/- {1.96 * s.std(ddof=1) / np.sqrt(len(s)):.3f}" if len(s) > 1 else "(need 2+)"
    print(f"{scenario:<12}{driver:<24}{len(s):>8}{s.mean():>9.3f}{ci:>12}")
