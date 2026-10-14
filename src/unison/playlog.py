"""Helpers for the play log (results/play_scores.csv)."""

from __future__ import annotations

import csv
from pathlib import Path


def read_rows(log: Path) -> list[dict]:
    if not log.exists():
        return []
    with log.open() as fh:
        return list(csv.DictReader(fh))


def latest_per_seed(rows: list[dict], scenario: str | None = None) -> list[dict]:
    """Keep only the most recent row for each (scenario, driver, seed).

    A seed identifies one exact swarm. If the same swarm was played more than once,
    only the latest attempt is kept, so each swarm counts once.
    """
    out: dict = {}
    for r in rows:
        if scenario is not None and r["scenario"] != scenario:
            continue
        out[(r["scenario"], r["driver"], int(r["seed"]))] = r
    return list(out.values())


def next_seed_base(rows: list[dict], scenario: str, floor: int) -> int:
    """Largest seed already logged for this scenario (or floor). New episodes
    start just above it, so a swarm is never reused across sessions."""
    return max([floor] + [int(r["seed"]) for r in rows if r["scenario"] == scenario])
