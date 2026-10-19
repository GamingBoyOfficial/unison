"""Steer the swarm by hand, or hand the wheel to an autopilot.

  python experiments/play.py          # the Y-branch split task
  python experiments/play.py split_long   # same, with a 1600-step time budget
  python experiments/play.py tube     # the straight tube
  python experiments/play.py bifurcation   # curved, tapering daughters with a sharp carina
  python experiments/play.py split --benchmark
                                      # the exact swarms the AI is scored on (seeds 20000+),
                                      # for a fair human-vs-AI comparison

Controls (click the plot window first):
  Arrow keys or W/A/S/D : set the field direction (hold to keep pushing)
  C                     : cycle the driver: Manual -> autopilot(s) -> Manual
  R                     : start a new episode on a fresh random swarm
  Q                     : quit

Each episode lasts a fixed number of steps, exactly like the scored evaluation.
When it ends, the final score is shown. If one driver controlled the WHOLE episode,
the result is appended to results/play_scores.csv, and a manual run's key presses
are saved to results/actions_<scenario>_<seed>.csv so it can be replayed exactly.
Run  python experiments/summarize_play.py  to get mean +/- 95% CI per driver.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from unison.arena import inside_mask

from unison.bots import make_bots
from unison.controllers import PushTowardTarget, SwitchController, WallHugging
from unison.evaluate import build_obs
from unison.playlog import next_seed_base, read_rows
from unison.scenarios import SCENARIOS
from unison.simulator import Simulator

RESULTS = Path("results")
SEED_BASE = 5000          # play episodes use their own seeds, apart from evaluation
BENCH_SEED_BASE = 20000   # benchmark mode: the swarms evaluate_model.py scores the AI on

KEY_DIRS = {"up": (0, 1), "w": (0, 1), "down": (0, -1), "s": (0, -1),
            "left": (-1, 0), "a": (-1, 0), "right": (1, 0), "d": (1, 0)}

AUTOPILOTS = {
    "split": [("Push toward target", PushTowardTarget()),
              ("Switch (0.4)", SwitchController(0.4))],
    "split_long": [("Push toward target", PushTowardTarget())],
    "bifurcation": [("Push toward target", PushTowardTarget())],
    "tube": [("Push toward target", PushTowardTarget()),
             ("Wall-hugging", WallHugging())],
}
# A trained AI will be added to this list in a later phase.


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    benchmark = "--benchmark" in sys.argv
    name = argv[0] if argv else "split"
    if name not in SCENARIOS:
        raise SystemExit(f"Unknown scenario '{name}'. Choose from: {', '.join(SCENARIOS)}")
    scenario = SCENARIOS[name]()
    drivers = [("Manual", None)] + AUTOPILOTS[name]

    for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[k] = []

    arena = scenario.make_arena()
    task = scenario.make_task(arena)
    # Continue numbering above every seed already logged, so a swarm is never reused.
    seed_base = next_seed_base(read_rows(RESULTS / "play_scores.csv"), name, SEED_BASE)
    bench_log = RESULTS / "benchmark_scores.csv"
    bench_done = [int(r["seed"]) for r in read_rows(bench_log) if r["scenario"] == name]
    bench_next = max([BENCH_SEED_BASE - 1] + bench_done) + 1
    state = {"episode": 0, "driver": 0, "held": set(), "sim": None, "seed": None,
             "step": 0, "actions": [], "used": set(), "done": False, "final": "",
             "bench_next": bench_next}

    def new_episode():
        state["episode"] += 1
        if benchmark:
            state["seed"] = state["bench_next"]
            state["bench_next"] += 1
        else:
            state["seed"] = seed_base + state["episode"]
        rng = np.random.default_rng(state["seed"])
        bots = make_bots(rng, start_center=scenario.start_center)
        state["sim"] = Simulator(arena, bots, rng)
        state.update(step=0, actions=[], used=set(), done=False, final="")

    new_episode()
    xmin, xmax, ymin, ymax = arena.bounds
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.4, ymax + 0.4)
    ax.set_aspect("equal")
    ax.set_facecolor("#05070a")
    ext = (xmin - 0.3, xmax + 0.3, ymin - 0.4, ymax + 0.4)
    mask = inside_mask(arena, ext)
    ax.imshow(np.where(mask, 1.0, np.nan), extent=(ext[0], ext[1], ext[2], ext[3]), origin="lower",
              cmap="Greys", vmin=0, vmax=6, alpha=0.9, zorder=0, interpolation="nearest")
    if hasattr(task, "target_circles"):
        for cx, cy, rad in task.target_circles():
            ax.add_patch(Circle((cx, cy), rad, facecolor="#1D9E75", alpha=0.3, zorder=1))
    else:
        for (x0, x1, y0, y1) in task.target_regions():
            x1 = min(x1, xmax)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#1D9E75",
                                   alpha=0.3, zorder=1))
    ax.annotate("flow", xy=(0.8, ymax - 0.3), xytext=(2.6, ymax - 0.3), color="#EF9F27",
                arrowprops=dict(arrowstyle="->", color="#EF9F27"), va="center")
    sim = state["sim"]
    scat = ax.scatter(sim.pos[:, 0], sim.pos[:, 1], c=sim.bots.speed_factor,
                      cmap="cool", s=10, vmin=0.7, vmax=1.3, zorder=2)
    title = ax.set_title("")
    fig.text(0.5, 0.015, "Arrows/WASD: steer    C: change driver    R: new episode    Q: quit",
             ha="center", color="#8b93a1")

    def field_now():
        f = np.zeros(2)
        for k in state["held"]:
            f += KEY_DIRS[k]
        n = np.linalg.norm(f)
        return f / n if n > 0 else f

    def finish():
        s = state["sim"]
        score = task.score(s.pos)
        state["done"] = True
        if benchmark:
            if state["used"] == {"Manual"}:
                RESULTS.mkdir(exist_ok=True)
                new_file = not bench_log.exists()
                with bench_log.open("a", newline="") as fh:
                    w = csv.writer(fh)
                    if new_file:
                        w.writerow(["scenario", "seed", "score"])
                    w.writerow([name, state["seed"], f"{score:.4f}"])
                np.savetxt(RESULTS / f"bench_actions_{name}_{state['seed']}.csv",
                           np.array(state["actions"]), delimiter=",", header="fx,fy", comments="")
                state["final"] = f"FINAL SCORE {score:.3f}  (benchmark swarm {state['seed']} logged)  press R for next"
            else:
                state["final"] = f"FINAL SCORE {score:.3f}  (driver changed, not logged)"
            return
        if len(state["used"]) != 1:
            state["final"] = f"FINAL SCORE {score:.3f}  (driver changed mid-episode, not logged)"
            return
        label = next(iter(state["used"]))
        RESULTS.mkdir(exist_ok=True)
        log = RESULTS / "play_scores.csv"
        new_file = not log.exists()
        with log.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["scenario", "seed", "driver", "score"])
            w.writerow([name, state["seed"], label, f"{score:.4f}"])
        if label == "Manual":
            np.savetxt(RESULTS / f"actions_{name}_{state['seed']}.csv",
                       np.array(state["actions"]), delimiter=",",
                       header="fx,fy", comments="")
        state["final"] = f"FINAL SCORE {score:.3f}  (logged as {label})  press R for next"

    def on_press(e):
        if e.key in KEY_DIRS:
            state["held"].add(e.key)
        elif e.key == "c" and not benchmark:
            state["driver"] = (state["driver"] + 1) % len(drivers)
        elif e.key == "r":
            new_episode()
            scat.set_array(state["sim"].bots.speed_factor)
        elif e.key == "q":
            plt.close(fig)

    def update(_):
        s = state["sim"]
        label, ctrl = drivers[state["driver"]]
        if not state["done"]:
            if ctrl is None:
                f = field_now()
            else:
                obs = build_obs(task, s, state["step"] / scenario.steps)
                f = ctrl.act(obs, s.rng)
            state["actions"].append(f)
            state["used"].add(label)
            s.step(f)
            state["step"] += 1
            if state["step"] >= scenario.steps:
                finish()
        scat.set_offsets(s.pos)
        tag = f"BENCHMARK swarm {state['seed']}   " if benchmark else ""
        head = f"{tag}[{label}]  step {state['step']}/{scenario.steps}   {task.status(s.pos)}"
        title.set_text(head + ("\n" + state["final"] if state["done"] else ""))
        return scat, title

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", lambda e: state["held"].discard(e.key))
    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
