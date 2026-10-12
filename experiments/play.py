"""Steer the swarm by hand in any scenario.

  python experiments/play.py          # the Y-branch split task
  python experiments/play.py tube     # the straight tube

Controls (click the plot window first):
  Arrow keys or W/A/S/D : set the field direction (hold to keep pushing)
  R                     : reset
  Q                     : quit
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from unison.bots import make_bots
from unison.scenarios import SCENARIOS
from unison.simulator import Simulator

KEY_DIRS = {"up": (0, 1), "w": (0, 1), "down": (0, -1), "s": (0, -1),
            "left": (-1, 0), "a": (-1, 0), "right": (1, 0), "d": (1, 0)}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "split"
    if name not in SCENARIOS:
        raise SystemExit(f"Unknown scenario '{name}'. Choose from: {', '.join(SCENARIOS)}")
    scenario = SCENARIOS[name]()

    for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[k] = []

    arena = scenario.make_arena()
    task = scenario.make_task(arena)
    rng = np.random.default_rng(7)
    state = {"sim": None, "held": set()}

    def new_sim():
        bots = make_bots(rng, start_center=scenario.start_center)
        state["sim"] = Simulator(arena, bots, rng)

    new_sim()
    xmin, xmax, ymin, ymax = arena.bounds
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.4, ymax + 0.4)
    ax.set_aspect("equal")
    ax.set_facecolor("#05070a")
    for (x0, x1, y0, y1) in arena.shapes():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#161b22",
                               edgecolor="#8b93a1", zorder=0))
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

    def field_now():
        f = np.zeros(2)
        for k in state["held"]:
            f += KEY_DIRS[k]
        n = np.linalg.norm(f)
        return f / n if n > 0 else f

    def on_press(e):
        if e.key in KEY_DIRS:
            state["held"].add(e.key)
        elif e.key == "r":
            new_sim()
            scat.set_array(state["sim"].bots.speed_factor)
        elif e.key == "q":
            plt.close(fig)

    def update(_):
        s = state["sim"]
        s.step(field_now())
        scat.set_offsets(s.pos)
        title.set_text(task.status(s.pos))
        return scat, title

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", lambda e: state["held"].discard(e.key))
    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
