"""Steer the swarm through a tube with flow, by hand.

Controls (click the plot window first):
  Arrow keys or W/A/S/D : set the field direction (hold to keep pushing)
  R                     : reset
  Q                     : quit
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from unison.arena import TubeArena
from unison.bots import make_bots
from unison.simulator import Simulator
from unison.task import ReachTargetTask

KEY_DIRS = {"up": (0, 1), "w": (0, 1), "down": (0, -1), "s": (0, -1),
            "left": (-1, 0), "a": (-1, 0), "right": (1, 0), "d": (1, 0)}


def main():
    for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[k] = []

    arena = TubeArena()
    rng = np.random.default_rng(7)
    task = ReachTargetTask(target_x=arena.length - 1.0)
    state = {"sim": None, "held": set()}

    def new_sim():
        bots = make_bots(rng, start_center=(0.5, 0.0))
        state["sim"] = Simulator(arena, bots, rng)

    new_sim()
    xmin, xmax, ymin, ymax = arena.bounds
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(xmin - 0.2, xmax + 0.2)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_aspect("equal")
    ax.set_facecolor("#0e1116")
    ax.axhline(ymin, color="#8b93a1")
    ax.axhline(ymax, color="#8b93a1")
    ax.axvspan(task.target_x, xmax, color="#1D9E75", alpha=0.25)
    ax.text(task.target_x + 0.1, ymax - 0.2, "target", color="#5DCAA5")
    ax.annotate("flow", xy=(1.5, 0.75), xytext=(3.2, 0.75), color="#EF9F27",
                arrowprops=dict(arrowstyle="->", color="#EF9F27"), va="center")
    sim = state["sim"]
    scat = ax.scatter(sim.pos[:, 0], sim.pos[:, 1], c=sim.bots.speed_factor,
                      cmap="cool", s=12, vmin=0.7, vmax=1.3)
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
        title.set_text(f"Reached target: {task.fraction_reached(s.pos) * 100:.0f}% of bots")
        return scat, title

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", lambda e: state["held"].discard(e.key))
    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
