"""
MicroSwarm - Phase 1
Many imperfect microbots, ONE global control signal.

Controls (click the plot window first):
  Arrow keys or W/A/S/D : set the field direction (hold to keep pushing)
  R                     : reset the swarm
  Q                     : quit

Requires: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N_BOTS = 300          # how many bots
SPREAD = 0.15         # how different the bots are (0.01 = nearly identical)
DT = 0.05             # time step
NOISE = 0.02          # random jitter (Brownian motion)
BOX = 5.0             # arena goes from -BOX to +BOX on both axes


def make_swarm(rng, n=N_BOTS, spread=SPREAD):
    """Create bots. Each bot is just a few numbers."""
    size = rng.normal(1.0, spread, n).clip(0.5, 1.5)
    magnetism = rng.normal(1.0, spread, n).clip(0.5, 1.5)
    drag = 1.0 + 0.5 * (size - 1.0)          # bigger bots feel more drag
    mobility = 1.0 / drag
    speed_factor = mobility * magnetism      # how strongly this bot reacts
    pos = rng.normal(0.0, 0.3, (n, 2))       # start as a tight cloud
    return pos, speed_factor


def step(pos, speed_factor, field, rng, dt=DT, noise=NOISE):
    """Move every bot one step. All bots feel the SAME field."""
    vel = speed_factor[:, None] * field[None, :]
    pos = pos + vel * dt + rng.normal(0.0, noise, pos.shape)
    return np.clip(pos, -BOX, BOX)           # walls


def spread_of(pos):
    """A simple number for how spread out the swarm is."""
    return float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))


KEY_DIRS = {
    "up": (0, 1), "w": (0, 1),
    "down": (0, -1), "s": (0, -1),
    "left": (-1, 0), "a": (-1, 0),
    "right": (1, 0), "d": (1, 0),
}


def main():
    for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[k] = []                 # free up the keys for steering

    rng = np.random.default_rng(42)
    state = {"pos": None, "sf": None, "held": set()}
    state["pos"], state["sf"] = make_swarm(rng)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-BOX, BOX)
    ax.set_ylim(-BOX, BOX)
    ax.set_aspect("equal")
    ax.set_facecolor("#0e1116")
    scat = ax.scatter(
        state["pos"][:, 0], state["pos"][:, 1],
        c=state["sf"], cmap="cool", s=14, vmin=0.7, vmax=1.3,
    )
    title = ax.set_title("")

    def field_now():
        f = np.zeros(2)
        for k in state["held"]:
            f += KEY_DIRS[k]
        norm = np.linalg.norm(f)
        return f / norm if norm > 0 else f

    def on_press(event):
        if event.key in KEY_DIRS:
            state["held"].add(event.key)
        elif event.key == "r":
            state["pos"], state["sf"] = make_swarm(rng)
            scat.set_array(state["sf"])
        elif event.key == "q":
            plt.close(fig)

    def on_release(event):
        state["held"].discard(event.key)

    def update(_):
        state["pos"] = step(state["pos"], state["sf"], field_now(), rng)
        scat.set_offsets(state["pos"])
        title.set_text(f"Swarm spread: {spread_of(state['pos']):.2f}   "
                       f"(color = how strongly each bot reacts)")
        return scat, title

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)
    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()