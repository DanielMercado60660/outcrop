"""Render a synthetic deep-water tsunami pulse after propagation."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call


def main():
    n, dx, depth, steps = 4000, 100.0, 4000.0, 1000
    x = (np.arange(n) + 0.5) * dx
    c = np.sqrt(9.81 * depth)
    dt = 0.4 * dx / c
    eta = np.exp(-0.5 * ((x - 200000.0) / 7000.0) ** 2)
    initial = np.column_stack((eta, np.zeros(n))).astype(np.float32)
    state = call("metal", "propagate", initial=initial, dx=dx, dt=float(dt),
                 steps=steps, mode="linear", depth=depth)["state"]

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.plot(x / 1000, eta, color="#738594", linestyle="--", linewidth=1.6,
            label="Initial uplift")
    ax.fill_between(x / 1000, state[:, 0], color="#39a5b2", alpha=0.3)
    ax.plot(x / 1000, state[:, 0], color="#087f8c", linewidth=2.4,
            label=f"After {steps * dt / 60:.1f} min")
    ax.axhline(0, color="#a4b2bb", linewidth=0.8)
    ax.set(xlim=(90, 310), ylim=(-0.04, 1.12), xlabel="Distance (km)",
           ylabel="Surface displacement (m)",
           title="Tsunami pulse splitting · shallow-water Rusanov solver")
    ax.grid(axis="y", color="#dce4e7", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
