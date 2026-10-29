"""Plot direct rays in a synthetic, horizontally layered Earth."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call


def main():
    interfaces = np.array([350, 850], dtype=np.float32)
    velocities = np.array([1800, 2400, 2000], dtype=np.float32)
    angles = np.linspace(-0.65, 0.65, 41, dtype=np.float32)
    # Interface-adjacent depths keep each evaluated endpoint inside a layer.
    depths = [349.999, 849.999, 1500.0]
    endpoints = [call("metal", "shoot_layered", interfaces=interfaces,
                      velocities=velocities, angles=angles, depth=depth)["endpoints"][:, 0]
                 for depth in depths]
    fig, ax = plt.subplots(figsize=(10, 6.3), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = BoundaryNorm([1700, 1900, 2200, 2500], cmap.N)
    for top, bottom, velocity in zip([0, 350, 850], [350, 850, 1500], velocities):
        ax.axhspan(top / 1000, bottom / 1000, color=cmap(norm(velocity)), alpha=0.9)
    for i in range(len(angles)):
        xs = np.array([0, *(x[i] for x in endpoints)]) / 1000
        ax.plot(xs, np.array([0, *depths]) / 1000, color="white", alpha=0.68, lw=0.8)
    for interface in interfaces:
        ax.axhline(interface / 1000, color="#d7e7e5", lw=0.8, alpha=0.65)
    ax.scatter([0], [0], marker="*", s=130, color="#ffe19a", edgecolor="#202d36", zorder=5)
    ax.set(xlim=(-2.6, 2.6), ylim=(1.5, 0), xlabel="Horizontal distance (km)",
           ylabel="Depth (km)", title="Direct ray tracing through layered velocities")
    ax.grid(color="white", alpha=0.15, linewidth=0.5)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02, ticks=[1800, 2000, 2400], shrink=0.88)
    cbar.set_label("P-wave velocity (m/s)")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
