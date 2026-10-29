"""Render a synthetic heterogeneous fast-marching travel-time field."""
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
    nx, nz = 280, 200
    dx = dz = 0.025
    x, z = np.meshgrid(np.arange(nx) * dx, np.arange(nz) * dz)
    source_x, source_z = 1.65, 1.05
    phi = (np.hypot(x - source_x, z - source_z) - 0.09).astype(np.float32)
    speed = (2.25 + 0.20 * z
             + 0.95 * np.exp(-((x - 4.0) / 1.15)**2 - ((z - 2.4) / 0.65)**2)
             - 0.38 * np.exp(-((z - 3.65) / 0.45)**2)).astype(np.float32)
    travel = call("metal", "travel_time", phi=phi, speed=speed,
                  spacing=[dz, dx], order=1)["time"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.6, 5.5), constrained_layout=True)
    im = ax.imshow(speed, extent=[0, x.max(), z.max(), 0], cmap="viridis",
                   aspect="equal", interpolation="bilinear")
    levels = np.arange(0.2, float(np.max(travel)), 0.2)
    contours = ax.contour(x, z, travel, levels=levels, colors="white",
                          linewidths=0.8, alpha=0.85)
    ax.clabel(contours, contours.levels[::2], inline=True, fmt="%.1f", fontsize=7)
    ax.plot(source_x, source_z, "*", color="#ffdb63", markersize=12,
            markeredgecolor="#202020", markeredgewidth=0.7, label="Source")
    ax.set(xlabel="Distance (km)", ylabel="Depth (km)",
           title="Fast marching · travel-time wavefronts (s)")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.colorbar(im, ax=ax, pad=0.025, label="Propagation speed (km/s)")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
