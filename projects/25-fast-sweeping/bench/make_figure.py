"""Render a synthetic heterogeneous eikonal travel-time field."""

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
    nx, nz, spacing = 241, 161, 10.0
    x = np.arange(nx, dtype=np.float32) * spacing
    z = np.arange(nz, dtype=np.float32) * spacing
    xx, zz = np.meshgrid(x, z, indexing="xy")
    source_x, source_z = 1200.0, 200.0
    phi = np.asarray(np.hypot(xx - source_x, zz - source_z) - 0.01 * spacing,
                     dtype=np.float32)
    lens = np.exp(-((xx - 1250.0) / 450.0) ** 2 - ((zz - 800.0) / 280.0) ** 2)
    speed = np.asarray(1900.0 + 0.6 * zz + 400.0 / (1.0 + np.exp(-(zz - 1100.0) / 100.0))
                       - 550.0 * lens, dtype=np.float32)
    travel = call("metal", "travel_time", phi=phi, speed=speed, spacing=spacing)["travel"]

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    extent = (x[0] / 1000, x[-1] / 1000, z[-1] / 1000, z[0] / 1000)
    image = ax.imshow(travel, extent=extent, origin="upper", cmap="viridis", aspect="equal")
    levels = np.linspace(0.1, float(np.max(travel)) * 0.95, 9)
    contours = ax.contour(x / 1000, z / 1000, travel, levels=levels,
                          colors="white", linewidths=0.7, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.1f s")
    ax.plot(source_x / 1000, source_z / 1000, "*", color="#ffdb5c",
            markersize=13, markeredgecolor="#202020", markeredgewidth=0.7, label="Source")
    ax.set(xlabel="Distance (km)", ylabel="Depth (km)",
           title="Fast sweeping: travel time through a synthetic velocity model")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.colorbar(image, ax=ax, label="Travel time (s)", shrink=0.9, pad=0.02)
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
