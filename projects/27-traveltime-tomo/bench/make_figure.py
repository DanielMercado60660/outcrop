"""Make a synthetic cross-grid fixed-ray tomography figure."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call


def main():
    nx, nz = 16, 16
    width, depth = 1600.0, 1600.0
    x = np.linspace(0, width, nx + 1, dtype=np.float32)
    z = np.linspace(0, depth, nz + 1, dtype=np.float32)
    levels = np.linspace(25, 1575, 32, dtype=np.float32)
    a, b = np.meshgrid(levels, levels, indexing="ij")
    a, b = a.ravel(), b.ravel()
    sources = np.concatenate((np.column_stack((np.zeros_like(a), a)),
                              np.column_stack((a, np.zeros_like(a))))).astype(np.float32)
    receivers = np.concatenate((np.column_stack((np.full_like(b, width), b)),
                                np.column_stack((b, np.full_like(b, depth))))).astype(np.float32)
    matrix = call("metal", "ray_matrix", x_edges=x, z_edges=z,
                  sources=sources, receivers=receivers)["matrix"]
    xx, zz = np.meshgrid((x[:-1] + x[1:]) / 2, (z[:-1] + z[1:]) / 2)
    anomaly = (0.000035 * np.exp(-((xx - 520) ** 2 / 260**2 + (zz - 750) ** 2 / 340**2))
               - 0.000035 * np.exp(-((xx - 1100) ** 2 / 280**2 + (zz - 900) ** 2 / 260**2)))
    prior = np.full(nx * nz, 0.0005, dtype=np.float32)
    truth = (0.0005 + anomaly).astype(np.float32).ravel()
    times = (matrix @ truth).astype(np.float32)
    recovered = call("metal", "solve_tomography", matrix=matrix, travel_times=times,
                     reference_slowness=prior, damping=20.0)["slowness"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharex=True, sharey=True, constrained_layout=True)
    for ax, values, title in zip(axes, (truth, recovered), ("Synthetic truth", "Recovered model"), strict=True):
        im = ax.imshow((1.0 / values).reshape(nz, nx), extent=(0, width, depth, 0),
                       cmap="RdBu_r", vmin=1850, vmax=2150, interpolation="nearest")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Distance (m)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Depth (m)")
    colorbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.025)
    colorbar.set_label("Velocity (m/s)")
    fig.suptitle("Straight-ray travel-time tomography", fontsize=15, weight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
