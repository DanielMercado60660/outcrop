"""Render a synthetic first-arrival travel-time field from the Metal solver."""
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
    nz, nx = 96, 144
    dz = dx = 20.0
    z, x = np.indices((nz, nx))
    anomaly = np.exp(-((x - 88) / 27)**2 - ((z - 53) / 18)**2)
    velocity = 1800 + 4.5 * z + 650 * anomaly
    slowness = (1.0 / velocity).astype(np.float32)
    source = np.array([[0, 24]], dtype=np.int32)
    receivers = np.column_stack((np.zeros(24), np.linspace(0, nx - 1, 24))).astype(np.int32)
    out = call("metal", "misfit_and_grad", slowness=slowness, spacing=[dz, dx],
               sources=source, receivers=receivers,
               observed=np.zeros((1, len(receivers)), np.float32),
               weights=np.ones((1, len(receivers)), np.float32))
    times = out["traveltime"][0]
    xx, zz = np.arange(nx) * dx / 1000, np.arange(nz) * dz / 1000
    fig, ax = plt.subplots(figsize=(9.2, 5.2), layout="constrained")
    image = ax.pcolormesh(xx, zz, times, shading="auto", cmap="viridis")
    levels = np.arange(0.15, float(times.max()), 0.15)
    contours = ax.contour(xx, zz, times, levels=levels, colors="white", linewidths=0.7, alpha=0.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    ax.scatter(source[0, 1] * dx / 1000, 0, marker="*", s=115,
               color="#ffdc68", edgecolor="#262626", linewidth=0.7, zorder=3, label="Source")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set(xlabel="Distance (km)", ylabel="Depth (km)",
           title="First arrivals through a fast inclusion · Metal eikonal")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.colorbar(image, ax=ax, label="Travel time (s)", pad=0.025)
    fig.savefig(ROOT / "figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
