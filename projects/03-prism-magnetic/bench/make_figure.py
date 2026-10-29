"""Render a synthetic prism magnetic anomaly using the Metal backend."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call


def main():
    east = np.linspace(-1000, 1000, 241, dtype=np.float32)
    north = np.linspace(-800, 800, 193, dtype=np.float32)
    xx, yy = np.meshgrid(east, north)
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 30, dtype=np.float32)))
    prisms = np.array([[-520, -180, -260, 160, -420, -100],
                       [100, 450, 50, 370, -390, -80],
                       [-60, 200, -540, -280, -490, -150]], dtype=np.float32)
    magnetization = np.array([[1.2, 0.5, -2.0], [0.9, 0.7, -1.6],
                              [-0.6, 1.2, -1.3]], dtype=np.float32)
    field = np.array([0.36, 0.48, -0.8], dtype=np.float32)
    anomaly = call("metal", "prism_magnetic", coordinates=coordinates, prisms=prisms,
                   magnetization=magnetization, inducing_field=field)["total_field"].reshape(xx.shape)
    limit = max(float(np.max(np.abs(anomaly))), 1e-12)
    fig, ax = plt.subplots(figsize=(8.2, 6.4), layout="constrained")
    mesh = ax.pcolormesh(xx, yy, anomaly, shading="auto", cmap="RdBu_r",
                         norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    for west, east_edge, south, north_edge, _, _ in prisms:
        ax.plot([west, east_edge, east_edge, west, west],
                [south, south, north_edge, north_edge, south],
                color="black", linewidth=0.9, alpha=0.7, linestyle="--")
    ax.set(xlabel="Easting (m)", ylabel="Northing (m)",
           title="Prism magnetic forward model | total-field anomaly")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, label="Total-field anomaly (nT)", shrink=0.88, pad=0.025)
    ax.text(0.02, 0.02, "Observation height: 30 m  ·  Dashed: prism footprints",
            transform=ax.transAxes, fontsize=8, color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 4})
    fig.savefig(PROJECT / "figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
