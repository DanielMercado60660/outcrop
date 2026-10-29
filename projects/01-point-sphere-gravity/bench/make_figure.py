"""Render a synthetic gravity anomaly using the Metal implementation."""

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
    x = np.linspace(-6, 6, 400, dtype=np.float32)
    y = np.linspace(-5, 5, 340, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    coordinates = np.stack((xx * 1000, yy * 1000, np.zeros_like(xx)), axis=-1)
    centers = np.array([[-1900, 300, -1500], [1800, -900, -2300], [1200, 2300, -1200]], dtype=np.float32)
    masses = np.array([5e15, 1.1e16, 2e15], dtype=np.float32)
    radii = np.array([850, 1300, 0], dtype=np.float32)
    field = call("metal", "gravity", coordinates=coordinates, centers=centers, masses=masses, radii=radii)
    downward_mgal = -field["acceleration"][..., 2] * 1e5

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 6.3), layout="constrained")
    image = ax.pcolormesh(x, y, downward_mgal, shading="auto", cmap="viridis", rasterized=True)
    ax.contour(xx, yy, downward_mgal, levels=9, colors="white", linewidths=0.55, alpha=0.48)
    ax.scatter(centers[:, 0] / 1000, centers[:, 1] / 1000, marker="+", s=85, linewidths=1.5, color="white", label="Source projections")
    ax.set(xlabel="East–west position (km)", ylabel="North–south position (km)", title="Metal · gravity of buried spheres and a point mass")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.88, fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025, shrink=0.92)
    colorbar.set_label("Downward gravitational acceleration (mGal)")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
