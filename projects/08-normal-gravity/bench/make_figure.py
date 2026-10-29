"""Render WGS84 normal gravity versus latitude and altitude using MLX."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call


def main():
    lat = np.linspace(-90, 90, 601, dtype=np.float32)
    altitude_km = np.linspace(0, 1000, 401, dtype=np.float32)
    latitude, altitude = np.meshgrid(lat, altitude_km)
    gravity = call("mlx", "normal_gravity", latitude=latitude,
                   height=(altitude * 1000).astype(np.float32))["gravity"]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 5.1), layout="constrained")
    image = ax.pcolormesh(lat, altitude_km, gravity, shading="auto", cmap="viridis")
    contours = ax.contour(lat, altitude_km, gravity, levels=6,
                          colors="white", linewidths=0.6, alpha=0.65)
    ax.clabel(contours, inline=True, fmt="%.1f", fontsize=8)
    ax.set(xlabel="Geodetic latitude (degrees)", ylabel="Height above ellipsoid (km)",
           title="WGS84 normal gravity · MLX")
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    fig.colorbar(image, ax=ax, pad=0.025, label="Normal gravity (m/s²)")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
