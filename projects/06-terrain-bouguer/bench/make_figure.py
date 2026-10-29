"""Render synthetic terrain and its Metal-computed gravity."""
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
    x = np.linspace(-6000, 6000, 64, dtype=np.float32)
    y = np.linspace(-6000, 6000, 64, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    hill = 650 * np.exp(-((xx + 1800)**2 / (2400**2) + (yy - 1200)**2 / (1800**2)))
    ridge = 430 * np.exp(-((xx - 1900)**2 / (1600**2) + (yy + 800)**2 / (3100**2)))
    depression = 250 * np.exp(-((xx + 2500)**2 / (1800**2) + (yy + 2700)**2 / (1800**2)))
    surface = (100 + hill + ridge - depression).astype(np.float32)
    reference = np.zeros_like(surface)
    density = np.full_like(surface, 2670)
    survey_x = np.linspace(x[0], x[-1], 72, dtype=np.float32)
    survey_y = np.linspace(y[0], y[-1], 72, dtype=np.float32)
    sx, sy = np.meshgrid(survey_x, survey_y)
    coords = np.column_stack((sx.ravel(), sy.ravel(),
                              np.full(sx.size, 1000, dtype=np.float32)))
    output = call("metal", "terrain_bouguer", easting=x, northing=y,
                  surface=surface, reference=reference, density=density,
                  coordinates=coords)
    gravity = output["layer_gz"].reshape(sx.shape)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "figure.facecolor": "#f8f9fa", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    extent = [x[0] / 1000, x[-1] / 1000, y[0] / 1000, y[-1] / 1000]
    topo = axes[0].imshow(surface, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    axes[0].contour(xx / 1000, yy / 1000, surface, levels=7, colors="white",
                    linewidths=0.45, alpha=0.5)
    axes[0].set_title("Synthetic signed relief")
    fig.colorbar(topo, ax=axes[0], shrink=0.84, label="Elevation relative to datum (m)")
    grav = axes[1].imshow(gravity, origin="lower", extent=extent, cmap="magma", aspect="equal")
    axes[1].set_title("Finite-layer gravity at 1 km elevation")
    fig.colorbar(grav, ax=axes[1], shrink=0.84, label="Downward gravity (mGal)")
    for ax in axes:
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
    fig.suptitle("Terrain gravity · signed prism layer (Metal)", fontsize=15, weight="bold")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
