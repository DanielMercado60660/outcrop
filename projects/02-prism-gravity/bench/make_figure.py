"""Plot the downward gravity anomaly of a synthetic buried-prism model."""
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
    axis = np.linspace(-1800, 1800, 220, dtype=np.float32)
    east, north = np.meshgrid(axis, axis)
    xyz = np.column_stack((east.ravel(), north.ravel(),
                           np.full(east.size, 50, dtype=np.float32)))
    prisms = np.array([[-950, -250, -400, 350, -650, -150],
                       [250, 850, 150, 800, -550, -120],
                       [100, 650, -850, -350, -800, -250]], dtype=np.float32)
    density = np.array([750, 550, -500], dtype=np.float32)
    gravity = call("metal", "prism_gravity", coordinates=xyz, prisms=prisms,
                   density=density, field="g_z")["g_z"].reshape(east.shape)
    limit = float(np.max(np.abs(gravity)))
    fig, ax = plt.subplots(figsize=(8, 6.7), layout="constrained")
    image = ax.pcolormesh(east / 1000, north / 1000, gravity, shading="auto",
                          cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
    for west, east_bound, south, north_bound, _, _ in prisms:
        ax.plot(np.array([west, east_bound, east_bound, west, west]) / 1000,
                np.array([south, south, north_bound, north_bound, south]) / 1000,
                color="0.2", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.set(xlabel="Easting (km)", ylabel="Northing (km)",
           title="Prism gravity · downward acceleration at 50 m elevation")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label="Downward gravity anomaly (mGal)", shrink=0.85)
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
