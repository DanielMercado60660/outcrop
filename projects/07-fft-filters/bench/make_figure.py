"""Plot a synthetic magnetic field and its upward continuation."""

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
    n = 512
    spacing = 50.0
    coords = (np.arange(n) - n / 2) * spacing / 1000.0
    x, y = np.meshgrid(coords, coords)
    # Localized features decay toward the edges of the periodic FFT grid.
    grid = (70 * np.exp(-((x + 3.3) ** 2 / 2.2 + (y - 1.3) ** 2 / 4.0))
            - 55 * np.exp(-((x + 1.6) ** 2 / 2.8 + (y - 0.3) ** 2 / 3.0))
            + 45 * np.exp(-((x - 3.6) ** 2 / 4.5 + (y + 2.9) ** 2 / 1.5))
            - 30 * np.exp(-((x - 5.3) ** 2 / 2.0 + (y + 4.0) ** 2 / 2.0))
            + 25 * np.exp(-((x + 4.3) ** 2 / 5.0 + (y + 5.0) ** 2 / 2.0)))
    grid = grid.astype(np.float32)
    height = 1000.0
    continued = call("mlx", "filter_grid", grid=grid, dx=spacing, dy=spacing,
                     operation="upward", height=height, inclination=50.0,
                     declination=-23.0, magnetization_inclination=50.0,
                     magnetization_declination=-23.0)["result"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    limit = float(np.max(np.abs(grid)))
    extent = [coords[0] - spacing / 2000, coords[-1] + spacing / 2000,
              coords[0] - spacing / 2000, coords[-1] + spacing / 2000]
    for ax, data, title in zip(axes, (grid, continued),
                               ("Observed field", "Continued upward 1 km")):
        image = ax.imshow(data, origin="lower", extent=extent, cmap="RdBu_r",
                          vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set(title=title, xlabel="Easting (km)", ylabel="Northing (km)")
        ax.set_aspect("equal")
    fig.colorbar(image, ax=axes, label="Magnetic anomaly (nT)", shrink=0.86, pad=0.03)
    fig.suptitle("Fourier upward continuation of a synthetic magnetic anomaly",
                 fontsize=14, fontweight="semibold")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
