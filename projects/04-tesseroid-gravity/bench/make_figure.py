"""Render a synthetic tesseroid gravity anomaly with Metal."""
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
    longitude = np.linspace(-4, 4, 120, dtype=np.float32)
    latitude = np.linspace(-3, 3, 90, dtype=np.float32)
    lon, lat = np.meshgrid(longitude, latitude)
    coordinates = np.column_stack((lon.ravel(), lat.ravel(),
                                   np.full(lon.size, 6_391_000, dtype=np.float32)))
    sources = np.array([[-2.1, -0.6, -0.8, 0.8, 6_351_000, 6_371_000],
                        [0.6, 2.0, -0.5, 1.0, 6_351_000, 6_371_000]],
                       dtype=np.float32)
    density = np.array([2670, -1800], dtype=np.float32)
    field = call("metal", "tesseroid_gravity", coordinates=coordinates,
                 tesseroids=sources, density=density, field="g_z",
                 radial_adaptive=True)["field"].reshape(lon.shape)
    limit = float(np.max(np.abs(field)))
    fig, ax = plt.subplots(figsize=(8.2, 5.5), constrained_layout=True)
    image = ax.pcolormesh(lon, lat, field, cmap="RdBu_r", shading="auto",
                          vmin=-limit, vmax=limit, rasterized=True)
    for source in sources:
        ax.plot([source[0], source[1], source[1], source[0], source[0]],
                [source[2], source[2], source[3], source[3], source[2]],
                color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
           title="Tesseroid gravity · adaptive Gauss–Legendre quadrature")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label="Downward gravity anomaly (mGal)", shrink=0.88)
    fig.text(0.5, 0.01, "Dashed outlines: synthetic density contrasts · observation altitude: 20 km",
             ha="center", fontsize=9, color="0.35")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
