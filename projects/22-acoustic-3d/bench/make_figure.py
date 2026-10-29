"""Render a synthetic 3D acoustic wavefield computed by Metal."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call


def main():
    n, spacing, dt, steps = 160, 10.0, 0.0015, 150
    coords = np.arange(n, dtype=np.float32) * spacing
    z, y, x = np.ogrid[:n, :n, :n]
    center = (n - 1) / 2
    radius2 = ((x - center)**2 + (y - center)**2 + (z - center)**2)
    initial = np.exp(-radius2 / (2 * 3.0**2)).astype(np.float32)
    inclusion_r2 = ((x - (center + 27))**2 + (y - center)**2 + (z - center)**2)
    velocity = (2000 + 450 * np.exp(-inclusion_r2 / (2 * 16.0**2))).astype(np.float32)
    out = call("metal", "propagate", velocity=velocity, initial=initial,
               previous=initial.copy(), spacing=spacing, dt=dt, steps=steps)
    field = out["final"][n // 2]
    vmax = float(np.max(np.abs(field)))
    extent = [coords[0], coords[-1], coords[-1], coords[0]]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), layout="constrained")
    im = ax.imshow(field, extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="bilinear")
    ax.contour(coords, coords, velocity[n // 2], levels=[2100, 2300],
               colors="#333333", linewidths=0.8, alpha=0.65)
    ax.plot(center * spacing, center * spacing, "+", color="black", markersize=8,
            markeredgewidth=1.2)
    ax.set(xlabel="x (m)", ylabel="y (m)",
           title=f"3D acoustic propagation · Metal\nHorizontal slice, t = {steps * dt:.3f} s")
    cb = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.025)
    cb.set_label("Pressure (arbitrary units)")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
