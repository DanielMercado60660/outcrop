"""Render surface deformation from a buried rectangular Okada fault."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call


def main():
    axis = np.linspace(-12, 12, 301, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    coordinates = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size, np.float32)))
    result = call("metal", "okada", coordinates=coordinates, source="finite_rectangle",
                  depth=6.0, dip_deg=50.0,
                  bounds=np.array([-3, 3, -2, 2], dtype=np.float32),
                  slip=np.array([1.0, 0.6, 0.2], dtype=np.float32), alpha=2.0 / 3.0)
    vertical = result["displacement"][:, 2].reshape(x.shape) * 1000.0
    limit = float(np.max(np.abs(vertical)))
    fig, ax = plt.subplots(figsize=(7.6, 6.3), layout="constrained")
    im = ax.pcolormesh(x, y, vertical, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    # Horizontal projection of the buried fault plane, shown for orientation.
    half_width = 2 * np.cos(np.deg2rad(50.0))
    ax.plot([-3, 3, 3, -3, -3], [-half_width, -half_width, half_width,
                                  half_width, -half_width],
            color="black", linewidth=1.0, linestyle="--", alpha=0.8,
            label="Buried fault projection")
    ax.set(xlabel="Along strike x (km)", ylabel="Across strike y (km)",
           title="Okada rectangular fault · surface vertical displacement", aspect="equal")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.colorbar(im, ax=ax, label="Vertical displacement (mm)", shrink=0.9, pad=0.03)
    fig.savefig(ROOT / "figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
