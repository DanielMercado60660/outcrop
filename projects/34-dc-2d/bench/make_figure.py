"""Draw a synthetic resistivity model and its Metal Wenner response."""
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
    nx, nz = 101, 48
    hx = np.full(nx, 0.5, dtype=np.float32)
    hz = np.full(nz, 0.5, dtype=np.float32)
    origin = np.array([-nx * 0.25, -nz * 0.5], dtype=np.float32)
    x = origin[0] + (np.arange(nx) + 0.5) * 0.5
    z = origin[1] + (np.arange(nz) + 0.5) * 0.5
    rho = np.full((nx, nz), 100.0, dtype=np.float32)
    rho[(np.abs(x[:, None] - 1.5) < 3.0) & (z[None, :] < -1.5) & (z[None, :] > -6.5)] = 25.0
    centers, spacing = np.meshgrid(np.linspace(-10, 10, 11), np.array([1., 1.5, 2., 2.5, 3.]))
    c, a = centers.ravel(), spacing.ravel()
    ab = np.column_stack((c - 1.5 * a, c + 1.5 * a)).astype(np.float32)
    mn = np.column_stack((c - 0.5 * a, c + 0.5 * a)).astype(np.float32)
    ky = np.logspace(-4, 1, 21)
    delta = np.diff(ky) / 2
    weights = np.r_[delta, 0] + np.r_[0, delta]
    weights[0] += ky[0]
    result = call("metal", "dc_forward", hx=hx, hz=hz, origin=origin,
                  conductivity=1 / rho, ab=ab, mn=mn,
                  currents=np.ones(len(c), dtype=np.float32),
                  ky=ky.astype(np.float32), weights=(weights / np.pi).astype(np.float32))
    apparent = result["apparent_resistivity"].reshape(centers.shape)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    fig.suptitle("2.5D DC resistivity · Metal forward modelling", fontsize=15, fontweight="bold")
    model = axes[0].pcolormesh(x, z, rho.T, shading="nearest", cmap="viridis", vmin=20, vmax=110)
    axes[0].set(xlim=(-15, 15), ylim=(-12, 0), xlabel="Distance (m)", ylabel="Elevation (m)",
                title="Synthetic subsurface resistivity")
    fig.colorbar(model, ax=axes[0], label="Resistivity (Ω m)", pad=0.02)
    response = axes[1].pcolormesh(centers, spacing, apparent, shading="nearest", cmap="viridis")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Array midpoint (m)", ylabel="Electrode spacing (m)",
                title="Wenner apparent resistivity")
    fig.colorbar(response, ax=axes[1], label="Apparent resistivity (Ω m)", pad=0.02)
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
