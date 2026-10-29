"""Plot a synthetic fault-slip recovery using the MLX inversion."""

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
    rng = np.random.default_rng(450102)
    nx, nz = 12, 8
    x = np.linspace(-11, 11, nx)
    depth = np.linspace(1, 15, nz)
    xx, zz = np.meshgrid(x, depth)
    truth = (1.45 * np.exp(-((xx + 3.5) / 4.0) ** 2 - ((zz - 5) / 3.2) ** 2)
             + 0.95 * np.exp(-((xx - 5) / 3.0) ** 2 - ((zz - 11) / 3.6) ** 2))

    # Smooth synthetic sensitivity functions; these are not an Okada Green matrix.
    ox, oz = np.meshgrid(np.linspace(-13, 13, 24), np.linspace(0, 16, 18))
    dx = ox.ravel()[:, None] - xx.ravel()[None, :]
    dz = oz.ravel()[:, None] - zz.ravel()[None, :]
    green = (0.006 * np.exp(-(dx / 2.6) ** 2 - (dz / 2.8) ** 2)).astype(np.float32)
    sigma = np.full(green.shape[0], 0.0005, dtype=np.float32)
    data = (green @ truth.ravel() + rng.normal(0, sigma)).astype(np.float32)
    edges = []
    for iz in range(nz):
        for ix in range(nx):
            a = iz * nx + ix
            for jz, jx in ((iz, ix + 1), (iz + 1, ix)):
                if jz < nz and jx < nx:
                    row = np.zeros(nx * nz, dtype=np.float32)
                    row[a], row[jz * nx + jx] = -1, 1
                    edges.append(row)
    regularizer = np.stack(edges)
    result = call("mlx", "invert_slip", green=green, data=data, sigma=sigma,
                  regularizer=regularizer, damping=2.0,
                  areas=np.full(nx * nz, 4e6, dtype=np.float32), shear_modulus=30e9)
    recovered = result["slip"].reshape(nz, nx)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained", sharey=True)
    vmax = max(float(np.max(truth)), float(np.max(recovered)))
    extent = [x[0] - 1, x[-1] + 1, depth[-1] + 1, depth[0] - 1]
    for ax, values, title in zip(axes, (truth, recovered), ("Synthetic truth", "Recovered slip"), strict=True):
        im = ax.imshow(values, origin="upper", extent=extent, cmap="viridis", vmin=0,
                       vmax=vmax, interpolation="nearest", aspect="equal")
        ax.set_title(title, weight="medium")
        ax.set_xlabel("Along strike (km)")
        ax.tick_params(length=3)
    axes[0].set_ylabel("Down-dip distance (km)")
    fig.colorbar(im, ax=axes, label="Slip (m)", shrink=0.82, pad=0.025)
    fig.suptitle("Regularized finite-fault slip inversion · MLX", weight="bold", fontsize=14)
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
