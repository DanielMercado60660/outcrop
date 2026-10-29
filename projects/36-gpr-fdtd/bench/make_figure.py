"""Render a synthetic GPR-style Yee electric-field snapshot using Metal."""
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
    n, spacing, steps = 512, 0.025, 430
    c = 299792458.0
    dt = 0.4 * spacing / c
    x = np.arange(n) * spacing
    z = np.arange(n) * spacing
    xx, zz = np.meshgrid(x, z, indexing="ij")
    target = ((xx - 6.4) / 0.75) ** 2 + ((zz - 7.0) / 0.48) ** 2 <= 1
    eps = np.where(zz >= 5.5, 4.0, 1.0)
    eps[target] = 9.0
    radius2 = (xx - 6.4) ** 2 + (zz - 2.7) ** 2
    electric = np.exp(-radius2 / (0.22 ** 2)).astype(np.float32)
    zeros = np.zeros((n, n), dtype=np.float32)
    fields = call("metal", "propagate", epsilon_r=eps.astype(np.float32),
                  electric=electric, hx=zeros, hz=zeros, dx=spacing, dz=spacing,
                  dt=dt, steps=steps)
    field = fields["electric"].T
    limit = np.percentile(np.abs(field), 99.7)
    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    image = ax.imshow(field, extent=(x[0], x[-1], z[-1], z[0]),
                      cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.contour(xx, zz, eps, levels=[2.5, 6.5], colors="0.2", linewidths=0.8,
               linestyles="--", alpha=0.7)
    ax.plot(6.4, 2.7, marker="*", color="#e9b949", markeredgecolor="black",
            markersize=10, label="Initial pulse")
    ax.set(xlabel="Horizontal distance x (m)", ylabel="Depth z (m)",
           title=f"2-D Yee FDTD: electric field at {steps * dt * 1e9:.1f} ns")
    ax.legend(loc="lower right", framealpha=0.9)
    bar = fig.colorbar(image, ax=ax, pad=0.025, shrink=0.9, extend="both")
    bar.set_label("Electric field $E_y$ (V/m)")
    fig.savefig(PROJECT / "figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
