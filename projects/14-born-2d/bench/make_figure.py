"""Make a synthetic Born-scattering snapshot with the Metal implementation."""
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
    ny, nx, nt = 160, 200, 400
    dx, dt = 10.0, 0.001
    y, x = np.mgrid[:ny, :nx]
    v = (1500 + 0.20 * y * dx + 60 * np.sin(x / 45)).astype(np.float32)
    dv = (100 * np.exp(-((y - 73) / 12) ** 2 - ((x - 78) / 19) ** 2)
          - 85 * np.exp(-((y - 91) / 10) ** 2 - ((x - 131) / 16) ** 2)).astype(np.float32)
    t = np.arange(nt) * dt - 0.06
    q = (np.pi * 15 * t) ** 2
    pulse = ((1 - 2 * q) * np.exp(-q)).astype(np.float32)[None, None, :]
    receivers = np.stack((y.ravel(), x.ravel()), axis=-1)[None].astype(np.int64)
    zero = np.zeros((1, ny, nx), dtype=np.float32)
    result = call("metal", "born", v=v, dv=dv, source_amplitudes=pulse,
                  source_locations=np.array([[[25, 100]]], dtype=np.int64),
                  receiver_locations=receivers, wavefield_0=zero.copy(),
                  wavefield_m1=zero.copy(), wavefield_sc_0=zero.copy(),
                  wavefield_sc_m1=zero.copy(), dx=dx, dt=dt, accuracy=4,
                  boundary="pml", pml_width=10, pml_freq=15.0, max_vel=1900.0)
    snapshot = result["scattered"][0, :, -1].reshape(ny, nx)
    limit = float(np.max(np.abs(snapshot))) or 1.0
    extent = (0, (nx - 1) * dx / 1000, (ny - 1) * dx / 1000, 0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    image0 = axes[0].imshow(dv, extent=extent, cmap="RdBu_r", vmin=-100, vmax=100, aspect="equal")
    image1 = axes[1].imshow(snapshot, extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="equal")
    for ax in axes:
        ax.plot(1.0, 0.25, marker="*", color="#202020", markersize=10)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Depth (km)")
    axes[0].set_title("Velocity perturbation")
    axes[1].set_title(f"Scattered wavefield at {(nt - 1) * dt:.3f} s")
    fig.colorbar(image0, ax=axes[0], label="Velocity perturbation (m/s)", shrink=0.8)
    fig.colorbar(image1, ax=axes[1], label="Wavefield amplitude (arbitrary units)", shrink=0.8)
    fig.suptitle("2-D acoustic Born scattering", fontsize=16, fontweight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
