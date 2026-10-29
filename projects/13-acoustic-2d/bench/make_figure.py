"""Render synthetic acoustic wavefield snapshots with the Metal solver."""
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
    nz, nx, nt = 150, 200, 450
    spacing, dt = 10.0, 0.001
    z, x = np.indices((nz, nx))
    interface = 82 + 12 * np.sin(x / 32)
    velocity = (1550 + 450 * (z > interface)
                + 160 * np.exp(-((x - 140)**2 / 500 + (z - 105)**2 / 180)))
    velocity = velocity.astype(np.float32)
    t = np.arange(nt) * dt
    q = np.pi * 16 * (t - 0.06)
    source = (((1 - 2 * q**2) * np.exp(-q**2)) / spacing**2).astype(np.float32)[None, None, :]
    source_locations = np.array([[[30, 100]]], dtype=np.int64)
    receiver_locations = np.stack((z.ravel(), x.ravel()), axis=-1)[None].astype(np.int64)
    traces = call("metal", "scalar", velocity=velocity, source=source,
                  source_locations=source_locations, receiver_locations=receiver_locations,
                  spacing=spacing, dt=dt, order=4, pml_width=[12] * 4,
                  pml_freq=16.0, max_vel=2200.0)["receiver_amplitudes"]
    samples = [180, 300, 420]
    snapshots = traces[0, :, samples].reshape(len(samples), nz, nx)
    limit = np.max(np.abs(snapshots))
    extent = [0, (nx - 1) * spacing / 1000, (nz - 1) * spacing / 1000, 0]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, snapshot, sample in zip(axes, snapshots, samples):
        image = ax.imshow(snapshot, extent=extent, cmap="RdBu_r", vmin=-limit,
                          vmax=limit, interpolation="bilinear", aspect="equal")
        ax.contour(x * spacing / 1000, z * spacing / 1000, velocity,
                   levels=[1775], colors="black", linewidths=0.8, alpha=0.55)
        ax.scatter([1.0], [0.3], marker="*", s=45, color="#f4c95d", edgecolors="black",
                   linewidths=0.5, zorder=3)
        ax.set_title(f"{sample * dt:.2f} s", fontsize=11, weight="bold")
        ax.set_xlabel("Distance (km)")
    axes[0].set_ylabel("Depth (km)")
    fig.colorbar(image, ax=axes, label="Wavefield amplitude (u)", shrink=0.74, pad=0.02)
    fig.suptitle("2D acoustic propagation · finite differences with PML", fontsize=15, weight="bold")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
