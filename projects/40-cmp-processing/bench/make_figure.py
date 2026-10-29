"""Render synthetic CMP gather, velocity spectrum and NMO correction."""
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
    dt = 0.002
    t = np.arange(751) * dt
    offsets = np.linspace(0, 1500, 81, dtype=np.float32)
    velocities = np.linspace(1500, 3300, 121, dtype=np.float32)
    gather = np.zeros((offsets.size, t.size), dtype=np.float64)
    events = [(0.35, 1800, 1.0), (0.72, 2400, -0.8), (1.08, 3000, 0.7)]
    for t0, velocity, amplitude in events:
        arrival = np.sqrt(t0**2 + (offsets.astype(np.float64) / velocity)**2)
        u = np.pi * 25 * (t[None, :] - arrival[:, None])
        gather += amplitude * (1 - 2*u*u) * np.exp(-u*u)
    gather = gather.astype(np.float32)
    common = {"gather": gather, "offsets": offsets, "dt": dt}
    spectrum = call("metal", "semblance", **common, velocities=velocities, half_window=5)["semblance"]
    corrected = call("metal", "nmo", **common, velocity=2400.0)["corrected"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), layout="constrained")
    fig.suptitle("CMP processing  |  velocity semblance and normal moveout", fontsize=15, fontweight="bold")
    extent = [float(offsets[0]), float(offsets[-1]), t[-1], t[0]]
    amp = max(float(np.max(np.abs(gather))), float(np.max(np.abs(corrected))))
    axes[0].imshow(gather.T, aspect="auto", extent=extent, cmap="RdBu_r", vmin=-amp, vmax=amp, interpolation="nearest")
    axes[0].set_title("Synthetic CMP gather")
    axes[0].set_xlabel("Offset (m)")
    axes[0].set_ylabel("Two-way time (s)")
    im = axes[1].imshow(spectrum.T, aspect="auto", extent=[velocities[0], velocities[-1], t[-1], t[0]],
                        cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[1].scatter([e[1] for e in events], [e[0] for e in events], marker="+", c="cyan", s=65, linewidths=1.5, label="True velocities")
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.85)
    axes[1].set_title("Velocity semblance")
    axes[1].set_xlabel("Trial velocity (m/s)")
    axes[1].set_ylabel("Two-way time (s)")
    fig.colorbar(im, ax=axes[1], label="Semblance", shrink=0.84)
    axes[2].imshow(corrected.T, aspect="auto", extent=extent, cmap="RdBu_r", vmin=-amp, vmax=amp, interpolation="nearest")
    axes[2].set_title("NMO corrected at 2400 m/s")
    axes[2].set_xlabel("Offset (m)")
    axes[2].set_ylabel("Zero-offset time (s)")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
