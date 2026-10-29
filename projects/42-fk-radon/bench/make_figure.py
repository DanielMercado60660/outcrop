"""Make a synthetic parabolic Radon demultiple illustration using Metal."""

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


def ricker(t, frequency):
    q = (np.pi * frequency * t) ** 2
    return (1 - 2 * q) * np.exp(-q)


def main():
    nx, nt, dt = 128, 1024, 0.002
    offsets = np.linspace(-600, 600, nx, dtype=np.float32)
    times = np.arange(nt) * dt
    x2 = offsets[:, None].astype(np.float64) ** 2
    slow = 1.0e-6
    primary = ricker(times[None, :] - (0.62 - 0.42e-6 * x2), 22)
    primary += 0.7 * ricker(times[None, :] - (1.05 + 0.25e-6 * x2), 19)
    multiple = 1.25 * ricker(times[None, :] - (1.38 + slow * x2), 18)
    gather = (primary + multiple).astype(np.float32)
    filtered = call("metal", "radon_demultiple", gather=gather, offsets=offsets,
                    dt=dt, slowness=slow)["filtered"]
    removed = gather - filtered

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "figure.facecolor": "#f8f9fb"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True,
                             layout="constrained")
    limit = np.percentile(np.abs(gather), 99.5)
    extent = [float(offsets[0]), float(offsets[-1]), times[-1], times[0]]
    for ax, data, title in zip(axes, (gather, filtered, removed),
                               ("Input gather", "After demultiple", "Removed component")):
        im = ax.imshow(data.T, extent=extent, aspect="auto", cmap="RdBu_r",
                       vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set(title=title, xlabel="Offset (m)")
        ax.set_facecolor("white")
    axes[0].set_ylabel("Time (s)")
    fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02, label="Amplitude")
    fig.suptitle("Parabolic Radon demultiple | synthetic seismic gather",
                 fontsize=15, fontweight="bold")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
