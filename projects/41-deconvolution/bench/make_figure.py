"""Illustrate causal spiking deconvolution on synthetic reflectivity."""

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
    rng = np.random.default_rng(4101)
    n, dt = 800, 0.002
    reflectivity = np.zeros(n, dtype=np.float32)
    indices = np.sort(rng.choice(np.arange(30, n - 30), 48, replace=False))
    reflectivity[indices] = rng.uniform(-1.0, 1.0, len(indices)).astype(np.float32)
    wavelet = np.array([1.0, 0.5, -0.2], dtype=np.float32)
    trace = np.convolve(reflectivity, wavelet)[:n].astype(np.float32)
    out = call("metal", "spiking_deconvolution", trace=trace, wavelet=wavelet,
               inverse_length=32, stabilization=1e-6)
    t = np.arange(n) * dt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.8), sharex=True,
                             constrained_layout=True)
    for ax, data, label, color in zip(
        axes, (reflectivity, trace, out["recovered"]),
        ("True reflectivity", "Observed trace", "Recovered reflectivity"),
        ("#3d5267", "#d17a39", "#007f83"), strict=True,
    ):
        ax.plot(t, data, color=color, linewidth=1.15)
        ax.axhline(0, color="#aeb9bf", linewidth=0.6)
        ax.set_ylabel("Amplitude\n(dimensionless)")
        ax.text(0.015, 0.86, label, transform=ax.transAxes, fontweight="bold",
                color=color, va="top")
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-1.55, 1.55)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Causal spiking deconvolution", fontsize=16, fontweight="bold")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
