"""Illustrate cycle skipping in synthetic arrival-time inversion."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call


def wavelet(time, delay):
    tau = time - 0.512 - delay
    return (np.exp(-0.5 * (tau / 0.05)**2) * np.cos(2 * np.pi * 10 * tau)).astype(np.float32)


def main():
    dt = 0.002
    time = np.arange(512) * dt
    observed = wavelet(time, 0)[None, :]
    delays = np.linspace(-0.18, 0.18, 101)
    curves = {"l2": [], "envelope": []}
    for delay in delays:
        predicted = wavelet(time, delay)[None, :]
        for kind, values in curves.items():
            out = call("mlx", "misfit_and_grad", predicted=predicted, observed=observed,
                       dt=dt, kind=kind, epsilon=0.01)
            values.append(float(out["objective"]))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f9fafb"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    fig.suptitle("Waveform and envelope misfits | MLX", fontsize=16, fontweight="bold")
    axes[0].plot(time, observed[0], color="#23395b", lw=2, label="Observed")
    axes[0].plot(time, wavelet(time, 0.08), color="#db6b35", lw=1.8,
                 label="Predicted (+80 ms)")
    axes[0].set(xlim=(0.25, 0.82), xlabel="Time (s)", ylabel="Amplitude (arbitrary units)",
                title="Synthetic 10 Hz arrivals")
    axes[0].legend(frameon=False, loc="upper right")
    for kind, color, label in [("l2", "#23395b", "Waveform L2"),
                               ("envelope", "#d45b2c", "Envelope")]:
        axes[1].plot(delays * 1000, curves[kind], color=color, lw=2.3, label=label)
    axes[1].axvline(0, color="#8c949e", lw=1, ls="--")
    axes[1].set(xlabel="Predicted arrival delay (ms)",
                ylabel="Misfit (amplitude² s)", title="Objective landscape")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(alpha=0.16)
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
