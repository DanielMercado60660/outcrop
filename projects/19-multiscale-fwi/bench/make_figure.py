"""Plot a synthetic transmission survey and its multiscale waveforms."""
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


def lowpass(trace, dt, cutoff):
    n = trace.size
    frequency = np.fft.rfftfreq(n, dt)
    weight = 1 / np.sqrt(1 + (frequency / cutoff) ** 8)
    return np.fft.irfft(np.fft.rfft(trace) * weight, n=n)


def main():
    # The locked fixture is an entirely synthetic two-region model.
    with np.load(ROOT / "judge" / "golden" / "case_0.npz") as data:
        g = {key: data[key] for key in data.files}
    args = {key: g[key] for key in ("basis", "source", "sources", "receivers", "observed")}
    args.update(parameters=g["initial"].astype(np.float32), dx=float(g["dx"]),
                dt=float(g["dt"]), cutoff=9.0)
    predicted = call("mlx", "misfit_and_grad", **args)["predicted"]
    truth = np.einsum("p,pzx->zx", g["truth"], g["basis"])
    dt, dx = float(g["dt"]), float(g["dx"])
    time = np.arange(g["source"].shape[-1]) * dt
    # One long-offset receiver makes the initial cycle mismatch visible.
    shot, receiver = 0, 3
    observed = g["observed"][shot, receiver].astype(np.float64)
    initial = predicted[shot, receiver].astype(np.float64)
    scale = max(np.max(np.abs(observed)), np.max(np.abs(initial)))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig = plt.figure(figsize=(10.5, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.3])
    ax = fig.add_subplot(grid[:, 0])
    extent = [-dx / 2, (truth.shape[1] - 0.5) * dx,
              (truth.shape[0] - 0.5) * dx, -dx / 2]
    image = ax.imshow(truth, extent=extent, cmap="viridis", aspect="equal",
                      vmin=1700, vmax=2700)
    src = g["sources"][shot, 0]
    rec = g["receivers"][shot]
    ax.scatter(src[1] * dx, src[0] * dx, marker="*", s=150, c="#ffcc42",
               edgecolors="#20252b", linewidths=0.7, label="Source", zorder=3)
    ax.scatter(rec[:, 1] * dx, rec[:, 0] * dx, marker="v", s=48, c="white",
               edgecolors="#20252b", linewidths=0.7, label="Receivers", zorder=3)
    ax.scatter(rec[receiver, 1] * dx, rec[receiver, 0] * dx, marker="v", s=75,
               facecolors="none", edgecolors="#f15b50", linewidths=2, zorder=4)
    ax.set(xlabel="Distance (m)", ylabel="Depth (m)", title="Synthetic true velocity")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.1, shrink=0.9,
                 label="Velocity (m/s)")
    for row, cutoff in enumerate((3.0, 18.0)):
        trace_ax = fig.add_subplot(grid[row, 1])
        trace_ax.plot(time, lowpass(observed, dt, cutoff) / scale,
                      color="#1b7390", lw=2, label="Observed (true model)")
        trace_ax.plot(time, lowpass(initial, dt, cutoff) / scale,
                      color="#d45446", lw=1.7, ls="--", label="Initial model")
        trace_ax.axhline(0, color="#89949b", lw=0.6)
        trace_ax.set(xlim=(0, time[-1]), xlabel="Time (s)",
                     ylabel="Amplitude / peak", title=f"{cutoff:g} Hz cutoff")
        trace_ax.grid(alpha=0.18)
        if row == 0:
            trace_ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Multiscale full-waveform inversion | transmission survey", fontsize=15,
                 fontweight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
