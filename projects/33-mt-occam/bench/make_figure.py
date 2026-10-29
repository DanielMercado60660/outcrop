"""Plot a synthetic 1D magnetotelluric Occam inversion."""
from __future__ import annotations

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
    h = np.array([40, 80, 160, 320, 640, 1280, 2560], dtype=np.float32)
    f = np.logspace(-3, 4, 33, dtype=np.float32)
    truth = np.array([2.2, 2.1, 1.85, 1.6, 1.65, 1.9, 2.15, 2.3], dtype=np.float32)
    sigma = np.tile(np.array([[0.015], [0.01]], dtype=np.float32), (1, f.size))
    clean = call("mlx", "forward_mt", model=truth, thickness=h, frequencies=f)["response"]
    rng = np.random.default_rng(330101)
    noise = rng.normal(size=clean.shape)
    noise *= 0.8 / np.sqrt(np.mean(noise**2))
    observed = (clean + noise * sigma).astype(np.float32)
    result = call("mlx", "invert_occam", initial_model=np.full(8, 2.65, np.float32),
                  thickness=h, frequencies=f, observed=observed, sigma=sigma,
                  target_rms=1.0, max_iterations=150)
    depth = np.r_[0, np.cumsum(h), 6500] / 1000
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    ax = axes[0]
    for model, label, color in [(truth, "Synthetic truth", "#314b63"),
                                 (result["model"], "Occam recovery", "#d17c31")]:
        ax.stairs(10**model, depth, orientation="horizontal", label=label,
                  color=color, linewidth=2.2)
    ax.set_xscale("log")
    ax.set_ylim(depth[-1], 0)
    ax.set_xlabel("Resistivity (Ω m)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Layered Earth")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for ax, row, title, ylabel in [
        (axes[1], 0, "Apparent resistivity", "Apparent resistivity (Ω m)"),
        (axes[2], 1, "Impedance phase", "Phase (degrees)"),
    ]:
        if row == 0:
            data, predicted = 10**observed[row], 10**result["response"][row]
        else:
            data, predicted = np.rad2deg(observed[row]), np.rad2deg(result["response"][row])
        ax.scatter(f, data, s=14, color="#314b63", alpha=0.7, label="Synthetic data")
        ax.plot(f, predicted, color="#d17c31", linewidth=2, label="Recovered response")
        ax.set_xscale("log")
        if row == 0:
            ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="major", color="#dfe5e8", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"1D magnetotelluric Occam inversion  |  weighted RMS = {float(result['rms']):.2f}",
                 fontsize=14, fontweight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
