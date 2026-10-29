"""Plot a synthetic layered-earth model and its Metal MT response."""
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
    rho = np.array([150, 12, 600, 80], dtype=np.float32)
    thickness = np.array([250, 800, 1700], dtype=np.float32)
    frequency = np.logspace(-3, 3, 500, dtype=np.float32)
    response = call("metal", "mt1d", resistivity=rho, thickness=thickness,
                    frequency=frequency)
    depths = np.r_[0, np.cumsum(thickness), 6000] / 1000
    colors = ["#f4c96b", "#39a8a0", "#d77b57", "#687fc3"]
    fig, (model, apparent, phase) = plt.subplots(
        1, 3, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25, 1.25]},
        layout="constrained")
    fig.patch.set_facecolor("#f9fafb")
    for i, value in enumerate(rho):
        model.fill_betweenx([depths[i], depths[i + 1]], 1, value, color=colors[i], alpha=0.8)
        model.plot([value, value], [depths[i], depths[i + 1]], color="#263647", lw=2)
        if i < len(rho) - 1:
            model.plot([value, rho[i + 1]], [depths[i + 1]] * 2, color="#263647", lw=2)
    model.set(xscale="log", xlim=(8, 1200), ylim=(6, 0), xlabel="Resistivity (Ω m)",
              ylabel="Depth (km)", title="Layered earth")
    apparent.loglog(frequency, response["apparent_resistivity"], color="#087e83", lw=2.5)
    apparent.set(xlabel="Frequency (Hz)", ylabel="Apparent resistivity (Ω m)",
                 title="Surface response")
    phase.semilogx(frequency, response["phase"], color="#d06247", lw=2.5)
    phase.axhline(45, color="#8c97a2", lw=1, ls="--")
    phase.set(xlabel="Frequency (Hz)", ylabel="Phase (degrees)", title="Impedance phase")
    for ax in (model, apparent, phase):
        ax.grid(True, which="major", color="#d9e0e5", lw=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("1D magnetotellurics · layered-earth recursion", fontsize=16, fontweight="bold")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
