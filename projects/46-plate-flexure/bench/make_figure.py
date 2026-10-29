"""Plot synthetic continuous- and broken-plate flexure using Metal."""

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
    distance = np.linspace(0, 600_000, 2001, dtype=np.float32)
    args = {"x": distance, "rigidity": 1e23, "foundation": 32000.0,
            "load": 1e12}
    continuous = call("metal", "plate_flexure", **args, plate="continuous")
    broken = call("metal", "plate_flexure", **args, plate="broken")
    x = distance / 1000

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    ax.axhline(0, color="#7b8794", lw=0.9)
    for data, label, color in (
        (continuous["deflection"], "Continuous plate", "#137c8b"),
        (broken["deflection"], "Broken plate", "#ce6938"),
    ):
        ax.plot(x, data, label=label, color=color, lw=2.7)
        ax.fill_between(x, 0, data, color=color, alpha=0.09)
    ax.invert_yaxis()
    ax.set(xlabel="Distance from line load (km)",
           ylabel="Deflection (m; positive downward)",
           title="Elastic plate flexure on a Winkler foundation · Metal")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, 0.04, r"$D=10^{23}$ N m    $k=3.2\times10^4$ N m$^{-3}$"
            "\n" r"$P=10^{12}$ N m$^{-1}$", transform=ax.transAxes,
            va="bottom", fontsize=9, color="#45515c")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
