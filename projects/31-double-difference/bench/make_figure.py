"""Plot a synthetic fault catalog before and after double-difference relocation."""
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
    rng = np.random.default_rng(3102)
    n = 32
    along = np.linspace(-3, 3, n)
    truth = np.column_stack((along, np.zeros(n), 5 + 0.25 * along))
    origins = rng.uniform(-0.4, 0.4, n)
    angles = np.arange(8) * 2 * np.pi / 8 + 0.15
    stations = np.column_stack((9 * np.cos(angles), 9 * np.sin(angles), np.zeros(8)))
    initial = truth + rng.normal(0, [0.18, 0.8, 0.15], (n, 3))
    initial_origins = origins + rng.normal(0, 0.035, n)
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    arrivals = origins[:, None] + np.linalg.norm(truth[:, None] - stations[None], axis=2) / 6.0
    observed = arrivals[pairs[:, 0]] - arrivals[pairs[:, 1]]
    observed += rng.normal(0, 0.002, observed.shape)
    result = call("mlx", "relocate", stations=stations.astype(np.float32),
                  initial_locations=initial.astype(np.float32),
                  initial_origins=initial_origins.astype(np.float32), pairs=pairs,
                  observed=observed.astype(np.float32), velocity=6.0, damping=1e-4)
    relocated = result["locations"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True, sharey=True)
    for ax, data, label in zip(axes, (initial, relocated), ("Initial catalog", "Relocated catalog")):
        ax.axhline(0, color="#78818a", linestyle="--", linewidth=1.2, label="True fault")
        points = ax.scatter(data[:, 0], data[:, 1], c=truth[:, 2], cmap="viridis",
                            vmin=4.25, vmax=5.75, s=48, edgecolor="white", linewidth=0.5)
        ax.set_title(label, loc="left", weight="bold")
        ax.set_xlabel("Along-fault position x (km)")
        ax.set_xlim(-3.8, 3.8)
        ax.set_ylim(-2.6, 2.6)
        ax.grid(color="#dce2e6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.text(0.03, 0.05, f"Cross-fault spread: {np.std(data[:, 1]):.2f} km",
                transform=ax.transAxes, fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    axes[0].set_ylabel("Cross-fault position y (km)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    bar = fig.colorbar(points, ax=axes, shrink=0.82, pad=0.03)
    bar.set_label("True event depth (km)")
    fig.suptitle("Double-difference earthquake relocation", fontsize=15, weight="bold")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
