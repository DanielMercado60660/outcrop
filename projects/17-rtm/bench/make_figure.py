"""Plot Metal RTM of the locked synthetic layered survey."""
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
    # This golden is generated from synthetic signed sheets, not Marmousi.
    with np.load(PROJECT / "judge" / "golden" / "layers_variable.npz") as g:
        args = {name: g[name].copy() for name in ("velocity", "source_amplitudes",
                "source_locations", "receiver_locations", "data")}
        args.update(grid_spacing=g["grid_spacing"].tolist(), dt=float(g["dt"]),
                    pml_width=int(g["pml_width"]), pml_freq=float(g["pml_freq"]))
        truth = g["truth"].copy()
    image = call("metal", "rtm", **args)["image"]
    dz, dx = args["grid_spacing"]
    nz, nx = truth.shape
    extent = [-dx / 2, (nx - 0.5) * dx, (nz - 0.5) * dz, -dz / 2]
    scale = max(float(np.percentile(np.abs(image), 99)), np.finfo(float).tiny)
    normalized = image / scale

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3), constrained_layout=True)
    v = float(np.max(np.abs(truth)))
    first = axes[0].imshow(truth, extent=extent, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    second = axes[1].imshow(normalized, extent=extent, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Synthetic velocity perturbation")
    axes[1].set_title("Migrated image")
    for ax in axes:
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
    fig.colorbar(first, ax=axes[0], label="Velocity perturbation (m/s)", shrink=0.83)
    fig.colorbar(second, ax=axes[1], label="Image / 99th percentile of |image|", shrink=0.83)
    fig.suptitle("Reverse-time migration · synthetic reflectors", fontsize=15, fontweight="bold")
    fig.savefig(PROJECT / "figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
