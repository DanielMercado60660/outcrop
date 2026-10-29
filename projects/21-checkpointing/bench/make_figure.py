"""Render a synthetic velocity model and its Metal checkpointed gradient."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_bench import call, make_inputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    inputs = make_inputs(nz=96, nx=128, nt=192, nshot=1, nreceiver=32, checkpoint_budget=7)
    output = call("metal", "misfit_and_grad", **inputs)
    velocity = inputs["velocity"]
    gradient = output["gradient"]
    extent = [0, (velocity.shape[1] - 1) * 10, (velocity.shape[0] - 1) * 8, 0]
    limit = max(float(np.max(np.abs(gradient))), np.finfo(np.float32).tiny)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    im = axes[0].imshow(velocity, extent=extent, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=axes[0], label="Velocity (m/s)", shrink=0.8)
    axes[0].set_title("Synthetic velocity")
    im = axes[1].imshow(gradient, extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="equal")
    fig.colorbar(im, ax=axes[1], label="Misfit gradient (U² / (m/s))", shrink=0.8, format="%.1e")
    axes[1].set_title("Checkpointed velocity gradient")
    for ax in axes:
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        sx = inputs["source_locations"][0, 0, 1] * 10
        sz = inputs["source_locations"][0, 0, 0] * 8
        ax.plot(sx, sz, "*", color="white", markeredgecolor="black", markersize=9)
    fig.suptitle("Acoustic adjoint with Metal checkpointing", fontsize=14, fontweight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
