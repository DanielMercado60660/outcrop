"""Illustrate a compact neural-prior acoustic inversion on synthetic data."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call, golden


def main():
    g = golden("noise_a")
    keys = ("latent", "initial_velocity", "observed", "source_amplitudes",
            "source_locations", "receiver_locations")
    kwargs = {key: g[key].copy() for key in keys}
    kwargs.update(parameters=g["neural_initial_parameters"].copy(),
                  parameterization="neural", spacing=float(g["spacing"]),
                  dt=float(g["dt"]), data_scale=float(g["data_scale"]),
                  velocity_scale=float(g["velocity_scale"]),
                  coarse_shape=g["coarse_shape"].tolist(), hidden_size=int(g["hidden_size"]),
                  steps=40, learning_rate=float(g["learning_rate"]))
    result = call("mlx", "invert", **kwargs)
    initial = g["initial_velocity"]
    truth = g["truth"]
    recovered = result["velocity"]
    nz, nx = initial.shape
    dx = float(g["spacing"])
    extent = [-dx / 2, (nx - 0.5) * dx, (nz - 0.5) * dx, -dx / 2]
    limit = 400
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained", sharex=True, sharey=True)
    for ax, model, title in zip(axes, (truth - initial, recovered - initial, recovered - truth),
                                ("Synthetic truth", "Neural recovery · 40 steps", "Recovery − truth")):
        im = ax.imshow(model, extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit,
                       interpolation="bilinear", aspect="equal")
        ax.set_title(title, fontsize=11, weight="semibold")
        ax.set_xlabel("Distance x (m)")
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Depth z (m)")
    fig.colorbar(im, ax=axes, label="Velocity perturbation / error (m/s)", shrink=0.8, pad=0.02)
    fig.suptitle("Neural-prior full-waveform inversion", fontsize=15, weight="bold")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
