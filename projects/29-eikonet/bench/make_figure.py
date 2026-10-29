"""Train a synthetic EikoNet and plot its travel-time field."""
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


def speed(points):
    u = points / 2000.0
    lens = np.exp(-np.sum(((u - np.array([0.60, 0.45])) / 0.22) ** 2, axis=-1))
    return 2600.0 + 1000.0 * u[..., 1] - 550.0 * lens


def main():
    rng = np.random.default_rng(29012020)
    width = 32
    source = np.array([500.0, 500.0], dtype=np.float32)
    receivers = rng.uniform(40, 1960, (1536, 2)).astype(np.float32)
    sources = np.repeat(source[None, :], len(receivers), axis=0)
    p = []
    for shape in ((4, width), (width,), (width, width), (width,), (width,), (1,)):
        p.append(rng.normal(0, 1 / np.sqrt(shape[0]) if len(shape) == 2 else 0.12, shape).ravel())
    initial = np.concatenate(p).astype(np.float32)
    initial[-width - 1:] = 0.0
    common = {"origin": np.zeros(2, dtype=np.float32), "length_scale": 2000.0,
              "reference_velocity": 3000.0, "width": width}
    trained = call("mlx", "train", initial_parameters=initial, sources=sources,
                   receivers=receivers, velocities=speed(receivers).astype(np.float32),
                   steps=1000, learning_rate=0.001, **common)["parameters"]
    axis = np.linspace(0, 2000, 201, dtype=np.float32)
    x, z = np.meshgrid(axis, axis)
    grid = np.column_stack((x.ravel(), z.ravel())).astype(np.float32)
    out = call("mlx", "evaluate", parameters=trained,
               sources=np.repeat(source[None, :], len(grid), axis=0),
               receivers=grid, **common)
    travel = out["time"].reshape(x.shape)
    velocity = speed(grid).reshape(x.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for ax, field, cmap, label in ((axes[0], velocity, "viridis", "Velocity (m/s)"),
                                    (axes[1], travel, "magma", "Travel time (s)")):
        image = ax.imshow(field, extent=(0, 2, 2, 0), cmap=cmap, aspect="equal")
        fig.colorbar(image, ax=ax, label=label, shrink=0.85)
        ax.plot(source[0] / 1000, source[1] / 1000, "w*", ms=10, mec="black", mew=0.6)
        ax.set_xlabel("Distance x (km)")
        ax.set_ylabel("Depth z (km)")
    axes[0].set_title("Synthetic velocity")
    axes[1].contour(x / 1000, z / 1000, travel, levels=10, colors="white", linewidths=0.55, alpha=0.65)
    axes[1].set_title("Predicted travel time")
    fig.suptitle("EikoNet · physics-informed travel times", fontsize=14, fontweight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
