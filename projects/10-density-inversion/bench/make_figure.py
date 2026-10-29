"""Make a synthetic gravity inversion figure with the Metal implementation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from run_bench import PROJECT, call, make_problem


def main():
    shape = [8, 8, 4]
    problem = make_problem({"shape": shape, "n_data": 144})
    result = call("metal", "invert", **problem)
    nx, ny, nz = shape
    truth = np.zeros(shape, dtype=np.float32)
    truth[nx // 4:nx // 2, ny // 4:ny // 2, nz // 2:] = 0.5
    truth[nx // 2:3 * nx // 4, ny // 2:3 * ny // 4, nz // 2:] = -0.4
    recovered = result["model"].reshape(shape, order="F")
    extent = [-200, 200, -200, 200]
    limit = max(0.5, float(np.max(np.abs(recovered[:, :, -1]))))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    for ax, model, label in zip(axes[:2], (truth, recovered),
                                ("True density", "Recovered density")):
        im = ax.imshow(model[:, :, -1].T, origin="lower", extent=extent,
                       cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set(title=label, xlabel="East (m)", ylabel="North (m)")
    fig.colorbar(im, ax=axes[:2], shrink=0.8, label="Density contrast (g/cm³)")
    anomaly = result["predicted"].reshape(12, 12)
    im2 = axes[2].imshow(anomaly, origin="lower", extent=extent,
                         cmap="RdBu_r", vmin=-np.max(np.abs(anomaly)),
                         vmax=np.max(np.abs(anomaly)), interpolation="bilinear")
    axes[2].set(title="Predicted gravity", xlabel="East (m)", ylabel="North (m)")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="Upward gravity (mGal)")
    fig.suptitle("Depth-weighted gravity inversion · Metal", fontsize=14, fontweight="bold")
    fig.savefig(Path(PROJECT) / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
