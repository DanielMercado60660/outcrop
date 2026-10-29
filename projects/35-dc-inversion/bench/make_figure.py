"""Illustrate a synthetic 2.5-D DC resistivity inversion."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_bench import ROOT, call, make_problem


def main():
    g = make_problem(nx=12, nz=7, electrodes=8, quadrature=13)
    x = g["origin"][0] + (np.arange(12) + 0.5) * 10
    z = g["origin"][1] + (np.arange(7) + 0.5) * 10
    xx, zz = np.meshgrid(x, z)
    initial = np.full((7, 12), 2.0, dtype=np.float32)
    truth = initial.copy()
    truth[(np.abs(xx + 10) <= 10) & (zz >= -35) & (zz <= -15)] = np.log10(25)
    geometry = {key: value for key, value in g.items() if key != "log10_rho"}
    observed = call("mlx", "dc_predict", **geometry, log10_rho=truth)["predicted"]
    sigma = (0.02 * np.abs(observed) + 0.001 * np.max(np.abs(observed))).astype(np.float32)
    active = ((np.abs(xx) <= 40) & (zz >= -45)).astype(np.int32)
    result = call("mlx", "dc_invert", **geometry, observed=observed.astype(np.float32),
                  standard_deviation=sigma, initial=initial, active=active, beta=0.02,
                  lower_bound=0.5, upper_bound=3.5, max_iterations=8)
    recovered = result["log10_rho"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True,
                             constrained_layout=True)
    extent = (g["origin"][0], -g["origin"][0], g["origin"][1], 0)
    for ax, model, title in zip(axes, (truth, recovered), ("Synthetic truth", "Recovered model")):
        image = ax.imshow(10**model, origin="lower", extent=extent, aspect="equal",
                          cmap="viridis", vmin=25, vmax=100, interpolation="nearest")
        ax.scatter(g["electrodes"][:, 0], np.zeros(len(g["electrodes"])),
                   marker="v", s=24, color="#f5a623", edgecolor="white", linewidth=0.4,
                   clip_on=False, zorder=3)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.tick_params(labelsize=9)
    bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.03)
    bar.set_label("Resistivity (Ω m)")
    fig.suptitle("2.5-D DC resistivity inversion", fontsize=15, weight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
