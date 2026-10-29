"""Illustrate synthetic hypocentre recovery and its uncertainty with Metal."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call


def main():
    rng = np.random.default_rng(30012026)
    angle = np.arange(16) * 2 * np.pi / 16
    radius = np.array([9000, 16000, 23000, 16000] * 4)
    stations = np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                                -rng.uniform(0, 450, 16))).astype(np.float32)
    sigma = np.linspace(0.015, 0.035, 16, dtype=np.float32)
    truth = np.array([[2800., -1900., 7800., 0.35]])
    arrivals = (truth[:, 3, None] + np.linalg.norm(
        truth[:, None, :3] - stations[None, :, :], axis=2) / 6000.
        + rng.normal(size=(1, 16)) * sigma).astype(np.float32)
    initial = (truth + [1200, -1000, 800, 0.15]).astype(np.float32)
    out = call("metal", "geiger", stations=stations, arrivals=arrivals,
               sigma=sigma, velocity=6000., initial=initial)
    model = out["models"][0].astype(float)
    ellipse = out["ellipse"][0].astype(float)
    eigval, eigvec = np.linalg.eigh(ellipse)
    phi = np.degrees(np.arctan2(eigvec[1, 1], eigvec[0, 1]))
    prediction = model[3] + np.linalg.norm(stations - model[:3], axis=1) / 6000.
    residual_ms = (arrivals[0] - prediction) * 1000

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax, detail) = plt.subplots(1, 2, figsize=(11, 5),
                                     gridspec_kw={"width_ratios": [1.15, 1]},
                                     layout="constrained")
    points = ax.scatter(stations[:, 0] / 1000, stations[:, 1] / 1000,
                        c=residual_ms, cmap="coolwarm", s=90, marker="^",
                        edgecolors="#263447", linewidths=0.7, zorder=3)
    lim = max(20., float(np.max(np.abs(residual_ms))))
    points.set_clim(-lim, lim)
    fig.colorbar(points, ax=ax, pad=0.02, label="Pick residual (ms)", shrink=0.8)
    ax.scatter(truth[0, 0] / 1000, truth[0, 1] / 1000, marker="*", s=180,
               color="#e49a24", edgecolors="#263447", zorder=4, label="True epicentre")
    ax.scatter(model[0] / 1000, model[1] / 1000, marker="x", s=75,
               color="#263447", linewidths=2, zorder=5, label="Geiger estimate")
    ax.set(xlabel="East (km)", ylabel="North (km)", title="Station network and pick residuals")
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    center = model[:2] / 1000
    widths = 2 * np.sqrt(np.maximum(eigval, 0)) / 1000
    detail.add_patch(Ellipse(center, widths[1], widths[0], angle=phi,
                             facecolor="#58a4a0", edgecolor="#246d70",
                             alpha=0.3, linewidth=1.5, label="95% confidence ellipse"))
    detail.scatter(*center, marker="x", s=85, color="#263447", linewidths=2,
                   zorder=4, label="Geiger estimate")
    detail.scatter(*(truth[0, :2] / 1000), marker="*", s=180, color="#e49a24",
                   edgecolors="#263447", zorder=3, label="True epicentre")
    span = max(0.2, widths[1] * 0.8, np.max(np.abs(center - truth[0, :2] / 1000)) * 1.5)
    midpoint = (center + truth[0, :2] / 1000) / 2
    detail.set_xlim(midpoint[0] - span, midpoint[0] + span)
    detail.set_ylim(midpoint[1] - span, midpoint[1] + span)
    detail.set_aspect("equal")
    detail.grid(alpha=0.18)
    detail.set(xlabel="East (km)", ylabel="North (km)", title="Epicentre and horizontal uncertainty")
    detail.legend(loc="upper right", fontsize=8)
    fig.suptitle("Earthquake location · Geiger inversion (Metal)", fontsize=15, weight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
