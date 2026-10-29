"""Render a synthetic polygon and its Metal-computed gravity response."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call


def main():
    vertices = np.array([[-1450, 650], [-850, 400], [150, 520], [1050, 850],
                         [1450, 1550], [400, 2350], [-950, 2000]], dtype=np.float32)
    x = np.linspace(-3500, 3500, 700, dtype=np.float32)
    observations = np.column_stack((x, np.zeros_like(x)))
    gz = call("metal", "polygon_gravity", vertices=vertices,
              observations=observations, density=500.0)["gz"]

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc",
                         "axes.facecolor": "#f8fafc"})
    fig, (ax, model) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.12})
    ax.plot(x / 1000, gz, color="#127f8b", lw=2.8)
    ax.fill_between(x / 1000, gz, 0, color="#45b4b8", alpha=0.16)
    ax.set_ylabel("Downward gravity (mGal)")
    ax.grid(axis="y", color="#cbd5e1", alpha=0.65, lw=0.8)
    ax.set_ylim(bottom=0)
    model.axhline(0, color="#64748b", lw=1.2)
    model.add_patch(Polygon(vertices / 1000, closed=True, facecolor="#d08b55",
                            edgecolor="#713f27", linewidth=1.8))
    model.text(0, 1.35, r"Density contrast: +500 kg/m$^3$", ha="center",
               va="center", fontsize=10, color="#422818")
    model.set_xlim(-3.5, 3.5)
    model.set_ylim(2.8, -0.15)
    model.set_xlabel("Horizontal distance (km)")
    model.set_ylabel("Depth (km)")
    model.grid(color="#cbd5e1", alpha=0.45, lw=0.8)
    model.set_axisbelow(True)
    fig.suptitle("Talwani 2D · gravity of a buried polygon", x=0.12, ha="left",
                 fontsize=17, fontweight="bold", color="#172b3a")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.1)
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
