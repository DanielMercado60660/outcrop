"""Plot a synthetic cooling-half-space geotherm computed with Metal."""
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


def main():
    depth = np.linspace(0, 200_000, 501, dtype=np.float32)
    ages = np.geomspace(1, 200, 450).astype(np.float32)
    result = call("metal", "geotherm", depth_m=depth, age_myr=ages,
                  surface_temp_c=0.0, mantle_temp_c=1350.0,
                  diffusivity_m2_s=1e-6, conductivity_w_m_k=3.3)
    temp = result["temperature_c"].T
    heat = result["heat_flow_w_m2"][:, 0] * 1000
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax, flux_ax) = plt.subplots(1, 2, figsize=(11, 5),
                                      gridspec_kw={"width_ratios": [2.4, 1]},
                                      constrained_layout=True)
    mesh = ax.pcolormesh(ages, depth / 1000, temp, shading="auto", cmap="inferno",
                         vmin=0, vmax=1350, rasterized=True)
    contours = ax.contour(ages, depth / 1000, temp, levels=[600, 1000, 1200],
                          colors="white", linewidths=0.8, alpha=0.8)
    ax.clabel(contours, fmt="%d °C", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(200, 0)
    ax.set_xlabel("Plate age (Myr)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Temperature evolution", loc="left", weight="bold")
    fig.colorbar(mesh, ax=ax, label="Temperature (°C)", pad=0.02)
    flux_ax.plot(ages, heat, color="#b34332", linewidth=2.5)
    flux_ax.set_xscale("log")
    flux_ax.set_xlabel("Plate age (Myr)")
    flux_ax.set_ylabel("Surface heat flow (mW/m²)")
    flux_ax.set_title("Surface heat loss", loc="left", weight="bold")
    flux_ax.grid(alpha=0.2, which="both")
    fig.suptitle("Cooling half-space geotherm · Metal", fontsize=15, weight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
