"""Render a synthetic mantle convection snapshot with the Metal solver."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call


def main():
    n, steps, dt, rayleigh = 128, 1500, 1.0e-5, 1.0e4
    axis = np.linspace(0.0, 1.0, n + 1)
    x, z = np.meshgrid(axis, axis)
    temperature = 1 - z + 0.12 * np.cos(np.pi * x) * np.sin(np.pi * z)
    temperature += 0.025 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z)
    temperature = np.asarray(temperature, dtype=np.float32, order="C")
    temperature[0, :], temperature[-1, :] = 1.0, 0.0
    out = call("metal", "convection", temperature=temperature,
               rayleigh=rayleigh, dt=dt, steps=steps)

    fig, ax = plt.subplots(figsize=(7.5, 6.4), layout="constrained")
    field = ax.pcolormesh(x, z, out["temperature"], shading="auto",
                          cmap="inferno", vmin=0, vmax=1, rasterized=True)
    ax.streamplot(axis, axis, out["u"], out["w"], color="white", density=1.15,
                  linewidth=0.65, arrowsize=0.8, integration_direction="both")
    ax.set(xlabel="Horizontal position / layer depth", ylabel="Height / layer depth",
           title="Mantle convection · Metal midpoint solver")
    ax.set_aspect("equal")
    ax.text(0.03, 0.03, f"Ra = {rayleigh:.0e}   |   t = {steps * dt:.3f}",
            transform=ax.transAxes, color="white", fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4})
    fig.colorbar(field, ax=ax, label="Dimensionless temperature", shrink=0.87)
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
