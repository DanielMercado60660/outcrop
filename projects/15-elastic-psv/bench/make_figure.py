"""Render a synthetic elastic-wave snapshot using the Metal implementation."""
from __future__ import annotations

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
    ny, nx, nt = 100, 150, 480
    spacing, dt = 10.0, 0.0004
    y, x = np.mgrid[:ny, :nx]
    lens = np.exp(-(((y - 64) / 15) ** 2 + ((x - 93) / 23) ** 2))
    vp = np.asarray(3000 + 350 * lens + 350 * (y >= 57), dtype=np.float32)
    vs = np.asarray(vp * 0.52, dtype=np.float32)
    rho = np.asarray(2000 + 150 * lens + 150 * (y >= 57), dtype=np.float32)
    t = np.arange(nt) * dt
    q = (np.pi * 25 * (t - 0.04)) ** 2
    source_y = np.asarray((1e8 * (1 - 2 * q) * np.exp(-q))[None, None, :],
                          dtype=np.float32)
    ys, xs = np.arange(0, ny - 1, 2), np.arange(0, nx - 1, 2)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    receivers = np.stack((yy.ravel(), xx.ravel()), axis=-1)[None].astype(np.int64)
    result = call("metal", "elastic", vp=vp, vs=vs, rho=rho,
                  source_y=source_y, source_x=np.zeros_like(source_y),
                  source_locations=np.array([[[25, 60]]], dtype=np.int64),
                  receiver_locations=receivers, spacing=[spacing, spacing], dt=dt,
                  pml_width=[12, 12, 12, 12], pml_freq=25.0, max_vel=4000.0)
    snapshot = result["p"][0, :, -1].reshape(len(ys), len(xs)) / 1e6
    limit = float(np.max(np.abs(snapshot))) or 1.0
    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    image = ax.imshow(snapshot, extent=[xs[0] * spacing / 1000, xs[-1] * spacing / 1000,
                                        ys[-1] * spacing / 1000, ys[0] * spacing / 1000],
                      cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="bilinear",
                      aspect="equal")
    ax.contour(x * spacing / 1000, y * spacing / 1000, vp, levels=[3300],
               colors="0.2", linewidths=0.7, alpha=0.65)
    ax.plot(0.6, 0.25, marker="*", markersize=12, markeredgecolor="white",
            markeredgewidth=0.8, color="#efb432", label="Force source")
    ax.set(xlabel="Horizontal distance (km)", ylabel="Depth (km)",
           title=f"Elastic P–SV propagation  |  pressure at {(nt - 1) * dt:.3f} s")
    ax.legend(loc="upper right", framealpha=0.88, fontsize=9)
    fig.colorbar(image, ax=ax, label="Pressure (MPa)", shrink=0.85, pad=0.02)
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
