"""Plot a synthetic acoustic pulse propagating through a 1-D domain."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call


def main():
    n, steps = 601, 800
    dx, c = 5.0, 1500.0
    dt = 0.54 * dx / c
    x = np.arange(n) * dx
    center, width = 1500.0, 100.0

    def pulse(z):
        return np.exp(-((z - center) / width) ** 2)

    u0 = pulse(x).astype(np.float32)
    u_prev = (0.5 * (pulse(x + c * dt) + pulse(x - c * dt))).astype(np.float32)
    wave = call("metal", "propagate", u0=u0, u_prev=u_prev, c=c,
                dx=dx, dt=dt, steps=steps)["wavefield"]

    fig, ax = plt.subplots(figsize=(9.0, 5.4), layout="constrained")
    image = ax.imshow(wave, origin="lower", aspect="auto", interpolation="nearest",
                      extent=(x[0] / 1000, x[-1] / 1000, 0, steps * dt),
                      cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xlabel="Distance (km)", ylabel="Time (s)",
           title="1D acoustic propagation · Metal")
    bar = fig.colorbar(image, ax=ax, pad=0.025, shrink=0.92)
    bar.set_label("Displacement (normalized)")
    fig.savefig(PROJECT / "figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
