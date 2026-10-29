"""Illustrate a synthetic bidirectional ambient-noise correlation stack."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call


def main():
    rng = np.random.default_rng(39001)
    windows, samples, dt, delay = 128, 1024, 0.01, 90
    a = rng.standard_normal((windows, samples), dtype=np.float32)
    directions = np.repeat([-1, 1], windows // 2)
    rng.shuffle(directions)
    b = np.stack([np.roll(trace, int(sign * delay)) for trace, sign in zip(a, directions)])
    b += 0.55 * rng.standard_normal(b.shape, dtype=np.float32)
    a += 0.55 * rng.standard_normal(a.shape, dtype=np.float32)
    result = call("metal", "ambient_noise", a=a, b=b, dt=dt, low=3.0, high=25.0, eps=1e-10)
    lag = (np.arange(samples) - samples // 2) * dt
    correlations = result["correlations"]
    stack = result["stack"]
    scale = np.max(np.abs(correlations))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bottom) = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True,
                                     gridspec_kw={"height_ratios": [2.1, 1]}, constrained_layout=True)
    image = ax.imshow(correlations / scale, aspect="auto", origin="lower",
                      extent=[lag[0] - dt / 2, lag[-1] + dt / 2, 0.5, windows + 0.5],
                      cmap="RdBu_r", vmin=-0.7, vmax=0.7, interpolation="nearest")
    ax.set_ylabel("Window number")
    ax.set_title("Ambient-noise interferometry | spectral whitening + correlation", loc="left", weight="bold")
    fig.colorbar(image, ax=ax, label="Normalized correlation", pad=0.02)
    bottom.plot(lag, stack, color="#253d65", lw=1.5, label="Stacked correlation")
    for i, t in enumerate((-delay * dt, delay * dt)):
        bottom.axvline(t, color="#d37835", ls="--", lw=1.2,
                       label="True travel time" if i == 0 else None)
    bottom.axhline(0, color="0.65", lw=0.7)
    bottom.set(xlim=(-2.0, 2.0), xlabel="Lag (s)", ylabel="Stack amplitude")
    bottom.legend(frameon=False, loc="upper right", fontsize=9)
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
