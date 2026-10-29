"""Show true, smoothed and inverted Marmousi2 using the Metal discrete adjoint."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_bench import (
    ATTRIBUTION,
    ROOT,
    SIZES,
    add_observations,
    build_inputs,
    invoke,
    smooth,
)

ITERATIONS = 3
SHOTS = 4
MAX_UPDATE_M_PER_S = 30.0
BACKTRACKS = 2


def invert(args):
    """Three projected, smoothed-gradient FWI iterations with a descent test.

    Only current velocity, observed traces and gradients enter updates. The
    known 450 m water layer is fixed. A failed line search leaves that
    iteration's velocity unchanged; no truth-dependent update is substituted.
    """
    velocity = args["velocity"].copy()
    water_rows = int(450 / args["grid_spacing"])
    current = invoke("metal", args)
    losses = [float(current["objective"])]
    accepted = 0
    for iteration in range(ITERATIONS):
        gradient = np.asarray(current["gradient"], dtype=np.float64).copy()
        gradient[:water_rows] = 0
        direction = smooth(gradient, 0.75).astype(np.float64)
        direction[:water_rows] = 0
        # Ensure smoothing has preserved a descent direction.
        if np.sum(direction * gradient) <= 0:
            direction = gradient
        scale = float(np.max(np.abs(direction)))
        if not np.isfinite(scale) or scale == 0:
            raise RuntimeError("Metal returned an unusable sediment gradient")
        for backtrack in range(BACKTRACKS):
            step = MAX_UPDATE_M_PER_S * 0.5**backtrack
            candidate = np.clip(velocity - step * direction/scale, 1000.0, args["max_vel"])
            candidate = np.ascontiguousarray(candidate, dtype=np.float32)
            candidate[:water_rows] = 1500.0
            trial = invoke("metal", {**args, "velocity": candidate})
            if float(trial["objective"]) < float(current["objective"]):
                velocity, current = candidate, trial
                accepted += 1
                break
        losses.append(float(current["objective"]))
        print(f"FWI {iteration+1}/{ITERATIONS}: objective={losses[-1]:.7g}", flush=True)
    return velocity, losses, accepted


def main():
    # Full licensed 70 x 340 model, 4.5 Hz, 4 s. Four shots limit the figure
    # cost to at most 32 one-shot API calls, including data and line searches.
    args, truth = build_inputs("medium", shots=SHOTS)
    add_observations(args, truth)
    initial = args["velocity"].copy()
    recovered, losses, accepted = invert(args)
    if losses[0] <= 0:
        raise RuntimeError("Starting model unexpectedly has zero data misfit")
    ratio = losses[-1] / losses[0]
    spacing_km = SIZES["medium"]["grid_spacing"] / 1000
    nz, nx = truth.shape
    # Block means cover [i*d,(i+1)*d); draw those cell footprints.
    extent = (0, nx * spacing_km, nz * spacing_km, 0)
    vmin, vmax = float(truth.min())/1000, float(truth.max())/1000
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.labelcolor": "#263244",
        "text.color": "#182538", "xtick.color": "#44536a", "ytick.color": "#44536a",
    })
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.88, top=0.87, bottom=0.19, hspace=0.36)
    titles = (
        "a  True Marmousi2 velocity",
        "b  Starting model · Gaussian smoothing, σ = 150 m",
        f"c  After {ITERATIONS} FWI iterations · {accepted} accepted updates · J / J₀ = {ratio:.3f}",
    )
    source_x = (args["source_locations"][:, 0, 1] + 0.5) * spacing_km
    source_z = (args["source_locations"][:, 0, 0] + 0.5) * spacing_km
    for ax, model, title in zip(axes, (truth, initial, recovered), titles):
        image = ax.imshow(model/1000, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax,
                          interpolation="nearest", aspect="equal")
        ax.scatter(source_x, source_z, marker="v", s=32, facecolor="white",
                   edgecolor="#182538", linewidth=0.7, zorder=3)
        ax.set_title(title, loc="left", fontsize=11, pad=8)
        ax.set_ylabel("Depth (km)")
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticks(np.arange(0, 18, 2))
    axes[-1].set_xlabel("Distance (km)")
    color_axis = fig.add_axes((0.905, 0.225, 0.017, 0.61))
    fig.colorbar(image, cax=color_axis, label="P-wave velocity (km/s)")
    fig.suptitle("Marmousi2 full-waveform inversion", x=0.075, y=0.965, ha="left",
                 fontsize=20, fontweight="bold")
    fig.text(0.075, 0.92,
             "Metal discrete adjoint  ·  50 m cells  ·  4.5 Hz source  ·  4 s records  ·  "
             "4 surface shots  ·  water layer fixed", fontsize=10, color="#44536a")
    caption = ATTRIBUTION + "\nCC BY 4.0: https://creativecommons.org/licenses/by/4.0/"
    fig.text(0.075, 0.065, "\n".join(
        textwrap.fill(line, width=143) for line in caption.splitlines()
    ), fontsize=8, va="bottom", linespacing=1.5, color="#44536a")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
