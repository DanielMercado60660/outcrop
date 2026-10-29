"""Image synthetic reflectors with the Metal Born operator and LSRTM."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from run_bench import ROOT, call, survey


def main():
    inputs = survey(nz=56, nx=72, nt=480, nshot=2, nrec=34, iterations=3)
    z, x = np.indices(inputs["v"].shape)
    truth = np.zeros_like(inputs["v"])
    for depth, slope, amp in ((16, 0.06, 100.0), (25, -0.045, -85.0),
                              (33, 0.035, 70.0)):
        center = depth + slope * (x - 35.5)
        truth += (amp * np.exp(-0.5 * ((z - center) / 0.9) ** 2)).astype(np.float32)
    truth *= inputs["mask"]
    forward_args = {k: v for k, v in inputs.items() if k not in ("observed", "iterations")}
    inputs["observed"] = call("metal", "born_model", model=truth, **forward_args)["predicted"]
    result = call("metal", "lsrtm", **inputs)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    fig.patch.set_facecolor("#f7f8fa")
    panels = ((truth, "Synthetic truth", "Velocity perturbation (m/s)"),
              (result["rtm"], "Reverse-time migration", "Normalized amplitude"),
              (result["image"], "LSRTM · 3 CGLS updates", "Normalized amplitude"))
    for ax, (data, title, label) in zip(axes, panels):
        peak = float(np.max(np.abs(data)))
        if label == "Normalized amplitude":
            data = data / peak if peak else data
            limit = 1.0
        else:
            limit = peak
        im = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit,
                       extent=(-5, 715, 555, -5), interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="semibold", pad=9)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.tick_params(labelsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025, label=label)
    fig.suptitle("Least-squares reverse-time migration · synthetic reflectors",
                 fontsize=14, fontweight="bold")
    fig.savefig(ROOT / "figure.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
