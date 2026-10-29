"""Train a small illustrative synthetic picker and plot one held-out trace."""
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


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    dt = 0.01
    t = np.arange(3000, dtype=np.float32) * dt
    data = rng.normal(0, 0.12, (n, t.size, 3)).astype(np.float32)
    labels = np.empty((n, 2), dtype=np.int32)
    for i in range(n):
        p = int(rng.integers(400, 950))
        s = int(p + rng.integers(500, 900))
        labels[i] = (p, s)
        for onset, freq, amplitudes in ((p, 7.0, (1.4, 0.35, 0.3)),
                                       (s, 4.0, (0.35, 1.2, 1.0))):
            elapsed = t - onset * dt
            envelope = np.where(elapsed >= 0, np.exp(-np.maximum(elapsed, 0) / 2.0), 0)
            for component, amp in enumerate(amplitudes):
                data[i, :, component] += amp * envelope * np.sin(
                    2 * np.pi * freq * elapsed + component * 0.7)
    return data, labels


def main():
    train, labels = synthetic(96, 43043)
    test, truth = synthetic(1, 43044)
    rng = np.random.default_rng(43045)
    steps, batch = 400, 32
    shape = (steps, batch)
    call("mlx", "train_phase_picker", waveforms=train, labels=labels,
         batch_indices=rng.integers(0, len(train), shape, dtype=np.int32),
         shifts=rng.integers(-150, 151, shape, dtype=np.int32),
         polarities=rng.choice(np.array([-1, 1], dtype=np.float32), shape),
         rotation_angles=rng.uniform(-np.pi, np.pi, shape).astype(np.float32),
         learning_rates=np.full(steps, 0.001, dtype=np.float32),
         sample_interval_s=0.01, seed=43043, label_sigma_samples=10.0)
    probabilities = call("mlx", "predict_phases", waveforms=test,
                         sample_interval_s=0.01)["probabilities"][0]
    time = np.arange(test.shape[1]) * 0.01
    colors = ("#266a9b", "#d05a3a")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for component, offset in enumerate((2.6, 0.0, -2.6)):
        axes[0].plot(time, test[0, :, component] + offset, color="#263b4a", lw=0.65)
    axes[0].set_yticks([2.6, 0, -2.6], ["Z", "N", "E"])
    axes[0].set_ylabel("Component\n(offset amplitude, counts)")
    for index, (label, color) in enumerate(zip(("P", "S"), colors)):
        onset = truth[0, index] * 0.01
        for ax in axes:
            ax.axvline(onset, color=color, ls="--", lw=1.1, alpha=0.8)
        axes[1].plot(time, probabilities[:, index + 1], color=color, lw=1.5,
                     label=f"{label} probability")
        axes[0].text(onset + 0.12, 3.85, f"{label} onset", color=color, fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Time from window start (s)")
    axes[1].legend(loc="upper right", frameon=True, ncol=2)
    axes[0].set_xlim(0, 30)
    fig.suptitle("MLX PhaseNet-style picker · synthetic seismogram", fontsize=15, weight="bold")
    axes[0].set_title("Illustrative model trained for 400 updates; dashed lines show synthetic onsets",
                      fontsize=9, color="#536575")
    fig.tight_layout()
    fig.savefig(ROOT / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
