"""Plot a synthetic multi-station matched-filter detection example."""
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
    rng = np.random.default_rng(3803)
    sample_rate = 100.0
    n, m, stations = 4000, 201, 4
    t_template = (np.arange(m) - (m - 1) / 2) / sample_rate
    templates = np.empty((1, stations, m), dtype=np.float32)
    for s in range(stations):
        f = 4.0 + 0.4 * s
        x = np.pi * f * t_template
        templates[0, s] = (1 - 2 * x**2) * np.exp(-x**2)
    data = rng.normal(0, 0.16, (stations, n)).astype(np.float32)
    event_starts = [650, 1880, 3100]
    for start, amp in zip(event_starts, [1.2, 0.9, 1.1], strict=True):
        data[:, start:start + m] += amp * templates[0]
    threshold = 0.7
    corr = call("metal", "template_match", data=data, templates=templates,
                threshold=threshold)["correlation"][0]
    time = np.arange(n) / sample_rate
    # Correlation is indexed by window start; plot at the window center.
    corr_time = (np.arange(corr.shape[1]) + (m - 1) / 2) / sample_rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.25]})
    fig.patch.set_facecolor("#f8f9fb")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e9ef", linewidth=0.7)
    ax1.plot(time, data[0], color="#24466d", linewidth=0.7)
    ax1.set_ylabel("Amplitude (a.u.)")
    ax1.set_title("Synthetic station 1 waveform", loc="left", fontsize=10, weight="bold")
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.85, stations))
    for s, color in enumerate(colors):
        ax2.plot(corr_time, corr[s], lw=1, color=color, label=f"Station {s + 1}")
    ax2.axhline(threshold, color="#b5463a", lw=1, ls="--", label="Detection threshold")
    for start in event_starts:
        center = (start + (m - 1) / 2) / sample_rate
        ax1.axvline(center, color="#d17b42", alpha=0.6, lw=0.9)
        ax2.axvline(center, color="#d17b42", alpha=0.6, lw=0.9)
    ax2.set(xlabel="Time (s)", ylabel="Correlation coefficient", ylim=(-0.55, 1.08), xlim=(0, n / sample_rate))
    ax2.set_title("Normalized template correlation", loc="left", fontsize=10, weight="bold")
    ax2.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Earthquake detection with template matching", fontsize=15, weight="bold", x=0.09, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
