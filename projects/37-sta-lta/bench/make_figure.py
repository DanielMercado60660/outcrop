"""Plot synthetic seismic arrivals and Metal STA/LTA triggers."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call


def main():
    rng = np.random.default_rng(3701)
    fs = 100.0
    t = np.arange(12_000, dtype=np.float64) / fs
    trace = rng.normal(0, 0.16, t.size)
    for arrival, duration, amplitude, frequency in [(22, 5, 1.0, 5),
                                                     (58, 8, 0.8, 3),
                                                     (91, 6, 1.2, 7)]:
        dt = t - arrival
        envelope = np.where(dt >= 0, np.exp(-np.maximum(dt, 0) / duration), 0)
        trace += amplitude * envelope * (np.sin(2 * np.pi * frequency * dt)
                                          + 0.35 * rng.normal(size=t.size))
    trace = trace.astype(np.float32)
    nsta, nlta = 50, 500
    on, off = 3.0, 1.5
    characteristic = call("metal", "classic_sta_lta", trace=trace,
                          nsta=nsta, nlta=nlta)["characteristic"]
    triggers = call("metal", "trigger_onset", characteristic=characteristic,
                    on=on, off=off)["triggers"]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor("#f8fafb")
    for ax in axes:
        ax.set_facecolor("#f8fafb")
        ax.grid(color="#dbe4e9", linewidth=0.7, alpha=0.8)
        for start, end in triggers:
            ax.axvspan(t[start], t[end], color="#e9a23b", alpha=0.23, lw=0)
    axes[0].plot(t, trace, color="#17627b", lw=0.65)
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].set_title("Synthetic seismic trace", loc="left", fontsize=11, weight="bold")
    axes[1].plot(t, characteristic, color="#244c70", lw=1.1, label="STA / LTA")
    axes[1].axhline(on, color="#c35a38", ls="--", lw=1, label="Trigger on: 3.0")
    axes[1].axhline(off, color="#8d6e3e", ls=":", lw=1.2, label="Trigger off: 1.5")
    axes[1].set_ylabel("STA / LTA ratio (1)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(t[0], t[-1])
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right", frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Classic STA/LTA · detecting seismic arrivals", x=0.08, ha="left",
                 fontsize=16, weight="bold", color="#153d52")
    fig.text(0.08, 0.9, "Metal  |  0.5 s short window  ·  5 s long window  |  shaded: active trigger",
             fontsize=10, color="#536b78")
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.87))
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
