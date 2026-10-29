"""Benchmark the combined f-k and Radon workload on each project path."""

import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"nx": 256, "nt": 1024, "dt_s": 0.004, "dx_m": 10.0},
    "medium": {"nx": 1024, "nt": 4096, "dt_s": 0.004, "dx_m": 10.0},
    "large": {"nx": 2048, "nt": 8192, "dt_s": 0.004, "dx_m": 10.0},
}
CUTOFF = 487.3
SLOWNESS = 8e-7
MB = 1024**2


def rss_bytes(who):
    """macOS reports bytes; Linux reports KiB."""
    value = resource.getrusage(who).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def inputs(size):
    nx, nt = size["nx"], size["nt"]
    rng = np.random.default_rng(4202)
    gather = rng.standard_normal((nx, nt), dtype=np.float32)
    offsets = np.linspace(-300, 300, nx, dtype=np.float32)
    fk = {"gather": gather, "dt": size["dt_s"], "dx": size["dx_m"],
          "velocity_cutoff": CUTOFF}
    radon = {"gather": gather, "offsets": offsets, "dt": size["dt_s"],
             "slowness": SLOWNESS}
    return fk, radon


def reference(fk, radon):
    """The NumPy FFT and projection formulas in judge/make_goldens.py."""
    gather = fk["gather"].astype(np.float64)
    nx, nt = gather.shape
    f = np.fft.fftfreq(nt, fk["dt"])
    k = np.fft.fftfreq(nx, fk["dx"])
    mask = np.abs(f)[None, :] > fk["velocity_cutoff"] * np.abs(k)[:, None]
    filtered_fk = np.fft.ifft2(np.fft.fft2(gather) * mask).real.astype(np.float32)
    offsets = radon["offsets"].astype(np.float64)
    steer = np.exp(-2j * np.pi * offsets[:, None] ** 2 * radon["slowness"] * f[None, :])
    spectrum = np.fft.fft(gather, axis=1)
    residual = spectrum - steer * np.mean(steer.conj() * spectrum, axis=0)
    filtered_radon = np.fft.ifft(residual, axis=1).real.astype(np.float32)
    return filtered_fk, filtered_radon


def workload(path, fk, radon):
    if path == "judge":
        return reference(fk, radon)
    return (call(path, "fk_filter", **fk)["filtered"],
            call(path, "radon_demultiple", **radon)["filtered"])


def max_rel(actual, expected):
    return max(float(np.max(np.abs(a.astype(np.float64) - b)) /
                     max(float(np.max(np.abs(b))), 1e-30))
               for a, b in zip(actual, expected))


def main():
    import mlx.core as mx

    results = {path: {} for path in ("judge", "mlx", "metal")}
    differences = {}
    for name, size in SIZES.items():
        fk, radon = inputs(size)
        medium_reference = None
        for path, path_results in results.items():
            print(f"{name}: {path}", flush=True)
            workload(path, fk, radon)
            timings = []
            peaks = []
            rss = []
            last = None
            for _ in range(3):
                if path == "mlx":
                    mx.reset_peak_memory()
                elif path == "judge":
                    tracemalloc.start()
                else:
                    before = rss_bytes(resource.RUSAGE_CHILDREN)
                start = time.perf_counter()
                last = workload(path, fk, radon)
                timings.append(time.perf_counter() - start)
                if path == "mlx":
                    peaks.append(mx.get_peak_memory())
                    rss.append(rss_bytes(resource.RUSAGE_SELF))
                elif path == "judge":
                    peaks.append(tracemalloc.get_traced_memory()[1])
                    tracemalloc.stop()
                else:
                    peaks.append(max(0, rss_bytes(resource.RUSAGE_CHILDREN) - before))
            methods = {
                "judge": "tracemalloc peak during workload (includes NumPy allocations)",
                "mlx": "MLX reset_peak_memory/get_peak_memory; process high-water RSS also reported",
                "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta across both child calls",
            }
            entry = {"wall_s": statistics.median(timings),
                     "peak_mem_mb": max(peaks) / MB, "mem_method": methods[path]}
            if path == "mlx":
                entry["process_peak_rss_mb"] = max(rss) / MB
            path_results[name] = entry
            if name == "medium":
                if path == "judge":
                    medium_reference = last
                else:
                    differences[path] = max_rel(last, medium_reference)
            del last
    output = {
        "project": "42-fk-radon", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": ("Each repeat runs fk_filter then radon_demultiple on the same seeded "
                  "float32 gather; wall time includes harness conversions, and Metal includes "
                  "two process launches and file I/O. F-k cutoff 487.3 m/s, Radon "
                  "slowness 8e-7 s/m^2, offsets linearly span -300 to 300 m. "
                  "Differences are the largest normalized maximum absolute error across "
                  "the two outputs. Memory is the maximum over three repeats, not a median. "
                  "Child ru_maxrss is a cumulative high-water mark, not additive or "
                  "resettable: after warmup its delta may be zero and understate child "
                  "memory. MLX process RSS is also a cumulative high-water mark. "
                  "Runtime and memory estimates are unverified on the target machine."),
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
