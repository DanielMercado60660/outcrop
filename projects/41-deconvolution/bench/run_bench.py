"""Benchmark causal spiking deconvolution from the project directory."""

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
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"trace_length": 16384, "wavelet_length": 3, "inverse_length": 160},
    "medium": {"trace_length": 65536, "wavelet_length": 3, "inverse_length": 320},
    "large": {"trace_length": 262144, "wavelet_length": 3, "inverse_length": 768},
}


def inputs(size):
    rng = np.random.default_rng(4101)
    reflectivity = rng.normal(0, 0.1, size["trace_length"]).astype(np.float32)
    wavelet = np.array([1.0, 0.5, -0.2], dtype=np.float32)
    trace = np.convolve(reflectivity, wavelet)[:len(reflectivity)].astype(np.float32)
    return {"trace": trace, "wavelet": wavelet,
            "inverse_length": size["inverse_length"], "stabilization": 1e-6}


def reference(trace, wavelet, inverse_length, stabilization):
    # Same float64 full-convolution matrix and normal equations as the judge.
    w = wavelet.astype(np.float64)
    a = np.zeros((len(w) + inverse_length - 1, inverse_length))
    for j in range(inverse_length):
        a[j:j + len(w), j] = w
    target = np.zeros(a.shape[0])
    target[0] = 1
    inverse = np.linalg.solve(a.T @ a + stabilization * np.eye(inverse_length),
                              a.T @ target)
    recovered = np.convolve(trace.astype(np.float64), inverse)[:len(trace)]
    return {"inverse": inverse, "recovered": recovered}


def run(path, args):
    if path == "judge":
        return reference(**args)
    return call(path, "spiking_deconvolution", **args)


def measure(path, args):
    import mlx.core as mx

    run(path, args)
    if path == "mlx":
        mx.reset_peak_memory()
    if path == "judge":
        tracemalloc.start()
    before_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    times = []
    for _ in range(3):
        start = time.perf_counter()
        run(path, args)
        times.append(time.perf_counter() - start)
    result = {"wall_s": float(statistics.median(times))}
    if path == "judge":
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.update(peak_mem_mb=peak / 1e6,
                      mem_method="tracemalloc peak during three calls, bytes / 1e6")
    elif path == "mlx":
        result.update(peak_mem_mb=mx.get_peak_memory() / 1e6,
                      mem_method="MLX peak allocated bytes after reset, bytes / 1e6",
                      process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    else:
        after_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        result.update(peak_mem_mb=max(0, after_child - before_child) / 1e6,
                      child_peak_rss_mb=after_child / 1e6,
                      mem_method="macOS RUSAGE_CHILDREN ru_maxrss high-water delta after warmup, bytes / 1e6")
    return result


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {name}...", flush=True)
            path_results[name] = measure(path, args)
    args = inputs(SIZES["medium"])
    expected = run("judge", args)
    differences = {}
    for path in ("mlx", "metal"):
        actual = run(path, args)
        differences[path] = max(
            float(np.max(np.abs(actual[key].astype(np.float64) - value)) /
                  max(float(np.max(np.abs(value))), 1e-30))
            for key, value in expected.items()
        )
    report = {
        "project": "41-deconvolution", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": (
            "Spiking deconvolution, seed 4101, stabilization 1e-6. Wall times include "
            "harness conversion and Metal process startup, shader compilation and file I/O. "
            "Relative difference is the largest per-output max absolute error divided "
            "by that reference output's max absolute value. Metal child RSS is a cumulative "
            "high-water mark, so its delta can be zero after warmup or an earlier larger child; "
            "child_peak_rss_mb also reports the absolute high-water mark. MLX process RSS is "
            "the lifetime high-water mark. Memory units are decimal MB on macOS."
        ),
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
