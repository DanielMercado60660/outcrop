"""Benchmark periodic shallow-water forward propagation."""
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
sys.path.insert(0, str(PROJECT / "judge"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)

from outcrop_harness import call
from make_goldens import advance

SIZES = {
    "small": {"n": 65536, "steps": 80, "dx_m": 100.0, "depth_m": 1000.0},
    "medium": {"n": 262144, "steps": 200, "dx_m": 100.0, "depth_m": 1000.0},
    "large": {"n": 1048576, "steps": 512, "dx_m": 100.0, "depth_m": 1000.0},
}


def inputs(size):
    rng = np.random.default_rng(4901)
    n = size["n"]
    depth = size["depth_m"]
    x = np.arange(n, dtype=np.float64) / n
    h = depth + 2.0 * np.sin(2.0 * np.pi * x) + rng.uniform(-0.05, 0.05, n)
    q = 10.0 * np.cos(2.0 * np.pi * x) + rng.uniform(-0.05, 0.05, n)
    initial = np.column_stack((h, q)).astype(np.float32)
    dx = size["dx_m"]
    dt = 0.25 * dx / np.sqrt(9.81 * (depth + 3.0))
    return {"initial": initial, "dx": dx, "dt": float(dt), "steps": size["steps"],
            "mode": "nonlinear", "depth": depth}


def run(path, args):
    if path == "judge":
        return advance(**args)
    return call(path, "propagate", **args)["state"]


def benchmark(path, args):
    if path == "mlx":
        import mlx.core as mx

    run(path, args)
    times, peaks, rss = [], [], []
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
            tracemalloc.reset_peak()
        elif path == "mlx":
            mx.reset_peak_memory()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        result = run(path, args)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory())
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, after - before))
        del result
    methods = {
        "judge": "tracemalloc peak traced bytes during call, including NumPy allocations",
        "mlx": "MLX peak allocated bytes after reset; process RSS high-water also reported",
        "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta, macOS bytes (not per-call RSS)",
    }
    entry = {"wall_s": statistics.median(times),
             "peak_mem_mb": max(peaks) / 1024**2, "mem_method": methods[path]}
    if rss:
        entry["process_peak_rss_mb"] = max(rss) / 1024**2
    return entry


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    differences = {}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {name}", flush=True)
            path_results[name] = benchmark(path, args)
        if name == "medium":
            reference = run("judge", args)
            for path in ("mlx", "metal"):
                actual = run(path, args).astype(np.float64)
                scales = np.maximum(np.max(np.abs(reference), axis=0), 1e-30)
                differences[path] = float(np.max(np.abs(actual - reference) / scales))
    output = {"project": "49-tsunami", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": differences,
              "notes": "Nonlinear periodic Rusanov propagation; seed 4901, CFL 0.25. "
              "Wall times include conversion and Metal process/IO startup. Memory is MiB; "
              "Metal child ru_maxrss is a cumulative high-water mark, so its delta can "
              "be zero after warm-up and is not an individual child peak. MLX RSS is "
              "the cumulative process high-water mark. Relative differences are maximum "
              "errors normalized by each reference component's maximum."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
