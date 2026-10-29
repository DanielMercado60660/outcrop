"""Benchmark the analytic plate-flexure paths from the project directory."""

import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
sys.path.insert(0, str(PROJECT / "judge"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)

from outcrop_harness import call
from make_goldens import solution

SIZES = {
    "small": {"n": 800_000, "extent_km": 600},
    "medium": {"n": 8_000_000, "extent_km": 600},
    "large": {"n": 80_000_000, "extent_km": 600},
}
PARAMS = {"rigidity": float(np.float32(1e23)),
          "foundation": float(np.float32(32000)),
          "load": float(np.float32(1e12)), "plate": "continuous"}
MB = 1024 * 1024


def inputs(size):
    # Fixed regular coordinates are deterministic and include both plate halves.
    extent = size["extent_km"] * 1000
    return {"x": np.linspace(-extent, extent, size["n"], dtype=np.float32),
            **PARAMS}


def evaluate(path, args):
    if path == "judge":
        return solution(args["x"].astype(np.float64), args["rigidity"],
                        args["foundation"], args["load"], args["plate"])
    return call(path, "plate_flexure", **args)["deflection"]


def benchmark(path, args):
    evaluate(path, args)  # Warm up, including Metal process and shader startup.
    times, peaks, rss_peaks, child_highs = [], [], [], []
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
        elif path == "mlx":
            mx.reset_peak_memory()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        result = evaluate(path, args)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / MB)
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory() / MB)
            rss_peaks.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB)
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, after - before) / MB)
            child_highs.append(after / MB)
        del result
    methods = {
        "judge": "tracemalloc peak bytes during call, including NumPy allocations",
        "mlx": "MLX reset_peak_memory/get_peak_memory; process RSS high-water separately",
        "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta (macOS bytes)",
    }
    record = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
              "mem_method": methods[path]}
    if rss_peaks:
        record["process_peak_rss_mb"] = max(rss_peaks)
    if child_highs:
        record["child_peak_rss_high_water_mb"] = max(child_highs)
    return record


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {name}", flush=True)
            path_results[name] = benchmark(path, args)
        del args

    args = inputs(SIZES["medium"])
    reference = evaluate("judge", args)
    scale = float(np.max(np.abs(reference)))
    differences = {}
    for path in ("mlx", "metal"):
        output = evaluate(path, args)
        differences[path] = float(np.max(np.abs(output - reference)) / scale)
        del output
    report = {
        "project": "46-plate-flexure", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": "Coordinates are fixed and evenly spaced; scalar parameters are "
        "rounded to float32. Times are median of three calls after one warm-up, "
        "including harness conversion and Metal file I/O, process startup and "
        "shader compilation. Judge timing includes tracemalloc overhead. "
        "Memory is MiB. Child ru_maxrss is a cumulative high-water mark, so its "
        "delta can be zero after warm-up and is not an individual child's peak; "
        "the absolute high-water mark is also reported. Process RSS is likewise "
        "a lifetime high-water mark. Relative differences use the peak reference "
        "magnitude as denominator to avoid singularities at zero crossings.",
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
