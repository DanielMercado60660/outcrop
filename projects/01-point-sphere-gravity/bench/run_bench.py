"""Benchmark the gravity implementations; run from the project directory."""

import json
import os
import resource
import runpy
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(ROOT / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)

from outcrop_harness import call

SIZES = {
    "small": {"observations": 16384, "sources": 64},
    "medium": {"observations": 131072, "sources": 64},
    "large": {"observations": 1048576, "sources": 128},
}
REFERENCE = runpy.run_path(str(PROJECT / "judge" / "make_goldens.py"))["field"]


def make_inputs(size):
    rng = np.random.default_rng(20261028)
    n, s = size["observations"], size["sources"]
    return {
        "coordinates": rng.uniform([-5000, -5000, 0], [5000, 5000, 2000], (n, 3)).astype(np.float32),
        "centers": rng.uniform([-4000, -4000, -3500], [4000, 4000, -500], (s, 3)).astype(np.float32),
        "masses": rng.uniform(1e14, 1e16, s).astype(np.float32),
        "radii": rng.uniform(0, 1000, s).astype(np.float32),
    }


def judge(inputs):
    """Apply the float64 oracle in observation blocks to bound temporary memory."""
    xyz = inputs["coordinates"]
    centers, masses, radii = (inputs[k].astype(np.float64) for k in ("centers", "masses", "radii"))
    potential = np.empty(len(xyz), dtype=np.float64)
    acceleration = np.empty((len(xyz), 3), dtype=np.float64)
    for start in range(0, len(xyz), 4096):
        stop = min(start + 4096, len(xyz))
        potential[start:stop], acceleration[start:stop] = REFERENCE(
            xyz[start:stop].astype(np.float64), centers, masses, radii
        )
    return {"potential": potential, "acceleration": acceleration}


def invoke(path, inputs):
    return judge(inputs) if path == "judge" else call(path, "gravity", **inputs)


def benchmark(path, inputs):
    if path == "mlx":
        import mlx.core as mx

    invoke(path, inputs)
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
        result = invoke(path, inputs)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory() / 1e6)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, after - before) / 1e6)
        del result
    methods = {
        "judge": "tracemalloc peak during call (includes traced NumPy allocations)",
        "mlx": "MLX reset_peak_memory/get_peak_memory; process RSS high-water mark separately",
        "metal": "macOS RUSAGE_CHILDREN ru_maxrss high-water delta; bytes converted to MB",
    }
    record = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks), "mem_method": methods[path]}
    if rss:
        record["process_peak_rss_mb"] = max(rss)
    return record


def max_relative_difference(actual, reference):
    # Normalize each physical field by its own maximum reference magnitude.
    errors = []
    for key in ("potential", "acceleration"):
        ref = reference[key]
        scale = float(np.max(np.abs(ref)))
        errors.append(float(np.max(np.abs(actual[key].astype(np.float64) - ref))) / scale if scale else 0.0)
    return max(errors)


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    differences = {}
    for name, size in SIZES.items():
        inputs = make_inputs(size)
        for path, path_results in results.items():
            path_results[name] = benchmark(path, inputs)
        if name == "medium":
            reference = judge(inputs)
            for path in ("mlx", "metal"):
                differences[path] = max_relative_difference(invoke(path, inputs), reference)
    report = {
        "project": PROJECT.name,
        "sizes": SIZES,
        "results": results,
        "medium_max_rel_diff": differences,
        "notes": (
            "Times are medians of three end-to-end calls after one warm-up; Metal includes "
            "file I/O, process startup and shader compilation per call. Judge uses float64 "
            "oracle blocks of 4096 observations and tracemalloc is active during timing. "
            "MLX memory is the device allocator peak, and process_peak_rss_mb is the Python "
            "process lifetime RSS high-water mark. Metal ru_maxrss is a cumulative child "
            "high-water mark, not a per-child counter: its delta can be zero after warm-up "
            "and does not measure Metal GPU allocations. All memory values use decimal MB. "
            "Relative differences are maximum absolute errors normalized by each field's "
            "maximum absolute reference value. Timings and size suitability are unverified."
        ),
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
