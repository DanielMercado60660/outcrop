"""Benchmark polygon gravity via the harness and independent reference."""
from __future__ import annotations

import importlib.util
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
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"vertices": 16, "observations": 512, "quadrature_order": 96},
    "medium": {"vertices": 32, "observations": 512, "quadrature_order": 96},
    "large": {"vertices": 128, "observations": 65536, "quadrature_order": 96},
}


def load_reference():
    spec = importlib.util.spec_from_file_location("gravity_reference", PROJECT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.integrated


def inputs(size):
    rng = np.random.default_rng(20261028)
    angle = np.arange(size["vertices"], dtype=np.float64) * (2 * np.pi / size["vertices"])
    vertices = np.column_stack((1000 * np.cos(angle), 1800 + 700 * np.sin(angle))).astype(np.float32)
    x = rng.uniform(-3500, 3500, size["observations"])
    observations = np.column_stack((x, np.zeros_like(x))).astype(np.float32)
    return {"vertices": vertices, "observations": observations, "density": 500.0}


def rss_mb(who):
    # On macOS ru_maxrss is in bytes, and is a lifetime high-water mark.
    return resource.getrusage(who).ru_maxrss / (1024 * 1024)


def measure(path, data, reference, order):
    def invoke():
        if path == "judge":
            return reference(**data, order=order)
        return call(path, "polygon_gravity", **data)["gz"]

    invoke()
    times, peaks, rss = [], [], []
    result = None
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
            tracemalloc.reset_peak()
        elif path == "mlx":
            mx.reset_peak_memory()
        before = rss_mb(resource.RUSAGE_CHILDREN) if path == "metal" else 0.0
        start = time.perf_counter()
        result = invoke()
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / (1024 * 1024))
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory() / (1024 * 1024))
            rss.append(rss_mb(resource.RUSAGE_SELF))
        else:
            peaks.append(max(0.0, rss_mb(resource.RUSAGE_CHILDREN) - before))
    methods = {
        "judge": "tracemalloc peak during call (MiB)",
        "mlx": "MLX mx.get_peak_memory after reset (MiB); process RSS high-water separately",
        "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta per call (MiB, macOS)",
    }
    entry = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks), "mem_method": methods[path]}
    if rss:
        entry["process_rss_mb"] = max(rss)
    return entry, np.asarray(result)


def main():
    reference = load_reference()
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium = {}
    for name, size in SIZES.items():
        data = inputs(size)
        for path, path_results in results.items():
            if path == "judge" and name == "large":
                path_results[name] = {"skipped": "96-point per-triangle area quadrature is expected to exceed 120 seconds."}
                continue
            print(f"Benchmarking {path} / {name}", flush=True)
            entry, output = measure(path, data, reference, size["quadrature_order"])
            path_results[name] = entry
            if name == "medium":
                medium[path] = output.astype(np.float64)
    denominator = float(np.max(np.abs(medium["judge"])))
    diffs = {path: float(np.max(np.abs(medium[path] - medium["judge"])) / denominator)
             for path in ("mlx", "metal")}
    payload = {
        "project": PROJECT.name, "sizes": SIZES, "results": results,
        "medium_max_rel_diff": diffs,
        "notes": "Times include harness conversion, Metal process launch, file I/O and runtime shader compilation. "
                 "Memory values are MiB; MLX peak is allocator memory and process RSS is a lifetime high-water mark. "
                 "RUSAGE_CHILDREN ru_maxrss is a cumulative child high-water mark, so its delta can be zero "
                 "after warmup and is not an individual child's true peak. Judge tracemalloc tracks Python/NumPy allocations. "
                 "Differences are maximum absolute error divided by the maximum absolute reference value. "
                 "Large runtime is an unverified estimate; GPU execution was reserved.",
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
