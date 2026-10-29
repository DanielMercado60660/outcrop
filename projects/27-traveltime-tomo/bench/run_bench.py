"""Benchmark fixed-ray damped slowness inversion from the project root."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"nx": 12, "nz": 12, "nrays": 576},
    "medium": {"nx": 18, "nz": 18, "nrays": 1296},
    "large": {"nx": 24, "nz": 24, "nrays": 2304},
}


def inputs(size):
    rng = np.random.default_rng(27001)
    n = size["nx"] * size["nz"]
    # The solver accepts arbitrary nonnegative linear ray matrices.
    matrix = rng.uniform(0, 100, (size["nrays"], n)).astype(np.float32)
    matrix *= (rng.random(matrix.shape) < 0.15).astype(np.float32)
    prior = np.full(n, 0.0005, dtype=np.float32)
    truth = (prior + rng.normal(0, 0.000025, n)).astype(np.float32)
    times = (matrix @ truth).astype(np.float32)
    return {"matrix": matrix, "travel_times": times, "reference_slowness": prior, "damping": 200.0}


def load_judge():
    spec = importlib.util.spec_from_file_location("tomo_judge", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(path, args, judge):
    if path == "judge":
        slowness, predicted, _ = judge.solution(
            args["matrix"], args["travel_times"], args["reference_slowness"], args["damping"]
        )
        return {"slowness": slowness, "predicted_times": predicted}
    return call(path, "solve_tomography", **args)


def benchmark(path, args, judge):
    invoke(path, args, judge)
    times = []
    output = None
    if path == "mlx":
        import mlx.core as mx
        mx.reset_peak_memory()
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    for _ in range(3):
        start = time.perf_counter()
        output = invoke(path, args, judge)
        times.append(time.perf_counter() - start)
    entry = {"wall_s": float(statistics.median(times))}
    if path == "mlx":
        entry["peak_mem_mb"] = float(mx.get_peak_memory() / 1e6)
        entry["process_peak_rss_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        entry["mem_method"] = "MLX peak allocation after reset; process lifetime peak RSS also reported (macOS bytes)"
    elif path == "metal":
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        entry["peak_mem_mb"] = float(max(0, after - before) / 1e6)
        entry["mem_method"] = "RUSAGE_CHILDREN ru_maxrss high-water delta (macOS bytes); may be zero if an earlier child peaked higher"
    else:
        tracemalloc.start()
        tracemalloc.reset_peak()
        invoke(path, args, judge)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        entry["peak_mem_mb"] = float(peak / 1e6)
        entry["mem_method"] = "tracemalloc peak in separate invocation; native library allocations may be omitted"
    return entry, output


def main():
    judge = load_judge()
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium_outputs = {}
    for size_name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            if path == "judge" and size_name == "large":
                path_results[size_name] = {"skipped": "Float64 augmented SVD is expected to exceed 120 s at this size."}
                continue
            print(f"Benchmarking {path} {size_name}...", flush=True)
            entry, output = benchmark(path, args, judge)
            path_results[size_name] = entry
            if size_name == "medium":
                medium_outputs[path] = output
    diffs = {}
    for path in ("mlx", "metal"):
        diffs[path] = max(
            float(np.max(np.abs(medium_outputs[path][key] - ref)) / max(float(np.max(np.abs(ref))), 1e-30))
            for key, ref in medium_outputs["judge"].items()
        )
    payload = {
        "project": "27-traveltime-tomo", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": diffs,
        "notes": "Damped solve_tomography only, with deterministic sparse nonnegative synthetic matrices, damping 200 m. "
        "One warmup and three timed calls; median includes harness conversion and Metal subprocess/file overhead. "
        "Relative difference is maximum absolute error normalized by each output's maximum reference magnitude. "
        "RSS uses macOS ru_maxrss units; Metal child high-water delta is not an individual child peak.",
    }
    (ROOT / "bench" / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
