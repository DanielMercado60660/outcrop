"""Benchmark Boule and MLX normal gravity from the project directory."""
from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import boule
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

# Batching bounds the working set even for the 134 million-point large case.
SIZES = {
    "small": {"points_per_batch": 1_048_576, "batches": 1},
    "medium": {"points_per_batch": 1_048_576, "batches": 16},
    "large": {"points_per_batch": 1_048_576, "batches": 128},
}


def inputs(n):
    rng = np.random.default_rng(82026)
    return {
        "latitude": rng.uniform(-90, 90, n).astype(np.float32),
        "height": rng.uniform(0, 2_000_000, n).astype(np.float32),
    }


def evaluate(path, data):
    if path == "judge":
        return np.asarray(boule.WGS84.normal_gravity(
            (None, data["latitude"].astype(np.float64),
             data["height"].astype(np.float64)),
            coordinate_system="geodetic", si_units=True,
        ))
    return call("mlx", "normal_gravity", **data)["gravity"]


def run_batch(path, data, count):
    for _ in range(count):
        result = evaluate(path, data)
    return result


def measure(path, data, count):
    run_batch(path, data, count)  # warm-up, including MLX compilation
    times = []
    for _ in range(3):
        start = time.perf_counter()
        run_batch(path, data, count)
        times.append(time.perf_counter() - start)
    if path == "judge":
        tracemalloc.start()
        tracemalloc.reset_peak()
        run_batch(path, data, count)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        method = "tracemalloc peak allocated bytes during a separate run"
        rss = None
    else:
        mx.reset_peak_memory()
        run_batch(path, data, count)
        peak = mx.get_peak_memory()
        # On macOS ru_maxrss is bytes; it is the process lifetime high-water mark.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        method = "MLX peak allocated bytes during a separate run; process RSS is lifetime high-water mark"
    result = {"wall_s": statistics.median(times), "peak_mem_mb": peak / 1024**2,
              "mem_method": method}
    if rss is not None:
        result["process_peak_rss_mb"] = rss
    return result


def main():
    data = inputs(SIZES["medium"]["points_per_batch"])
    reference = evaluate("judge", data)
    actual = evaluate("mlx", data).astype(np.float64)
    relative = float(np.max(np.abs(actual - reference) / np.maximum(np.abs(reference), 1e-30)))
    del reference, actual
    results = {"judge": {}, "mlx": {}}
    for path, path_results in results.items():
        for name, size in SIZES.items():
            if name == "large" and path == "judge":
                medium = path_results["medium"]["wall_s"]
                estimate = medium * size["batches"] / SIZES["medium"]["batches"]
                if estimate > 120:
                    path_results[name] = {"skipped": f"estimated {estimate:.1f} s per run from medium"}
                    continue
            path_results[name] = measure(path, data, size["batches"])
            print(path, name, path_results[name], flush=True)
    output = {
        "project": "08-normal-gravity", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": {"mlx": relative},
        "notes": "Seed 82026; each size repeats the same float32 batch of uniformly sampled latitudes and heights (0–2000 km). Times include input conversion and output materialization through the harness; memory is measured in an additional run. MLX peak includes live MLX allocations at reset. RSS is macOS process lifetime high-water, not a per-run delta. Large timing is hardware-dependent.",
    }
    (ROOT / "bench" / "results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
