"""Benchmark first-order eikonal solvers from the project directory."""

import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import skfmm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"shape": [28, 28, 28], "spacing_m": 10.0, "seed": 25},
    "medium": {"shape": [64, 64, 64], "spacing_m": 10.0, "seed": 25},
    "large": {"shape": [128, 128, 128], "spacing_m": 10.0, "seed": 25},
}


def inputs(params):
    """A smooth heterogeneous speed and a small point-source level set."""
    shape = params["shape"]
    h = params["spacing_m"]
    rng = np.random.default_rng(params["seed"])
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    axes = [np.arange(n, dtype=np.float32) for n in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
    center = [(n - 1) // 2 for n in shape]
    radius = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2
                     + (z - center[2]) ** 2)
    phi = np.asarray((radius - 0.01) * h, dtype=np.float32)
    speed = np.asarray(2500.0 + 250.0 * np.sin(x / shape[0] * 6 + phases[0])
                       * np.cos(y / shape[1] * 6 + phases[1])
                       + 180.0 * np.sin(z / shape[2] * 6 + phases[2]),
                       dtype=np.float32)
    return {"phi": phi, "speed": speed, "spacing": h}


def solve(path, data):
    if path == "judge":
        return np.asarray(skfmm.travel_time(data["phi"], data["speed"],
                                            dx=data["spacing"], order=1))
    return call(path, "travel_time", **data)["travel"]


def measure(path, data):
    solve(path, data)
    times, peaks = [], []
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        result = solve(path, data)
        times.append(time.perf_counter() - start)
        if path == "judge":
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peak = max(0, after - before)
        peaks.append(peak / (1024 ** 2))
    method = ("tracemalloc peak allocated during call; excludes untracked native allocations"
              if path == "judge" else
              "RUSAGE_CHILDREN ru_maxrss high-water delta (macOS bytes); "
              "may be zero after warmup and excludes GPU allocations")
    return {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
            "mem_method": method}, result


def main():
    results = {"judge": {}, "metal": {}}
    medium = {}
    for name, params in SIZES.items():
        data = inputs(params)
        for path, path_results in results.items():
            entry, output = measure(path, data)
            path_results[name] = entry
            if name == "medium":
                medium[path] = output
            print(f"{path:5s} {name:6s}: {entry['wall_s']:.3f} s", flush=True)
    reference = medium["judge"].astype(np.float64)
    difference = np.abs(medium["metal"].astype(np.float64) - reference)
    max_ref = np.max(np.abs(reference))
    rel = float(np.max(difference) / max_ref) if max_ref else float(np.max(difference))
    report = {
        "project": "25-fast-sweeping", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": {"metal": rel},
        "notes": "Times include harness serialization, process startup and shader compilation "
                 "for Metal. Relative difference is max absolute error normalized by maximum "
                 "absolute judge travel time. The Metal memory metric is a child high-water "
                 "increase, not an individual process peak; warmup can make it zero. "
                 "Judge tracemalloc may miss native allocations. Grid sizes and timings are "
                 "estimates pending execution on the target machine.",
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
