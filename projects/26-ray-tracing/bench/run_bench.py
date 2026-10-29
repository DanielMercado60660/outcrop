"""Benchmark layered ray bending against the scalar judge."""
import json
import math
import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
sys.path.insert(0, str(PROJECT / "judge"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call
from make_goldens import layered

SIZES = {
    "small": {"rays": 1000, "layers": 8, "batches": 1},
    "medium": {"rays": 10000, "layers": 8, "batches": 1},
    "large": {"rays": 100000, "layers": 8, "batches": 10},
}
DEPTH = 1600.0


def inputs(size):
    rng = np.random.default_rng(2601)
    return {
        "interfaces": np.arange(200, 1600, 200, dtype=np.float32),
        "velocities": np.array([1800, 2000, 2200, 2100, 2400, 2300, 2600, 2500], dtype=np.float32),
        "offsets": rng.uniform(-2000, 2000, size["rays"]).astype(np.float32),
        "depth": DEPTH,
    }


def reference(interfaces, velocities, offsets, depth):
    interfaces = interfaces.astype(np.float64)
    velocities = velocities.astype(np.float64)
    limit = 0.999 / np.max(velocities)
    times = np.empty(len(offsets), dtype=np.float64)
    angles = np.empty(len(offsets), dtype=np.float64)
    for i, offset in enumerate(offsets):
        p = brentq(lambda p, target=float(offset): layered(p, interfaces, velocities, depth)[0] - target,
                   -limit, limit, xtol=1e-16)
        _, times[i] = layered(p, interfaces, velocities, depth)
        angles[i] = math.asin(p * velocities[0])
    return {"endpoints": np.column_stack((offsets, np.full(len(offsets), depth))),
            "times": times, "angles": angles}


def run(path, kwargs, batches):
    result = None
    for _ in range(batches):
        result = reference(**kwargs) if path == "judge" else call(path, "bend_layered", **kwargs)
    return result


def measure(path, kwargs, batches):
    import mlx.core as mx

    run(path, kwargs, batches)
    if path == "mlx":
        mx.reset_peak_memory()
    elif path == "judge":
        tracemalloc.start()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        run(path, kwargs, batches)
        timings.append(time.perf_counter() - start)
    result = {"wall_s": float(np.median(timings))}
    if path == "judge":
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.update(peak_mem_mb=peak / 1024**2, mem_method="tracemalloc peak after warm-up")
    elif path == "mlx":
        result.update(peak_mem_mb=mx.get_peak_memory() / 1024**2,
                      mem_method="MLX peak allocation after reset; RSS is lifetime high-water")
        result["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    else:
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        result.update(peak_mem_mb=max(0, child_after - child_before) / 1024**2,
                      mem_method="macOS RUSAGE_CHILDREN ru_maxrss high-water delta after warm-up")
    return result


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for name, size in SIZES.items():
        kwargs = inputs(size)
        for path, path_results in results.items():
            if path == "judge" and name == "large":
                path_results[name] = {"skipped": "Scalar SciPy root solves for one million rays may exceed 120 s."}
            else:
                print(f"{path}: {name}", flush=True)
                path_results[name] = measure(path, kwargs, size["batches"])
    kwargs = inputs(SIZES["medium"])
    expected = reference(**kwargs)
    differences = {}
    for path in ("mlx", "metal"):
        actual = call(path, "bend_layered", **kwargs)
        differences[path] = max(float(np.max(np.abs(actual[k] - v)) /
                                      max(float(np.max(np.abs(v))), 1e-30))
                                for k, v in expected.items())
    report = {
        "project": "26-ray-tracing", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": "bend_layered; seed 2601, depth 1600 m, offsets uniform in [-2000, 2000] m. Wall times include batches and harness I/O. Memory in MiB. Child ru_maxrss is a cumulative high-water mark: its post-warm-up delta can be zero and is not the absolute peak. MLX RSS is the process lifetime peak. Differences normalize each field's maximum absolute error by its maximum reference magnitude.",
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
