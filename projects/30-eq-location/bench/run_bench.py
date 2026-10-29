"""Benchmark the profiled-origin-time earthquake grid search."""
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
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"events": 8, "stations": 32, "nx": 29, "ny": 29, "nz": 29},
    "medium": {"events": 24, "stations": 32, "nx": 49, "ny": 49, "nz": 49},
    "large": {"events": 64, "stations": 32, "nx": 69, "ny": 69, "nz": 69},
}


def make_inputs(size):
    rng = np.random.default_rng(30012026)
    n, b = size["stations"], size["events"]
    angle = np.arange(n) * (2 * np.pi / n)
    radius = np.tile([9000., 16000., 23000., 16000.], (n + 3) // 4)[:n]
    stations = np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                                -rng.uniform(0, 450, n))).astype(np.float32)
    sigma = np.linspace(0.015, 0.035, n, dtype=np.float32)
    truth = np.column_stack((rng.uniform(-4500, 4500, (b, 2)),
                             rng.uniform(4500, 10500, b), rng.uniform(-2, 2, b)))
    arrivals = (truth[:, 3, None] + np.linalg.norm(
        truth[:, None, :3] - stations[None, :, :], axis=2) / 6000.
        + rng.normal(size=(b, n)) * sigma).astype(np.float32)
    return {"stations": stations, "arrivals": arrivals, "sigma": sigma, "velocity": 6000.,
            "x": np.linspace(-6000, 6000, size["nx"], dtype=np.float32),
            "y": np.linspace(-6000, 6000, size["ny"], dtype=np.float32),
            "z": np.linspace(3000, 12000, size["nz"], dtype=np.float32)}


def load_reference():
    spec = importlib.util.spec_from_file_location("location_reference", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(path, inputs, ref):
    if path == "judge":
        return ref.grid(inputs["stations"].astype(np.float64),
                        inputs["arrivals"].astype(np.float64),
                        inputs["sigma"].astype(np.float64), inputs["velocity"],
                        [inputs[k].astype(np.float64) for k in ("x", "y", "z")])
    return call(path, "grid_search", **inputs)


def measure(path, inputs, ref):
    import mlx.core as mx

    run_one(path, inputs, ref)
    times, peaks, rss_values = [], [], []
    output = None
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
        elif path == "mlx":
            mx.reset_peak_memory()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        output = run_one(path, inputs, ref)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory() / 1024**2)
            rss_values.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, after - before) / 1024**2)
    methods = {"judge": "tracemalloc peak traced Python and NumPy allocations",
               "mlx": "MLX reset_peak_memory/get_peak_memory; process peak RSS also reported",
               "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta (macOS bytes)"}
    result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
              "mem_method": methods[path]}
    if rss_values:
        result["process_peak_rss_mb"] = max(rss_values)
    return result, output


def main():
    ref = load_reference()
    results = {path: {} for path in ("judge", "mlx", "metal")}
    differences = {}
    for size_name, size in SIZES.items():
        inputs = make_inputs(size)
        medium_outputs = {}
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {size_name}", flush=True)
            path_results[size_name], output = measure(path, inputs, ref)
            if size_name == "medium":
                medium_outputs[path] = output
        if size_name == "medium":
            expected = medium_outputs["judge"]
            for path in ("mlx", "metal"):
                differences[path] = max(
                    float(np.max(np.abs(medium_outputs[path][key] - expected[key])) /
                          max(np.max(np.abs(expected[key])), 1e-12))
                    for key in expected)
    report = {"project": "30-eq-location", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": differences,
              "notes": "Grid search; seed 30012026, one warm-up and median of three calls. Timings include conversion and Metal process startup and shader compilation. Relative difference is the maximum across output arrays of max absolute error divided by the reference array's max absolute value. Metal child RSS is a cumulative high-water mark: the delta can be zero when an earlier child had a larger peak and excludes GPU allocations. MLX peak is allocator-wide; process RSS is a lifetime high-water mark. Memory is in MiB. Large timing is an unmeasured estimate."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
