"""Benchmark the locked 1-D MT implementations on deterministic inputs."""
import json
import os
import resource
import statistics
import subprocess
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
    "small": {"layers": 16, "frequencies": 20480},
    "medium": {"layers": 32, "frequencies": 131072},
    "large": {"layers": 64, "frequencies": 524288},
}
MU0 = 1.25663706212e-6
MB = 1024 ** 2


def inputs(size):
    rng = np.random.default_rng(3201)
    layers, count = size["layers"], size["frequencies"]
    return {
        "resistivity": np.exp(rng.uniform(np.log(10), np.log(1000), layers)).astype(np.float32),
        "thickness": np.exp(rng.uniform(np.log(20), np.log(500), layers - 1)).astype(np.float32),
        "frequency": np.logspace(-3, 3, count, dtype=np.float32),
    }


def reference(data):
    from simpeg.electromagnetics.natural_source import Simulation1DRecursive

    rho = data["resistivity"].astype(np.float64)
    depth = data["thickness"].astype(np.float64)
    freq = data["frequency"].astype(np.float64)
    z = -Simulation1DRecursive()._get_recursive_impedances(
        freq, depth[::-1], (1.0 / rho)[::-1]
    )
    return {"apparent_resistivity": np.abs(z) ** 2 / (MU0 * 2 * np.pi * freq),
            "phase": np.degrees(np.angle(z))}


def metal_worker(size_name):
    # A fresh parent gives each child a fresh RUSAGE_CHILDREN high-water mark.
    data = inputs(SIZES[size_name])
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    call("metal", "mt1d", **data)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": max(0, after - before) / MB}))


def run_metal(size_name):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--metal-worker", size_name],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def benchmark(path, size_name, data):
    if path == "metal":
        run_metal(size_name)
        samples = [run_metal(size_name) for _ in range(3)]
        return {"wall_s": statistics.median(s["wall_s"] for s in samples),
                "peak_mem_mb": max(s["peak_mem_mb"] for s in samples),
                "mem_method": "fresh parent RUSAGE_CHILDREN ru_maxrss delta (macOS bytes); maximum of repeats"}

    if path == "mlx":
        import mlx.core as mx
        execute = lambda: call("mlx", "mt1d", **data)
    else:
        execute = lambda: reference(data)
    execute()
    times, peaks, rss = [], [], []
    for _ in range(3):
        if path == "mlx":
            mx.reset_peak_memory()
        else:
            tracemalloc.start()
            tracemalloc.reset_peak()
        start = time.perf_counter()
        out = execute()
        times.append(time.perf_counter() - start)
        if path == "mlx":
            peaks.append(mx.get_peak_memory() / MB)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB)
        else:
            peaks.append(tracemalloc.get_traced_memory()[1] / MB)
            tracemalloc.stop()
        del out
    result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks)}
    if path == "mlx":
        result.update(mem_method="MLX allocator peak after reset; maximum of repeats",
                      process_peak_rss_mb=max(rss))
    else:
        result["mem_method"] = "tracemalloc peak per call, including NumPy tracked allocations; maximum of repeats"
    return result


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for size_name, size in SIZES.items():
        data = inputs(size)
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {size_name}", flush=True)
            path_results[size_name] = benchmark(path, size_name, data)
    data = inputs(SIZES["medium"])
    expected = reference(data)
    differences = {}
    for path in ("mlx", "metal"):
        actual = call(path, "mt1d", **data)
        differences[path] = max(float(np.max(np.abs(actual[k].astype(np.float64) - expected[k]) /
                                             np.abs(expected[k]))) for k in expected)
    report = {"project": "32-mt-1d", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": differences,
              "notes": "Seed 3201; frequencies log-spaced 0.001–1000 Hz; layer count includes half-space. "
                       "Wall time includes output materialization; Metal includes process startup, shader compilation "
                       "and harness file I/O, but excludes its fresh measurement-parent startup. "
                       "MLX RSS is process lifetime high-water, not an incremental allocation. "
                       "Memory is MiB. Large size runtimes are estimates until measured."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-worker":
        metal_worker(sys.argv[2])
    else:
        main()
