"""Benchmark the prism forward model; run from the project directory."""

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
    "small": {"observations": 1024, "prisms": 128},
    "medium": {"observations": 4096, "prisms": 256},
    "large": {"observations": 16384, "prisms": 1024},
}
KEYS = ("b_e", "b_n", "b_u", "total_field")
MB = 1024 * 1024


def inputs(size):
    rng = np.random.default_rng(20261028)
    n, p = size["observations"], size["prisms"]
    coordinates = np.column_stack((rng.uniform(-1000, 1000, n),
                                   rng.uniform(-1000, 1000, n),
                                   rng.uniform(20, 120, n)))
    x = rng.uniform(-700, 700, p)
    y = rng.uniform(-700, 700, p)
    z = rng.uniform(-450, -180, p)
    prisms = np.column_stack((x - 60, x + 60, y - 60, y + 60, z - 80, z + 80))
    magnetization = rng.uniform(-2, 2, (p, 3))
    return {"coordinates": coordinates.astype(np.float32),
            "prisms": prisms.astype(np.float32),
            "magnetization": magnetization.astype(np.float32),
            "inducing_field": np.array([0.36, 0.48, -0.8], dtype=np.float32)}


def reference(args):
    import harmonica as hm

    xyz, p, m, direction = (args[k].astype(np.float64) for k in
                             ("coordinates", "prisms", "magnetization", "inducing_field"))
    b = np.stack(hm.prism_magnetic(xyz.T, p, m.T, field="b", parallel=False,
                                   dtype=np.float64), axis=-1)
    return {"b_e": b[:, 0], "b_n": b[:, 1], "b_u": b[:, 2],
            "total_field": b @ direction}


def metal_worker(size_name):
    # This process has no previous children, so its cumulative child high-water
    # mark starts at zero and describes this one release-binary invocation.
    args = inputs(SIZES[size_name])
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    call("metal", "prism_magnetic", **args)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": (after - before) / MB}))


def run_metal(size_name):
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                           "--metal-worker", size_name], check=True,
                          capture_output=True, text=True)
    return json.loads(proc.stdout)


def measure(path, size_name):
    args = inputs(SIZES[size_name])
    if path == "metal":
        run_metal(size_name)  # warm-up, excluded from the samples
        samples = [run_metal(size_name) for _ in range(3)]
        return {"wall_s": statistics.median(s["wall_s"] for s in samples),
                "peak_mem_mb": max(s["peak_mem_mb"] for s in samples),
                "mem_method": "fresh worker RUSAGE_CHILDREN ru_maxrss delta; macOS bytes; maximum of repeats"}

    if path == "mlx":
        import mlx.core as mx

        invoke = lambda: call("mlx", "prism_magnetic", **args)
    else:
        invoke = lambda: reference(args)
    invoke()
    times, peaks = [], []
    rss = []
    for _ in range(3):
        if path == "mlx":
            mx.reset_peak_memory()
        else:
            tracemalloc.start()
            tracemalloc.reset_peak()
        start = time.perf_counter()
        invoke()
        times.append(time.perf_counter() - start)
        if path == "mlx":
            peaks.append(mx.get_peak_memory() / MB)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB)
        else:
            peaks.append(tracemalloc.get_traced_memory()[1] / MB)
            tracemalloc.stop()
    result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks)}
    if path == "mlx":
        result["mem_method"] = "MLX reset_peak_memory/get_peak_memory; maximum of repeats"
        result["process_peak_rss_mb"] = max(rss)
        result["process_rss_method"] = "RUSAGE_SELF ru_maxrss, macOS bytes; cumulative process high-water mark"
    else:
        result["mem_method"] = "tracemalloc peak during call; maximum of repeats"
    return result


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for path, path_results in results.items():
        for size_name in SIZES:
            if path == "judge" and size_name == "large":
                path_results[size_name] = {"skipped": "16.8 million CPU prism-observation pairs may exceed 120 seconds"}
                continue
            print(f"Benchmarking {path} {size_name}...", flush=True)
            path_results[size_name] = measure(path, size_name)

    args = inputs(SIZES["medium"])
    expected = reference(args)
    differences = {}
    for path in ("mlx", "metal"):
        actual = call(path, "prism_magnetic", **args)
        differences[path] = max(float(np.max(np.abs(actual[k].astype(np.float64) - expected[k])))
                                / max(float(np.max(np.abs(expected[k]))), 1e-30)
                                for k in KEYS)
    report = {"project": "03-prism-magnetic", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": differences,
              "notes": "Seed 20261028; one warm-up and three timed calls, median wall time. Metal timing includes binary startup, compilation, and file I/O, but excludes Python worker startup. Memory is the maximum over repeats, in MiB; RSS excludes GPU allocations not reflected in process RSS. Relative difference is the worst field-wise max absolute error divided by that reference field's max absolute value. Large sizes are estimates and have not been timed on M5 Max."}
    destination = PROJECT / "bench" / "results.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-worker":
        metal_worker(sys.argv[2])
    else:
        main()
