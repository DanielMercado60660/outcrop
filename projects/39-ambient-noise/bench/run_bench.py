"""Benchmark the ambient-noise paths from the project root."""
import ast
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
    "small": {"windows": 128, "samples": 1024},
    "medium": {"windows": 512, "samples": 4096},
    "large": {"windows": 2048, "samples": 8192},
}
KEYS = ("whitened_a", "whitened_b", "correlations", "stack")


def inputs(size):
    rng = np.random.default_rng(39001)
    shape = (size["windows"], size["samples"])
    return {"a": rng.standard_normal(shape, dtype=np.float32),
            "b": rng.standard_normal(shape, dtype=np.float32),
            "dt": 0.01, "low": 3.0, "high": 25.0, "eps": 1e-10}


def load_reference():
    # Importing make_goldens would generate and write the golden data.
    source = (PROJECT / "judge" / "make_goldens.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reference")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PROJECT / "judge" / "make_goldens.py"), "exec"), namespace)  # noqa: S102
    return namespace["reference"]


REFERENCE = load_reference()


def judge(args):
    return dict(zip(KEYS, REFERENCE(**args)))


def metal_worker(size_name):
    args = inputs(SIZES[size_name])
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    result = call("metal", "ambient_noise", **args)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss - before
    assert set(result) == set(KEYS)
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": peak / 1024**2}))


def run_metal(size_name):
    # Each worker has a fresh RUSAGE_CHILDREN high-water mark.
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--metal-worker", size_name],
                          check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def benchmark(path, size_name, args):
    if path == "metal":
        run_metal(size_name)
        records = [run_metal(size_name) for _ in range(3)]
        return {"wall_s": statistics.median(r["wall_s"] for r in records),
                "peak_mem_mb": max(r["peak_mem_mb"] for r in records),
                "mem_method": "fresh worker RUSAGE_CHILDREN ru_maxrss delta (macOS bytes); binary RSS"}

    if path == "mlx":
        import mlx.core as mx
        call("mlx", "ambient_noise", **args)
        times, peaks, rss = [], [], []
        for _ in range(3):
            mx.reset_peak_memory()
            start = time.perf_counter()
            output = call("mlx", "ambient_noise", **args)
            times.append(time.perf_counter() - start)
            peaks.append(mx.get_peak_memory() / 1024**2)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
            del output
        return {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
                "process_peak_rss_mb": max(rss),
                "mem_method": "MLX reset_peak_memory/get_peak_memory; process lifetime peak RSS separately (macOS bytes)"}

    judge(args)
    times, peaks = [], []
    for _ in range(3):
        tracemalloc.start()
        start = time.perf_counter()
        output = judge(args)
        times.append(time.perf_counter() - start)
        peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
        tracemalloc.stop()
        del output
    return {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
            "mem_method": "tracemalloc peak during reference call, excluding prebuilt inputs"}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-worker":
        metal_worker(sys.argv[2])
        return
    results = {path: {} for path in ("judge", "mlx", "metal")}
    differences = {}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {name}", flush=True)
            path_results[name] = benchmark(path, name, args)
        if name == "medium":
            ref = judge(args)
            for path in ("mlx", "metal"):
                actual = call(path, "ambient_noise", **args)
                errors = []
                for key in KEYS:
                    scale = float(np.max(np.abs(ref[key])))
                    error = float(np.max(np.abs(actual[key] - ref[key])))
                    errors.append(error / scale if scale else error)
                differences[path] = max(errors)
    report = {"project": "39-ambient-noise", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": differences,
              "notes": "Seed 39001, dt=0.01 s, band 3–25 Hz, eps=1e-10. Median of three calls after one warmup; memory is the maximum of three peaks. Inputs are prebuilt. Metal timing includes harness file I/O, process startup and shader compilation. Relative error is the largest per-output peak-normalized absolute error. MLX RSS is a lifetime high-water mark. Metal RSS excludes separate GPU allocations. RSS units assume macOS."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
