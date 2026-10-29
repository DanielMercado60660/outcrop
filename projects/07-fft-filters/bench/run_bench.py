"""Benchmark upward continuation against the Harmonica reference."""

import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import harmonica as hm
import mlx.core as mx
import numpy as np
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)

from outcrop_harness import call

SIZES = {
    "small": {"ny": 768, "nx": 768},
    "medium": {"ny": 2048, "nx": 2048},
    "large": {"ny": 8192, "nx": 8192},
}
DX = 100.0
DY = 100.0
HEIGHT = 250.0
MB = 1024 ** 2


def inputs(size):
    rng = np.random.default_rng(7142026)
    grid = rng.standard_normal((size["ny"], size["nx"]), dtype=np.float32)
    return {"grid": grid, "dx": DX, "dy": DY, "operation": "upward",
            "height": HEIGHT, "inclination": 50.0, "declination": -23.0,
            "magnetization_inclination": 50.0, "magnetization_declination": -23.0}


def reference(args):
    grid = args["grid"]
    da = xr.DataArray(
        grid.astype(np.float64), dims=("northing", "easting"),
        coords={"northing": np.arange(grid.shape[0]) * args["dy"],
                "easting": np.arange(grid.shape[1]) * args["dx"]},
    )
    return hm.upward_continuation(da, args["height"]).values


def run(path, args):
    if path == "judge":
        return reference(args)
    return call(path, "filter_grid", **args)["result"]


def benchmark(path, args):
    run(path, args)  # Warm up FFTs, imports, and device initialization.
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = run(path, args)
        times.append(time.perf_counter() - start)
        del result

    if path == "judge":
        tracemalloc.start()
        result = run(path, args)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        method = "tracemalloc peak during a separate call, including NumPy allocations"
        extra = {}
    else:
        mx.reset_peak_memory()
        result = run(path, args)
        peak = mx.get_peak_memory()
        method = "MLX peak active memory during a separate call"
        # macOS ru_maxrss is bytes and is the process lifetime high-water mark.
        extra = {"process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB}
    del result
    return {"wall_s": statistics.median(times), "peak_mem_mb": peak / MB,
            "mem_method": method, **extra}


def main():
    results = {"judge": {}, "mlx": {}}
    differences = {}
    for size_name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            print(f"Benchmarking {path} {size_name} {size}", flush=True)
            path_results[size_name] = benchmark(path, args)
        if size_name == "medium":
            expected = reference(args)
            actual = run("mlx", args)
            denominator = float(np.max(np.abs(expected)))
            differences["mlx"] = float(np.max(np.abs(actual - expected)) / denominator)
            del expected, actual

    output = {
        "project": "07-fft-filters", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": ("Upward continuation, dx=dy=100 m, height=250 m; fixed-seed "
                  "float32 normal input. Median of three calls after one warm-up; "
                  "each MLX call includes conversion of the output to NumPy. "
                  "Memory is measured in a separate call; process_peak_rss_mb is "
                  "the macOS process lifetime high-water RSS, not a per-call delta. "
                  "Relative difference is max absolute error divided by the "
                  "maximum absolute reference value over the full grid."),
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
