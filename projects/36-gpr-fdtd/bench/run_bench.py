"""Benchmark the periodic 2-D Yee propagator."""
import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
sys.path.insert(0, str(PROJECT / "judge"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)

import mlx.core as mx
from outcrop_harness import call
from make_goldens import solve

SIZES = {
    "small": {"nx": 256, "nz": 256, "steps": 240},
    "medium": {"nx": 512, "nz": 512, "steps": 350},
    "large": {"nx": 1536, "nz": 1536, "steps": 700},
}
C = 299792458.0


def inputs(size):
    rng = np.random.default_rng(3601)
    shape = (size["nx"], size["nz"])
    return {
        "epsilon_r": rng.uniform(1.0, 5.0, shape).astype(np.float32),
        "electric": rng.normal(0, 0.1, shape).astype(np.float32),
        "hx": rng.normal(0, 0.1, shape).astype(np.float32),
        "hz": rng.normal(0, 0.1, shape).astype(np.float32),
        "dx": 0.05,
        "dz": 0.05,
        "dt": 0.4 * 0.05 / C,
        "steps": size["steps"],
    }


def run(path, args):
    if path == "judge":
        values = solve(args["epsilon_r"].astype(np.float64), args["electric"],
                       args["hx"], args["hz"], args["dx"], args["dz"],
                       args["dt"], args["steps"])
        return dict(zip(("electric", "hx", "hz"), values))
    return call(path, "propagate", **args)


def benchmark(path, args):
    run(path, args)
    times, peaks, rss = [], [], []
    output = None
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
        elif path == "mlx":
            mx.reset_peak_memory()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        output = run(path, args)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory() / 1e6)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, after - before) / 1e6)
    methods = {
        "judge": "tracemalloc peak Python/NumPy allocations during call (MB)",
        "mlx": "MLX peak allocated memory after reset (MB); process RSS high-water mark separately",
        "metal": "delta of RUSAGE_CHILDREN ru_maxrss high-water mark (macOS bytes, MB); may be zero after an earlier larger child",
    }
    result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
              "mem_method": methods[path]}
    if rss:
        result["process_peak_rss_mb"] = max(rss)
    return result, output


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium = {}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            if path == "judge" and name == "large":
                path_results[name] = {"skipped": "Large float64 sparse reference may exceed 120 seconds; medium provides the parity check."}
                continue
            path_results[name], out = benchmark(path, args)
            if name == "medium":
                medium[path] = out
            print(f"{path} {name}: {path_results[name]}", flush=True)
    diffs = {}
    for path in ("mlx", "metal"):
        diffs[path] = max(
            float(np.max(np.abs(medium[path][key].astype(np.float64) - medium["judge"][key]))
                  / max(np.max(np.abs(medium["judge"][key])), 1e-30))
            for key in ("electric", "hx", "hz")
        )
    report = {
        "project": "36-gpr-fdtd", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": diffs,
        "notes": "Seed 3601; dx=dz=0.05 m, dt=0.4*dx/c. Median of three calls after one warmup; includes harness conversion, file I/O and Metal process startup. Relative error is the largest field-wise max-norm error. Memory is decimal MB; child ru_maxrss is a cumulative high-water mark, so its delta is not a reliable per-call peak. MLX RSS is the process lifetime high-water mark, not an incremental allocation. Timing and size estimates are unverified on M5 Max.",
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
