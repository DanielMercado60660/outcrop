"""Benchmark the finite-rectangle Okada solver through the shared harness."""
from __future__ import annotations

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
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
sys.path.insert(0, str(ROOT / "judge"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)

from outcrop_harness import call
from precision_audit import evaluate, load_reference

SIZES = {
    "small": {"n_observations": 512},
    "medium": {"n_observations": 2048},
    "large": {"n_observations": 131072},
}
SEED = 4401
MB = 1024 * 1024


def inputs(n):
    rng = np.random.default_rng(SEED)
    coordinates = np.zeros((n, 3), dtype=np.float32)
    coordinates[:, :2] = rng.uniform(-12.0, 12.0, (n, 2))
    return {"coordinates": coordinates, "source": "finite_rectangle", "depth": 6.0,
            "dip_deg": 50.0, "bounds": np.array([-3, 3, -2, 2], np.float32),
            "slip": np.array([1.0, 0.6, 0.2], np.float32), "alpha": 2.0 / 3.0}


def judge_call(ref, args):
    rows = evaluate(ref, args, np.float64)
    return {"displacement": rows[:, :3],
            "horizontal_derivatives": rows[:, 3:].reshape(-1, 3, 2)}


def max_relative(actual, reference):
    # A global scale per output avoids dividing by zeros in individual entries.
    return max(float(np.max(np.abs(actual[key] - reference[key])) /
                     max(float(np.max(np.abs(reference[key]))), 1e-30))
               for key in reference)


def main():
    # Avoid writing bytecode in read-only implementation and judge directories.
    ref = load_reference(ROOT.parents[1] / "data" / "okada" / "reviewer" / "dc3d.py")
    import mlx.core as mx

    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium_outputs = {}
    judge_seconds_per_observation = None
    for name, size in SIZES.items():
        args = inputs(size["n_observations"])
        for path, path_results in results.items():
            if (path == "judge" and judge_seconds_per_observation is not None
                    and judge_seconds_per_observation * size["n_observations"] > 120):
                path_results[name] = {"skipped": "Extrapolated reference time exceeds 120 s per call."}
                continue
            run = (lambda args=args: judge_call(ref, args)) if path == "judge" else (
                lambda path=path, args=args: call(path, "okada", **args))
            run()  # Warm up each path and size, including Metal compilation/startup.
            if path == "mlx":
                mx.reset_peak_memory()
            child_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            elapsed = []
            for _ in range(3):
                start = time.perf_counter()
                output = run()  # Harness converts MLX outputs to NumPy, synchronizing GPU work.
                elapsed.append(time.perf_counter() - start)
            wall = statistics.median(elapsed)
            if path == "judge":
                judge_seconds_per_observation = wall / size["n_observations"]
                tracemalloc.start()
                run()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem = peak / MB
                method = "tracemalloc peak during a separate reference call"
            elif path == "mlx":
                mem = mx.get_peak_memory() / MB
                method = "MLX allocator peak after reset; process peak RSS also reported"
            else:
                child_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                mem = max(0, child_after - child_before) / MB
                method = "RUSAGE_CHILDREN ru_maxrss high-water delta after warmup (macOS bytes)"
            result = {"wall_s": wall, "peak_mem_mb": mem, "mem_method": method}
            if path == "mlx":
                result["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB
            path_results[name] = result
            if name == "medium":
                medium_outputs[path] = output
            print(f"{path:5s} {name:6s}: {wall:.3f} s, {mem:.1f} MiB", flush=True)

    medium_args = inputs(SIZES["medium"]["n_observations"])
    if "judge" not in medium_outputs:
        # Retain a bounded sanity check if the full reference is too expensive.
        medium_args["coordinates"] = medium_args["coordinates"][:256]
        reference = judge_call(ref, medium_args)
        comparisons = {path: call(path, "okada", **medium_args) for path in ("mlx", "metal")}
        comparison_note = "Medium comparison uses the first 256 observations because the full judge was skipped."
    else:
        reference = medium_outputs["judge"]
        comparisons = {path: medium_outputs[path] for path in ("mlx", "metal")}
        comparison_note = "Medium comparison uses all observations."
    report = {
        "project": "44-okada", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": {path: max_relative(out, reference) for path, out in comparisons.items()},
        "notes": ("Surface observations uniformly sampled in [-12, 12] km with seed 4401; "
                  "depth 6 km, dip 50 degrees, bounds [-3, 3, -2, 2] km, "
                  "slip [1, 0.6, 0.2] m. Times include harness conversion and Metal "
                  "process startup, I/O and compilation. Memory is MiB. Metal's child "
                  "ru_maxrss is a cumulative high-water mark, so its delta can be zero "
                  "even when the child uses memory and excludes GPU allocations. "
                  "MLX process RSS is a cumulative high-water mark. Relative differences "
                  "use the maximum absolute reference value separately for each output. "
                  + comparison_note),
    }
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
