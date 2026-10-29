"""Benchmark the coupled convection API from the project directory."""
from __future__ import annotations

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
    "small": {"n": 32, "steps": 30, "dt": 1.0e-5, "rayleigh": 10000.0},
    "medium": {"n": 64, "steps": 250, "dt": 1.0e-5, "rayleigh": 10000.0},
    "large": {"n": 128, "steps": 1500, "dt": 1.0e-5, "rayleigh": 10000.0},
}


def load_judge():
    spec = importlib.util.spec_from_file_location("convection_judge", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(params):
    n = params["n"]
    axis = np.linspace(0.0, 1.0, n + 1)
    x, z = np.meshgrid(axis, axis)
    rng = np.random.default_rng(481989)
    t = 1 - z + 0.08 * np.cos(np.pi * x) * np.sin(np.pi * z)
    for k, ell in ((2, 1), (1, 2), (3, 2)):
        t += rng.uniform(-0.015, 0.015) * np.cos(k * np.pi * x) * np.sin(ell * np.pi * z)
    t = np.asarray(t, dtype=np.float32, order="C")
    t[0, :], t[-1, :] = 1.0, 0.0
    return {"temperature": t, "rayleigh": params["rayleigh"],
            "dt": params["dt"], "steps": params["steps"]}


def max_relative_difference(actual, reference):
    errors = []
    for key, expected in reference.items():
        a = np.asarray(actual[key], dtype=np.float64)
        b = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(b)))
        error = float(np.max(np.abs(a - b)))
        errors.append(error / scale if scale else error)
    return max(errors)


def main():
    import mlx.core as mx

    judge = load_judge()
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium_outputs = {}
    for name, params in SIZES.items():
        kwargs = inputs(params)
        for path, path_results in results.items():
            def run(path=path, kwargs=kwargs):
                if path == "judge":
                    return judge.convection(**kwargs)
                return call(path, "convection", **kwargs)

            print(f"{path} {name}: warming up", flush=True)
            run()
            times, peaks = [], []
            for _ in range(3):
                if path == "mlx":
                    mx.reset_peak_memory()
                if path == "metal":
                    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                start = time.perf_counter()
                output = run()
                times.append(time.perf_counter() - start)
                if path == "mlx":
                    peaks.append(mx.get_peak_memory())
                elif path == "metal":
                    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                    peaks.append(max(0, after - before))
            result = {"wall_s": statistics.median(times)}
            if path == "judge":
                tracemalloc.start()
                run()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                result.update(peak_mem_mb=peak / 1024**2,
                              mem_method="tracemalloc peak, separate untimed call (MiB)")
            elif path == "mlx":
                result.update(peak_mem_mb=max(peaks) / 1024**2,
                              process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                              mem_method="MLX reset_peak_memory/get_peak_memory; process lifetime peak RSS also reported (MiB, macOS)")
            else:
                result.update(peak_mem_mb=max(peaks) / 1024**2,
                              mem_method="RUSAGE_CHILDREN ru_maxrss high-water delta (MiB, macOS)")
            path_results[name] = result
            if name == "medium":
                medium_outputs[path] = output
            print(f"{path} {name}: {result['wall_s']:.3f} s", flush=True)
    diffs = {path: max_relative_difference(medium_outputs[path], medium_outputs["judge"])
             for path in ("mlx", "metal")}
    payload = {"project": "48-mantle-convection", "sizes": SIZES,
               "results": results, "medium_max_rel_diff": diffs,
               "notes": "Wall times are medians of three calls after one warm-up, including harness conversion and Metal process startup. N is the number of intervals per dimension. Relative differences are the maximum across outputs of max absolute error divided by each reference output's maximum absolute value (absolute error for zero reference). Memory is in MiB. macOS child ru_maxrss is a cumulative high-water mark, so its delta after warm-up can be zero and is not a per-call allocation peak; GPU allocations may not be represented. MLX peak is its device allocator high-water mark and process RSS is a lifetime high-water mark."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
