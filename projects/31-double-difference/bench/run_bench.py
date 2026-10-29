"""Benchmark the damped double-difference relocation paths."""
import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"events": 12, "stations": 8, "pairs": 66},
    "medium": {"events": 24, "stations": 8, "pairs": 276},
    "large": {"events": 48, "stations": 8, "pairs": 1128},
}


def make_inputs(size):
    rng = np.random.default_rng(3101)
    n, s = size["events"], size["stations"]
    x = np.linspace(-3, 3, n)
    truth = np.column_stack((x, np.zeros(n), 5 + 0.25 * x))
    origins = rng.uniform(-0.4, 0.4, n)
    angles = np.arange(s) * 2 * np.pi / s + 0.15
    stations = np.column_stack((9 * np.cos(angles), 9 * np.sin(angles), np.zeros(s)))
    initial = truth + rng.normal(0, [0.18, 0.8, 0.15], (n, 3))
    initial_origins = origins + rng.normal(0, 0.035, n)
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    arrivals = origins[:, None] + np.linalg.norm(truth[:, None] - stations[None], axis=2) / 6.0
    observed = arrivals[pairs[:, 0]] - arrivals[pairs[:, 1]]
    observed += rng.normal(0, 0.002, observed.shape)
    return {"stations": stations.astype(np.float32),
            "initial_locations": initial.astype(np.float32),
            "initial_origins": initial_origins.astype(np.float32),
            "pairs": pairs, "observed": observed.astype(np.float32),
            "velocity": 6.0, "damping": 1e-4}


def reference(stations, initial_locations, initial_origins, pairs, observed, velocity, damping):
    """Same float64 numerical-Jacobian TRF objective as judge/make_goldens.py."""
    stations = np.asarray(stations, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    n = len(initial_locations)
    x0 = np.column_stack((initial_locations, initial_origins)).astype(np.float64).ravel()

    def residual(x):
        events = x.reshape(n, 4)
        t = events[:, 3, None] + np.linalg.norm(
            events[:, None, :3] - stations[None], axis=2) / velocity
        dd = t[pairs[:, 0]] - t[pairs[:, 1]] - observed
        delta = (x - x0).reshape(n, 4).copy()
        delta[:, :3] /= velocity
        return np.concatenate((dd.ravel(), np.sqrt(damping) * delta.ravel()))

    result = least_squares(residual, x0, method="trf", ftol=1e-13, xtol=1e-13,
                           gtol=1e-13, max_nfev=1000)
    if not result.success or np.linalg.norm(result.grad, ord=np.inf) > 1e-8:
        raise RuntimeError(f"Reference did not converge: {result.message}")
    fit = result.x.reshape(n, 4)
    return {"locations": fit[:, :3], "origins": fit[:, 3]}


def benchmark(path, inputs):
    if path == "mlx":
        import mlx.core as mx

    def run():
        return reference(**inputs) if path == "judge" else call(path, "relocate", **inputs)

    run()
    times, peaks, rss = [], [], []
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
            tracemalloc.reset_peak()
        else:
            mx.reset_peak_memory()
        start = time.perf_counter()
        out = run()
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        else:
            peaks.append(mx.get_peak_memory() / 1e6)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    method = ("tracemalloc peak traced Python/NumPy allocations" if path == "judge" else
              "MLX reset_peak_memory/get_peak_memory; process lifetime peak RSS also reported")
    result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
              "mem_method": method}
    if rss:
        result["process_peak_rss_mb"] = max(rss)
    return result, out


def main():
    results = {"judge": {}, "mlx": {}}
    medium = {}
    for size_name, size in SIZES.items():
        inputs = make_inputs(size)
        for path, path_results in results.items():
            if path == "judge" and size_name == "large":
                path_results[size_name] = {"skipped": "Dense numerical-Jacobian reference may exceed 120 s."}
                continue
            print(f"Benchmarking {path} / {size_name}", flush=True)
            path_results[size_name], out = benchmark(path, inputs)
            if size_name == "medium":
                medium[path] = out
    errors = []
    for key in ("locations", "origins"):
        ref = medium["judge"][key]
        denom = np.max(np.abs(ref))
        errors.append(float(np.max(np.abs(medium["mlx"][key] - ref)) / max(denom, 1e-30)))
    report = {"project": "31-double-difference", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": {"mlx": max(errors)},
              "notes": "All unordered pairs, 0.002 s independent noise, seed 3101; "
                       "median of three runs after one warmup. Relative difference is the "
                       "larger of the maximum absolute errors normalized by the maximum "
                       "reference magnitude separately for locations and origins. Memory "
                       "peaks are maxima across repeats; RSS is a lifetime high-water mark. "
                       "Timings include host conversion. Large size is an unmeasured estimate."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
