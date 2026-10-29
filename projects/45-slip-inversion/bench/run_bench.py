"""Benchmark the supplied-matrix slip inversion; run from the project folder."""

import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)

from outcrop_harness import call

SIZES = {
    "small": {"npatch": 224, "nobs": 448, "nreg": 224},
    "medium": {"npatch": 512, "nobs": 1024, "nreg": 512},
    "large": {"npatch": 1024, "nobs": 2048, "nreg": 1024},
}


def make_problem(npatch, nobs, nreg):
    rng = np.random.default_rng(450101 + npatch)
    green = (rng.standard_normal((nobs, npatch)) * (0.02 / np.sqrt(npatch))).astype(np.float32)
    truth = rng.uniform(-0.5, 1.5, npatch).astype(np.float32)
    sigma = np.full(nobs, 0.002, dtype=np.float32)
    data = (green @ truth + rng.normal(0, 0.002, nobs)).astype(np.float32)
    regularizer = np.eye(nreg, npatch, dtype=np.float32)
    return {"green": green, "data": data, "sigma": sigma, "regularizer": regularizer,
            "damping": 0.5, "areas": np.full(npatch, 1e6, dtype=np.float32),
            "shear_modulus": 30e9}


def reference(green, data, sigma, regularizer, damping, areas, shear_modulus):
    # Same float64 augmented SVD solve as judge/make_goldens.py.
    g, d, sig, reg = (np.asarray(x, dtype=np.float64) for x in
                      (green, data, sigma, regularizer))
    augmented = np.vstack((g / sig[:, None], damping * reg))
    rhs = np.concatenate((d / sig, np.zeros(reg.shape[0])))
    slip, _, _, _ = lstsq(augmented, rhs, lapack_driver="gelsd")
    return {"slip": slip, "predicted": g @ slip,
            "moment": np.asarray(shear_modulus * (areas.astype(np.float64) @ np.abs(slip)))}


def run(path, problem):
    if path == "judge":
        return reference(**problem)
    return call(path, "invert_slip", **problem)


def benchmark(path, problem):
    import mlx.core as mx

    run(path, problem)  # Harness converts outputs to NumPy and synchronizes MLX.
    times, peaks, rss = [], [], []
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
            tracemalloc.reset_peak()
        else:
            mx.reset_peak_memory()
        start = time.perf_counter()
        result = run(path, problem)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        else:
            peaks.append(mx.get_peak_memory() / 1e6)
            # macOS ru_maxrss is bytes and is a process lifetime high-water mark.
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    method = ("tracemalloc peak during call; native allocations may be omitted" if path == "judge"
              else "MLX peak allocation after reset; process RSS is lifetime high-water mark")
    record = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
              "mem_method": method}
    if rss:
        record["process_peak_rss_mb"] = max(rss)
    return record, result


def relative_difference(actual, expected):
    errors = []
    for key in expected:
        a = np.asarray(actual[key], dtype=np.float64)
        b = np.asarray(expected[key], dtype=np.float64)
        scale = np.max(np.abs(b))
        errors.append(float(np.max(np.abs(a - b)) / scale if scale else np.max(np.abs(a - b))))
    return max(errors)


def main():
    results = {"judge": {}, "mlx": {}}
    medium = {}
    for name, dimensions in SIZES.items():
        problem = make_problem(**dimensions)
        for path, path_results in results.items():
            if path == "judge" and name == "large":
                path_results[name] = {"skipped": "Float64 SVD may exceed 120 seconds at this size."}
                continue
            print(f"Benchmarking {path} / {name}", flush=True)
            path_results[name], output = benchmark(path, problem)
            if name == "medium":
                medium[path] = output
    report = {"project": "45-slip-inversion", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": {"mlx": relative_difference(medium["mlx"], medium["judge"])},
              "notes": "Fixed-seed synthetic full-rank weighted inverse; one warmup, median of three calls. "
                       "Relative difference is the largest per-output max absolute error divided by "
                       "that output's maximum absolute reference value. MLX uses CPU QR; its peak "
                       "allocator statistic can exclude CPU allocations. MB uses 1e6 bytes."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
