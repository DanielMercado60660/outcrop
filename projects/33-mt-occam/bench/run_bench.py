"""Benchmark the MT objective and gradient; run from the project directory."""
from __future__ import annotations

import json
import os
import resource
import runpy
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

JUDGE = runpy.run_path(str(ROOT / "judge" / "make_goldens.py"))
SIZES = {
    "small": {"layers": 16, "frequencies": 8192},
    "medium": {"layers": 48, "frequencies": 32768},
    "large": {"layers": 128, "frequencies": 131072},
}


def inputs(size):
    rng = np.random.default_rng(330101)
    layers, count = size["layers"], size["frequencies"]
    depth = np.linspace(0, 1, layers)
    model = (2.0 + 0.35 * np.sin(2 * np.pi * depth)).astype(np.float32)
    thickness = np.geomspace(30, 300, layers - 1).astype(np.float32)
    frequencies = np.geomspace(1e-3, 1e4, count).astype(np.float32)
    truth = (model + 0.12 * np.cos(np.pi * depth)).astype(np.float32)
    clean = JUDGE["forward"](truth, thickness, frequencies)
    sigma = np.broadcast_to(np.array([[0.03], [0.02]], dtype=np.float32),
                            (2, count)).copy()
    observed = (clean + rng.normal(size=clean.shape) * sigma * 0.5).astype(np.float32)
    return {"model": model, "thickness": thickness, "frequencies": frequencies,
            "observed": observed, "sigma": sigma, "alpha": 0.7}


def evaluate(path, kw):
    if path == "mlx":
        return call("mlx", "objective_and_grad", **kw)
    value, gradient = JUDGE["objective"](**kw)
    return {"objective": value, "gradient": gradient}


def measure(path, kw):
    evaluate(path, kw)
    times, peaks, rss = [], [], []
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
        else:
            mx.reset_peak_memory()
        start = time.perf_counter()
        evaluate(path, kw)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 2**20)
            tracemalloc.stop()
        else:
            peaks.append(mx.get_peak_memory() / 2**20)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
              "mem_method": ("tracemalloc peak during call, MiB" if path == "judge"
                             else "MLX reset_peak_memory/get_peak_memory, MiB")}
    if rss:
        result["process_peak_rss_mb"] = max(rss)
        result["process_rss_method"] = "RUSAGE_SELF ru_maxrss, macOS lifetime high-water mark, MiB"
    return result


def main():
    results = {"judge": {}, "mlx": {}}
    medium_diff = {}
    for name, size in SIZES.items():
        kw = inputs(size)
        for path in ("judge", "mlx"):
            if path == "judge" and name == "large":
                results[path][name] = {"skipped": "Centered finite-difference Jacobian scales quadratically with layer count and is expected to exceed 120 s at this size."}
                continue
            print(f"Benchmarking {path} / {name}", flush=True)
            results[path][name] = measure(path, kw)
        if name == "medium":
            ref, actual = evaluate("judge", kw), evaluate("mlx", kw)
            medium_diff["mlx"] = max(float(np.max(np.abs(np.asarray(actual[k], dtype=np.float64)
                - np.asarray(ref[k], dtype=np.float64)) /
                np.maximum(np.abs(np.asarray(ref[k], dtype=np.float64)), 1e-8)))
                for k in ("objective", "gradient"))
    payload = {"project": "33-mt-occam", "sizes": SIZES, "results": results,
               "medium_max_rel_diff": medium_diff,
               "notes": "Objective and gradient, alpha=0.7; deterministic float32 inputs, seed 330101. Times are median of three calls after one warm-up, including harness conversion; judge timing includes tracemalloc overhead. Peak is the maximum of three runs in MiB. MLX RSS is process-lifetime high-water, not per-call incremental memory. Medium relative difference is componentwise with denominator floor 1e-8. Large timing is an unverified estimate; frequencies are dense to scale the workload."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
