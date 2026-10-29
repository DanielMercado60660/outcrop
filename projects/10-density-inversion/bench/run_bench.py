"""Benchmark the density inversion paths; run from the project directory."""
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

import numpy as np
from scipy.linalg import cho_factor, cho_solve

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"shape": [4, 4, 2], "n_data": 36},
    "medium": {"shape": [8, 8, 4], "n_data": 144},
    "large": {"shape": [12, 12, 6], "n_data": 324},
}


def make_problem(spec):
    rng = np.random.default_rng(1010)
    nx, ny, nz = spec["shape"]
    h = 50.0
    x = (np.arange(nx) + 0.5 - nx / 2) * h
    y = (np.arange(ny) + 0.5 - ny / 2) * h
    z = (np.arange(nz) + 0.5 - nz) * h
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    centers = np.column_stack([a.ravel(order="F") for a in (xx, yy, zz)])
    side = int(np.sqrt(spec["n_data"]))
    rx, ry = np.meshgrid(np.linspace(-nx * h / 2, nx * h / 2, side),
                         np.linspace(-ny * h / 2, ny * h / 2, side))
    dx = rx.ravel()[:, None] - centers[:, 0]
    dy = ry.ravel()[:, None] - centers[:, 1]
    dz = 10.0 - centers[:, 2]
    g = (-6.6743e-11 * 1000 * h**3 * 1e5 * dz /
         (dx**2 + dy**2 + dz**2)**1.5).astype(np.float32)
    truth = np.zeros((nx, ny, nz), dtype=np.float32)
    truth[nx // 4:nx // 2, ny // 4:ny // 2, nz // 2:] = 0.5
    truth[nx // 2:3 * nx // 4, ny // 2:3 * ny // 4, nz // 2:] = -0.4
    clean = g @ truth.ravel(order="F")
    sigma = np.full(clean.shape, max(float(np.max(np.abs(clean))) * 0.03, 1e-5), dtype=np.float32)
    data = (clean + sigma * rng.standard_normal(clean.size)).astype(np.float32)
    return {"sensitivity": g, "data": data, "uncertainty": sigma,
            "reference": np.zeros(nx * ny * nz, dtype=np.float32),
            "shape": spec["shape"], "cell_size": h, "depth_offset": 25.0,
            "alpha_xyz": np.array([0.4, 0.4, 0.4], dtype=np.float32), "beta": 10.0}


def judge_invert(**p):
    # Same explicit precision and Cholesky solve as the judge's oracle.
    nx, ny, nz = p["shape"]
    h, offset = p["cell_size"], p["depth_offset"]
    depth = (nz - np.arange(nz) - 0.5) * h
    w = np.broadcast_to((h / 2 + offset) / (depth + offset), (nx, ny, nz)).ravel(order="F")
    q = EXPLICIT_PRECISION(p["shape"], w, p["alpha_xyz"])
    g = p["sensitivity"].astype(np.float64)
    a = g / p["uncertainty"][:, None]
    ref = p["reference"].astype(np.float64)
    rhs = a.T @ ((p["data"] - g @ ref) / p["uncertainty"])
    model = ref + cho_solve(cho_factor(a.T @ a + p["beta"] * q), rhs)
    return {"model": model, "predicted": g @ model}


def main():
    import mlx.core as mx

    global EXPLICIT_PRECISION
    EXPLICIT_PRECISION = runpy.run_path(str(PROJECT / "judge" / "make_goldens.py"))["explicit_precision"]
    results = {path: {} for path in ("judge", "mlx", "metal")}
    differences = {}
    for name, spec in SIZES.items():
        problem = make_problem(spec)
        outputs = {}
        for path, size_results in results.items():
            invoke = (lambda problem=problem: judge_invert(**problem)) if path == "judge" else (
                lambda path=path, problem=problem: call(path, "invert", **problem))
            invoke()
            times, peaks, rss = [], [], []
            for _ in range(3):
                if path == "judge":
                    tracemalloc.start()
                elif path == "mlx":
                    mx.reset_peak_memory()
                else:
                    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                start = time.perf_counter()
                out = invoke()
                times.append(time.perf_counter() - start)
                if path == "judge":
                    peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
                    tracemalloc.stop()
                elif path == "mlx":
                    peaks.append(mx.get_peak_memory() / 1024**2)
                    rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
                else:
                    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                    peaks.append(max(0, after - before) / 1024**2)
            methods = {"judge": "tracemalloc peak during call",
                       "mlx": "MLX reset_peak_memory/get_peak_memory",
                       "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta (macOS bytes)"}
            record = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
                      "mem_method": methods[path]}
            if rss:
                record["process_peak_rss_mb"] = max(rss)
            size_results[name] = record
            outputs[path] = out
            print(f"{path} {name}: {record}", flush=True)
        if name == "medium":
            for path in ("mlx", "metal"):
                differences[path] = max(
                    float(np.max(np.abs(outputs[path][key] - outputs["judge"][key])) /
                          max(float(np.max(np.abs(outputs["judge"][key]))), 1e-30))
                    for key in ("model", "predicted"))
    payload = {"project": "10-density-inversion", "sizes": SIZES, "results": results,
               "medium_max_rel_diff": differences,
               "notes": "Synthetic point-cell gravity, fixed beta=10; timing includes harness transfers and Metal process startup. Relative difference is max absolute error divided by reference maximum, across model and predicted. macOS ru_maxrss is a cumulative child high-water mark, so deltas can be zero after warm-up and are not per-call absolute peaks. MLX RSS is a cumulative high-water mark; tracemalloc may omit native allocations."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
