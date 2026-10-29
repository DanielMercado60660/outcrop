"""Benchmark the discrete eikonal objective and slowness gradient."""
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
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"nz": 10, "nx": 10, "ns": 1, "nr": 10},
    "medium": {"nz": 32, "nx": 32, "ns": 2, "nr": 32},
    "large": {"nz": 100, "nx": 100, "ns": 2, "nr": 100},
}


def inputs(size):
    rng = np.random.default_rng(2801)
    nz, nx, ns, nr = (size[k] for k in ("nz", "nx", "ns", "nr"))
    z, x = np.indices((nz, nx))
    slowness = (0.0005 * (1 + 0.12 * np.sin(x * 0.13 + z * 0.09)
                          + 0.015 * rng.standard_normal((nz, nx)))).astype(np.float32)
    sources = np.column_stack((np.zeros(ns), np.linspace(0, nx - 1, ns))).astype(np.int32)
    receivers = np.column_stack((np.full(nr, nz - 1),
                                 np.linspace(0, nx - 1, nr))).astype(np.int32)
    distance = np.sqrt(((receivers[None, :, 0] - sources[:, None, 0]) * 40.)**2
                       + ((receivers[None, :, 1] - sources[:, None, 1]) * 40.)**2)
    observed = (distance * 0.00048 + 0.01).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, (ns, nr)).astype(np.float32)
    return {"slowness": slowness, "spacing": [40., 40.], "sources": sources,
            "receivers": receivers, "observed": observed, "weights": weights}


def load_reference():
    spec = importlib.util.spec_from_file_location("eikonal_reference", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve


def main():
    import mlx.core as mx

    reference = load_reference()
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium = {}
    for name, size in SIZES.items():
        args = inputs(size)
        for path in ("judge", "mlx", "metal"):
            if path == "judge" and name == "large":
                results[path][name] = {"skipped": "Reference marching constructs a full-grid mask per update; large case may exceed 120 s."}
                continue

            def run(path=path, args=args):
                if path == "judge":
                    return reference(args["slowness"].astype(np.float64), args["spacing"],
                                     args["sources"], args["receivers"],
                                     args["observed"], args["weights"])
                return call(path, "misfit_and_grad", **args)

            run()
            times, peaks, rss = [], [], []
            for _ in range(3):
                if path == "judge":
                    tracemalloc.start()
                    tracemalloc.reset_peak()
                elif path == "mlx":
                    mx.reset_peak_memory()
                else:
                    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                start = time.perf_counter()
                output = run()
                times.append(time.perf_counter() - start)
                if path == "judge":
                    peaks.append(tracemalloc.get_traced_memory()[1] / 2**20)
                    tracemalloc.stop()
                elif path == "mlx":
                    peaks.append(mx.get_peak_memory() / 2**20)
                    rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
                else:
                    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                    peaks.append(max(0, after - before) / 2**20)
            method = {"judge": "tracemalloc peak traced allocations (MiB)",
                      "mlx": "MLX peak allocated memory (MiB); process RSS is lifetime high-water mark",
                      "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta (MiB, macOS); cumulative high-water, may be zero"}[path]
            entry = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
                     "mem_method": method}
            if rss:
                entry["process_peak_rss_mb"] = max(rss)
            results[path][name] = entry
            if name == "medium":
                medium[path] = output
            print(path, name, entry, flush=True)

    differences = {}
    for path in ("mlx", "metal"):
        errors = []
        for key, expected in medium["judge"].items():
            expected = np.asarray(expected, dtype=np.float64)
            actual = np.asarray(medium[path][key], dtype=np.float64)
            scale = float(np.max(np.abs(expected)))
            errors.append(float(np.max(np.abs(actual - expected))) / (scale or 1.0))
        differences[path] = max(errors)
    report = {"project": "28-diff-eikonal", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": differences,
              "notes": "Median of three end-to-end calls after one warmup; memory is the maximum of three measurements in MiB. Metal includes subprocess and file I/O. Child ru_maxrss is a cumulative high-water mark, so its delta is not an individual child peak and can be zero. MLX RSS is the Python process lifetime high-water mark. Medium relative difference is the maximum across output fields of infinity-norm error divided by reference infinity norm."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
