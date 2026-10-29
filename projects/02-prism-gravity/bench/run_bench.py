"""Benchmark the downward gravity field through the project API."""
from __future__ import annotations

import json
import multiprocessing as mp
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"observations": 256, "prisms": 256},
    "medium": {"observations": 2048, "prisms": 512},
    "large": {"observations": 8192, "prisms": 800},
}


def inputs(size):
    rng = np.random.default_rng(20261028)
    n, p = size["observations"], size["prisms"]
    coordinates = np.column_stack((rng.uniform(-1000, 1000, (n, 2)),
                                   rng.uniform(80, 180, n))).astype(np.float32)
    centers = rng.uniform(-800, 800, (p, 2))
    widths = rng.uniform(30, 100, (p, 2))
    top = rng.uniform(-150, -20, p)
    bottom = top - rng.uniform(40, 180, p)
    prisms = np.column_stack((centers[:, 0] - widths[:, 0],
                              centers[:, 0] + widths[:, 0],
                              centers[:, 1] - widths[:, 1],
                              centers[:, 1] + widths[:, 1], bottom, top)).astype(np.float32)
    density = rng.uniform(200, 800, p).astype(np.float32)
    return {"coordinates": coordinates, "prisms": prisms, "density": density, "field": "g_z"}


def reference(args):
    return hm.prism_gravity(tuple(args["coordinates"].astype(np.float64).T),
                            args["prisms"].astype(np.float64),
                            args["density"].astype(np.float64), field="g_z",
                            dtype=np.float64, parallel=False)


def metal_worker(conn, args):
    """A fresh worker gives the Metal child a clean child-RSS high-water mark."""
    try:
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        output = call("metal", "prism_gravity", **args)["g_z"]
        elapsed = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        conn.send((elapsed, max(0, after - before) / 1024**2, output, None))
    except Exception as exc:  # noqa: BLE001 - propagate worker failures to the parent
        conn.send((None, None, None, repr(exc)))
    finally:
        conn.close()


def metal_once(args):
    ctx = mp.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    process = ctx.Process(target=metal_worker, args=(writer, args))
    process.start()
    writer.close()
    try:
        result = reader.recv()
    finally:
        reader.close()
        process.join()
    elapsed, memory, output, error = result
    if error or process.exitcode:
        raise RuntimeError(f"Metal benchmark worker failed: {error}, exit {process.exitcode}")
    return elapsed, memory, output


def run():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    diffs = {}
    for size_name, size in SIZES.items():
        args = inputs(size)
        outputs = {}
        for path, path_results in results.items():
            if path == "judge" and size_name == "large":
                path_results[size_name] = {"skipped": "Large serial float64 reference may exceed 120 s."}
                continue
            print(f"{path} {size_name}", flush=True)
            if path == "judge":
                reference(args)
            elif path == "metal":
                metal_once(args)
            else:
                call(path, "prism_gravity", **args)
            times, memories, rss = [], [], []
            for _ in range(3):
                if path == "metal":
                    elapsed, memory, out = metal_once(args)
                elif path == "mlx":
                    mx.reset_peak_memory()
                    start = time.perf_counter()
                    out = call(path, "prism_gravity", **args)["g_z"]
                    elapsed = time.perf_counter() - start
                    memory = mx.get_peak_memory() / 1024**2
                    rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
                else:
                    tracemalloc.start()
                    start = time.perf_counter()
                    out = reference(args)
                    elapsed = time.perf_counter() - start
                    memory = tracemalloc.get_traced_memory()[1] / 1024**2
                    tracemalloc.stop()
                times.append(elapsed)
                memories.append(memory)
            methods = {
                "judge": "tracemalloc peak during call (Python-tracked allocations)",
                "mlx": "MLX reset_peak_memory/get_peak_memory, bytes to MiB",
                "metal": "fresh worker RUSAGE_CHILDREN ru_maxrss delta, macOS bytes to MiB",
            }
            entry = {"wall_s": statistics.median(times), "peak_mem_mb": max(memories),
                     "mem_method": methods[path]}
            if rss:
                entry["process_peak_rss_mb"] = max(rss)
            path_results[size_name] = entry
            if size_name == "medium":
                outputs[path] = np.asarray(out, dtype=np.float64)
        if size_name == "medium":
            ref = outputs["judge"]
            scale = float(np.max(np.abs(ref)))
            for path in ("mlx", "metal"):
                error = float(np.max(np.abs(outputs[path] - ref)))
                diffs[path] = error / scale if scale else error
    report = {
        "project": "02-prism-gravity", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": diffs,
        "notes": ("g_z in mGal; deterministic seed 20261028. Wall time is the median of three "
                  "calls after one warm-up, including harness serialization and Metal process startup, "
                  "but excluding fresh measurement-worker startup. Memory is the largest of three "
                  "peaks in MiB; process RSS is a lifetime high-water mark. Metal child RSS excludes "
                  "GPU allocations; MLX peak measures MLX allocations, not total RSS. Reference "
                  "tracemalloc excludes untracked native allocations. Relative difference is maximum "
                  "absolute error divided by maximum absolute reference value. Timings and size "
                  "estimates are unvalidated on the target GPU."),
    }
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
