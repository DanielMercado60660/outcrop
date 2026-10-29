"""Benchmark order-1 fast marching on synthetic 2D grids."""
import json
import multiprocessing as mp
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
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"shape": [48, 48], "spacing_km": [0.025, 0.025], "order": 1},
    "medium": {"shape": [160, 160], "spacing_km": [0.025, 0.025], "order": 1},
    "large": {"shape": [480, 480], "spacing_km": [0.025, 0.025], "order": 1},
}


def inputs(params):
    n, m = params["shape"]
    y, x = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m), indexing="ij")
    phi = np.hypot(x - 0.38, y - 0.45) - 0.035
    rng = np.random.default_rng(24001)
    speed = 2.5 + 0.35 * np.sin(4 * x) * np.cos(3 * y)
    speed += rng.uniform(-0.025, 0.025, size=(n, m))
    return {"phi": phi.astype(np.float32), "speed": speed.astype(np.float32),
            "spacing": params["spacing_km"], "order": params["order"]}


def invoke(path, args):
    if path == "judge":
        import skfmm
        return np.asarray(skfmm.travel_time(args["phi"].astype(np.float64),
                                           args["speed"].astype(np.float64),
                                           dx=args["spacing"], order=args["order"]))
    return call(path, "travel_time", **args)["time"]


def metal_worker(conn, args):
    # A fresh parent makes RUSAGE_CHILDREN independent for each repeat.
    try:
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        result = invoke("metal", args)
        elapsed = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        conn.send((elapsed, max(0, after - before) / 1024**2, result, None))
    except Exception as exc:  # noqa: BLE001 -- relay worker errors to the parent.
        conn.send((None, None, None, repr(exc)))
    finally:
        conn.close()


def measure(path, args):
    if path == "metal":
        ctx = mp.get_context("spawn")
        recv, send = ctx.Pipe(duplex=False)
        process = ctx.Process(target=metal_worker, args=(send, args))
        process.start()
        send.close()
        try:
            elapsed, memory, output, error = recv.recv()
        finally:
            recv.close()
            process.join()
        if process.exitcode != 0 or error:
            raise RuntimeError(f"Metal worker failed: {error}, exit={process.exitcode}")
        return elapsed, memory, output, None
    if path == "judge":
        tracemalloc.start()
        tracemalloc.reset_peak()
    else:
        import mlx.core as mx
        mx.reset_peak_memory()
    start = time.perf_counter()
    output = invoke(path, args)
    elapsed = time.perf_counter() - start
    rss = None
    if path == "judge":
        memory = tracemalloc.get_traced_memory()[1] / 1024**2
        tracemalloc.stop()
    else:
        memory = mx.get_peak_memory() / 1024**2
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    return elapsed, memory, output, rss


def main():
    results = {p: {} for p in ("judge", "mlx", "metal")}
    medium_outputs = {}
    methods = {
        "judge": "tracemalloc peak traced Python/NumPy allocations during call",
        "mlx": "MLX reset_peak_memory/get_peak_memory; process_rss_mb is process high-water RSS",
        "metal": "fresh worker RUSAGE_CHILDREN ru_maxrss delta (macOS bytes), child RSS high-water",
    }
    for size, params in SIZES.items():
        args = inputs(params)
        for path, path_results in results.items():
            print(f"{size}: {path}", flush=True)
            invoke(path, args)
            times, memories, rss_values = [], [], []
            for _ in range(3):
                elapsed, memory, output, rss = measure(path, args)
                times.append(elapsed)
                memories.append(memory)
                if rss is not None:
                    rss_values.append(rss)
            entry = {"wall_s": statistics.median(times), "peak_mem_mb": max(memories),
                     "mem_method": methods[path]}
            if rss_values:
                entry["process_rss_mb"] = max(rss_values)
            path_results[size] = entry
            if size == "medium":
                medium_outputs[path] = output
    reference = medium_outputs["judge"]
    scale = float(np.max(np.abs(reference)))
    diffs = {p: float(np.max(np.abs(medium_outputs[p].astype(np.float64) - reference)) / scale)
             for p in ("mlx", "metal")}
    report = {"project": "24-fast-marching", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": diffs,
              "notes": "Order 1; float32 synthetic inputs, scikit-fmm float64 reference. "
                       "Wall times include harness serialization and Metal process startup, "
                       "but exclude input construction and benchmark-worker startup. "
                       "Memory is the maximum of three repeats in MiB; methods are not "
                       "directly comparable. tracemalloc can miss native allocations; MLX "
                       "peak excludes CPU marching allocations and RSS is a lifetime high-water "
                       "mark. Metal RSS excludes GPU allocation accounting and is measured "
                       "in a fresh worker for each repeat. Relative error is max absolute "
                       "difference divided by maximum absolute reference time."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
