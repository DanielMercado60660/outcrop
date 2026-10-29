"""Benchmark the forward P-SV solver from the project directory."""
from __future__ import annotations

import json
import os
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"ny": 48, "nx": 64, "nt": 60, "nreceiver": 24},
    "medium": {"ny": 96, "nx": 144, "nt": 300, "nreceiver": 72},
    "large": {"ny": 160, "nx": 240, "nt": 600, "nreceiver": 120},
}


def make_inputs(ny, nx, nt, nreceiver):
    rng = np.random.default_rng(150127)
    y, x = np.mgrid[:ny, :nx]
    lens = np.exp(-(((y - 0.6 * ny) / (0.19 * ny)) ** 2
                    + ((x - 0.58 * nx) / (0.22 * nx)) ** 2))
    jitter = rng.normal(0, 0.002, (ny, nx))
    vp = np.asarray(3000 + 350 * lens + 300 * (y > 0.55 * ny) + 3000 * jitter,
                    dtype=np.float32)
    vs = np.asarray(vp * 0.52, dtype=np.float32)
    rho = np.asarray(2000 + 160 * lens + 100 * (y > 0.55 * ny), dtype=np.float32)
    dt = 0.0004
    t = np.arange(nt) * dt
    q = (np.pi * 25.0 * (t - 0.04)) ** 2
    source_y = np.asarray(((1 - 2 * q) * np.exp(-q) * 1e8)[None, None, :],
                          dtype=np.float32)
    locations = np.zeros((1, nreceiver, 2), dtype=np.int64)
    locations[0, :, 0] = ny // 4
    locations[0, :, 1] = np.linspace(2, nx - 2, nreceiver, dtype=np.int64)
    return {"vp": vp, "vs": vs, "rho": rho, "source_y": source_y,
            "source_x": np.zeros_like(source_y),
            "source_locations": np.array([[[ny // 3, nx // 2]]], dtype=np.int64),
            "receiver_locations": locations, "spacing": [10.0, 10.0], "dt": dt,
            "pml_width": [12, 12, 12, 12], "pml_freq": 25.0, "max_vel": 4000.0}


def judge_call(args):
    from judge.make_goldens import simulate

    case = dict(args, spacing=np.asarray(args["spacing"]),
                pml_width=np.asarray(args["pml_width"]))
    return simulate(case)


def metal_worker(size):
    args = make_inputs(**SIZES[size])
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    call("metal", "elastic", **args)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss - before
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": peak / 1e6}))


def benchmark(path, size, args):
    invoke = (lambda: judge_call(args)) if path == "judge" else (
        lambda: call(path, "elastic", **args))
    invoke()
    times, peaks = [], []
    if path == "metal":
        # Fresh workers keep the child high-water mark specific to each call.
        for _ in range(3):
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--metal-worker", size], check=True,
                                  capture_output=True, text=True)
            measurement = json.loads(proc.stdout)
            times.append(measurement["wall_s"])
            peaks.append(measurement["peak_mem_mb"])
        method = "fresh worker RUSAGE_CHILDREN ru_maxrss delta (macOS bytes); Metal child RSS"
        return {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
                "mem_method": method}
    if path == "mlx":
        import mlx.core as mx
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
            tracemalloc.reset_peak()
        else:
            mx.reset_peak_memory()
        start = time.perf_counter()
        invoke()
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        else:
            peaks.append(mx.get_peak_memory() / 1e6)
    if path == "judge":
        method = "tracemalloc peak Python allocations (excludes native Torch allocations)"
    else:
        method = "MLX reset_peak_memory/get_peak_memory; process RSS is lifetime high-water mark"
    result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
              "mem_method": method}
    if path == "mlx":
        result["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-worker":
        metal_worker(sys.argv[2])
        return
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for name, params in SIZES.items():
        args = make_inputs(**params)
        for path, path_results in results.items():
            if path == "judge" and name == "large":
                path_results[name] = {"skipped": "Conservative CPU runtime limit for the large case"}
                continue
            print(f"Benchmarking {path} / {name}", flush=True)
            path_results[name] = benchmark(path, name, args)
    args = make_inputs(**SIZES["medium"])
    reference = judge_call(args)
    differences = {}
    for path in ("mlx", "metal"):
        actual = call(path, "elastic", **args)
        errors = []
        for key in ("vy", "vx", "p"):
            ref = np.asarray(reference[key], dtype=np.float64)
            delta = np.max(np.abs(np.asarray(actual[key], dtype=np.float64) - ref))
            scale = np.max(np.abs(ref))
            errors.append(float(delta / scale if scale else delta))
        differences[path] = max(errors)
    report = {"project": "15-elastic-psv", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": differences,
              "notes": "Forward elastic only; three timed repeats after one warmup. "
              "Errors are the maximum of component-wise max absolute error divided "
              "by that reference component's peak. Memory is the maximum across repeats; "
              "tracemalloc excludes native allocations, and RSS includes runtime overhead. "
              "Metal timing includes harness file I/O and process startup. "
              "Sizes and runtime targets are estimates, not measured on the reserved GPU."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
