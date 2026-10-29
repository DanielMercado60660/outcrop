"""Benchmark the acoustic propagators from the project directory."""
import importlib.util
import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"nz": 64, "nx": 64, "nt": 24, "shots": 1, "receivers": 32},
    "medium": {"nz": 128, "nx": 128, "nt": 240, "shots": 1, "receivers": 64},
    "large": {"nz": 256, "nx": 256, "nt": 1000, "shots": 1, "receivers": 128},
}


def inputs(size):
    rng = np.random.default_rng(1301)
    nz, nx, nt = size["nz"], size["nx"], size["nt"]
    z, x = np.indices((nz, nx))
    velocity = (1600 + 160 * (z > nz * 0.55) + 55 * np.sin(x / 17)
                + rng.normal(0, 2, (nz, nx))).astype(np.float32)
    dt = 0.001
    t = np.arange(nt) * dt
    q = np.pi * 18 * (t - 0.055)
    wavelet = ((1 - 2 * q**2) * np.exp(-q**2) / 100).astype(np.float32)
    source = np.broadcast_to(wavelet, (size["shots"], 1, nt)).copy()
    sl = np.tile(np.array([[[nz // 4, nx // 2]]], dtype=np.int64),
                 (size["shots"], 1, 1))
    rx = np.linspace(2, nx - 3, size["receivers"], dtype=np.int64)
    rec = np.column_stack((np.full(rx.size, nz // 4), rx))
    rl = np.tile(rec[None, :, :], (size["shots"], 1, 1))
    return {"velocity": velocity, "source": source, "source_locations": sl,
            "receiver_locations": rl, "spacing": 10.0, "dt": dt, "order": 4,
            "pml_width": [12, 12, 12, 12], "pml_freq": 18.0, "max_vel": 1900.0}


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "acoustic_reference", PROJECT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def invoke(path, args, reference):
    if path == "judge":
        return reference(args["velocity"], args["source"], args["source_locations"],
                         args["receiver_locations"], args["spacing"], args["dt"],
                         args["order"], args["pml_width"], args["pml_freq"],
                         args["max_vel"])
    return call(path, "scalar", **args)["receiver_amplitudes"]


def measure(path, args, reference):
    invoke(path, args, reference)
    times, peaks = [], []
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
            tracemalloc.reset_peak()
        elif path == "mlx":
            mx.reset_peak_memory()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        result = invoke(path, args, reference)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory() / 1e6)
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, after - before) / 1e6)
        del result
    methods = {
        "judge": "tracemalloc peak (Python allocations; excludes native Torch allocations)",
        "mlx": "MLX reset_peak_memory/get_peak_memory (bytes; includes live MLX allocations)",
        "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta, macOS bytes; may be zero after warm-up",
    }
    entry = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
             "mem_method": methods[path]}
    if path == "mlx":
        entry["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return entry


def main():
    reference = load_reference()
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, measurements in results.items():
            print(f"{path} {name}", flush=True)
            measurements[name] = measure(path, args, reference)
    args = inputs(SIZES["medium"])
    truth = np.asarray(invoke("judge", args, reference), dtype=np.float64)
    scale = float(np.max(np.abs(truth)))
    diffs = {}
    for path in ("mlx", "metal"):
        actual = np.asarray(invoke(path, args, reference), dtype=np.float64)
        error = float(np.max(np.abs(actual - truth)))
        diffs[path] = error / scale if scale else error
    report = {"project": "13-acoustic-2d", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": diffs,
              "notes": "Median of three wall-clock calls after one warm-up; inputs fixed by seed 1301. "
                       "Spacing 10 m, dt 0.001 s, order 4, 12-cell PML. "
                       "Metal timing includes process startup, shader compilation and file I/O. "
                       "Memory MB is decimal; MLX RSS is the parent lifetime high-water mark. "
                       "Child RSS is a cumulative high-water statistic, so its delta is not an individual child peak. "
                       "Relative difference is maximum absolute error divided by reference peak amplitude."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
