"""Benchmark the acoustic objective and gradient on deterministic surveys."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"nz": 12, "nx": 24, "nt": 64, "shots": 1, "receivers": 4},
    "medium": {"nz": 25, "nx": 61, "nt": 256, "shots": 2, "receivers": 8},
    "large": {"nz": 40, "nx": 80, "nt": 1024, "shots": 2, "receivers": 16},
}


def make_inputs(size):
    nz, nx, nt = size["nz"], size["nx"], size["nt"]
    shots, nr = size["shots"], size["receivers"]
    dx, dt = 20.0, 0.002
    basis = np.zeros((2, nz, nx), dtype=np.float32)
    basis[0, :, :nx // 2] = 1
    basis[1, :, nx // 2:] = 1
    t = np.arange(nt, dtype=np.float64) * dt
    a = (np.pi * 12.0 * (t - 0.06)) ** 2
    pulse = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)
    source = np.tile(pulse, (shots, 1, 1))
    source_x = np.linspace(2, nx - 3, shots, dtype=np.int64)
    sources = np.array([[[nz // 2, x]] for x in source_x], dtype=np.int64)
    receiver_x = np.linspace(3, nx - 4, nr, dtype=np.int64)
    receiver_line = np.array([[nz // 2, x] for x in receiver_x], dtype=np.int64)
    receivers = np.tile(receiver_line[None, :, :], (shots, 1, 1))
    # A deterministic nonzero synthetic target avoids timing data generation.
    rng = np.random.default_rng(190028)
    delays = rng.uniform(0.02, 0.04, (shots, nr, 1))
    b = (np.pi * 12.0 * (t[None, None, :] - 0.06 - delays)) ** 2
    observed = ((1 - 2 * b) * np.exp(-b)).astype(np.float32)
    return {"parameters": np.array([1900, 2300], dtype=np.float32), "basis": basis,
            "source": source, "sources": sources, "receivers": receivers,
            "observed": observed, "dx": dx, "dt": dt, "cutoff": 9.0}


def reference(args, oracle):
    value, gradient, predicted = oracle["evaluate"](
        args["parameters"], args, args["cutoff"])
    return {"predicted": predicted, "objective": np.asarray(value), "gradient": gradient}


def rss_mb():
    # ru_maxrss is bytes on macOS and KiB on Linux; this is a process high-water mark.
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def main():
    import mlx.core as mx

    oracle = runpy.run_path(str(ROOT / "judge" / "make_goldens.py"))
    results = {"judge": {}, "mlx": {}}
    outputs = {}
    for size_name, size in SIZES.items():
        args = make_inputs(size)
        for path in ("judge", "mlx"):
            fn = (lambda args=args: reference(args, oracle)) if path == "judge" else (
                lambda args=args: call("mlx", "misfit_and_grad", **args))
            fn()  # Warm up compilation, allocator and caches.
            if path == "mlx":
                mx.reset_peak_memory()
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                out = fn()
                samples.append(time.perf_counter() - start)
            if size_name == "medium":
                outputs[path] = out
            if path == "judge":
                tracemalloc.start()
                fn()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                method = "tracemalloc peak during separate call (Python allocations only)"
                memory = peak / 1024**2
            else:
                memory = mx.get_peak_memory() / 1024**2
                method = "MLX get_peak_memory after reset, across three timed calls"
            results[path][size_name] = {
                "wall_s": statistics.median(samples), "peak_mem_mb": memory,
                "mem_method": method, "process_peak_rss_mb": rss_mb(),
            }
            print(f"{path:5s} {size_name:6s}: {statistics.median(samples):.3f} s", flush=True)
    relative = {}
    for key in ("predicted", "objective", "gradient"):
        actual = np.asarray(outputs["mlx"][key], dtype=np.float64)
        expected = np.asarray(outputs["judge"][key], dtype=np.float64)
        denom = float(np.max(np.abs(expected)))
        error = float(np.max(np.abs(actual - expected)))
        relative[key] = error / denom if denom else error
    document = {
        "project": "19-multiscale-fwi", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": {"mlx": max(relative.values())},
        "notes": "misfit_and_grad; seed 190028; dx=20 m, dt=0.002 s, cutoff=9 Hz. "
        "Observed traces are synthetic Ricker-shaped targets, not modeled wavefields. "
        "Relative difference is the maximum of per-output max absolute differences divided "
        "by each reference output's maximum absolute value. "
        f"Per-output differences: {relative}. MLX uses CPU float64. Process RSS is a "
        "cumulative high-water mark, and tracemalloc excludes native PyTorch allocations.",
    }
    (ROOT / "bench" / "results.json").write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
