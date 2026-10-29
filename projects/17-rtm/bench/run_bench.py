"""Benchmark the RTM adjoint on deterministic synthetic surveys."""
from __future__ import annotations

import importlib.util
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

# Work scales roughly with shots * padded grid area * time steps.
SIZES = {
    "small": {"nz": 24, "nx": 24, "nt": 100, "nshot": 1, "nrec": 8},
    "medium": {"nz": 48, "nx": 48, "nt": 300, "nshot": 2, "nrec": 20},
    "large": {"nz": 96, "nx": 96, "nt": 700, "nshot": 4, "nrec": 44},
}


def inputs(size):
    p = SIZES[size]
    nz, nx, nt, shots, nrec = (p[k] for k in ("nz", "nx", "nt", "nshot", "nrec"))
    rng = np.random.default_rng(170101)
    z, x = np.indices((nz, nx))
    velocity = (1850 + 3 * z + 25 * np.cos(x / 12)).astype(np.float32)
    dt, freq = 0.001, 25.0
    t = np.arange(nt) * dt
    a = (np.pi * freq * (t - 1.5 / freq)) ** 2
    wavelet = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)
    source = np.tile(wavelet, (shots, 1, 1))
    sx = np.linspace(4, nx - 5, shots).round().astype(np.int64)
    rx = np.linspace(3, nx - 4, nrec).round().astype(np.int64)
    sl = np.zeros((shots, 1, 2), dtype=np.int64)
    sl[:, 0, 0], sl[:, 0, 1] = 3, sx
    rl = np.zeros((shots, nrec, 2), dtype=np.int64)
    rl[:, :, 0], rl[:, :, 1] = 3, rx
    data = rng.standard_normal((shots, nrec, nt)).astype(np.float32)
    return {"velocity": velocity, "source_amplitudes": source, "source_locations": sl,
            "receiver_locations": rl, "data": data, "grid_spacing": [10.0, 10.0],
            "dt": dt, "pml_width": 12, "pml_freq": freq}


def metal_worker(size):
    args = inputs(size)
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    call("metal", "rtm", **args)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": max(0, after - before) / 2**20}))


def run_metal(size):
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--metal-worker", size],
                          check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def main():
    import mlx.core as mx
    import torch

    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("rtm_judge", PROJECT / "judge" / "make_goldens.py")
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium_outputs = {}
    for size in SIZES:
        args = inputs(size)
        # The oracle expects spacing as an ndarray, while the API expects a list.
        g = dict(args, grid_spacing=np.asarray(args["grid_spacing"]))
        for path in ("judge", "mlx", "metal"):
            print(f"{path} {size}", flush=True)
            if path == "judge":
                invoke = lambda g=g, args=args: judge.migrate(g, args["data"])
            elif path == "mlx":
                invoke = lambda args=args: call("mlx", "rtm", **args)["image"]
            else:
                invoke = lambda size=size: run_metal(size)
            invoke()  # warm-up is outside the timing
            times, peaks, rss = [], [], []
            for _ in range(3):
                if path == "metal":
                    measurement = invoke()
                    times.append(measurement["wall_s"])
                    peaks.append(measurement["peak_mem_mb"])
                    continue
                if path == "judge":
                    tracemalloc.start()
                    tracemalloc.reset_peak()
                else:
                    mx.reset_peak_memory()
                start = time.perf_counter()
                out = invoke()
                times.append(time.perf_counter() - start)
                if path == "judge":
                    peaks.append(tracemalloc.get_traced_memory()[1] / 2**20)
                    tracemalloc.stop()
                else:
                    peaks.append(mx.get_peak_memory() / 2**20)
                    rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
                if size == "medium":
                    medium_outputs[path] = np.asarray(out, dtype=np.float64)
            method = {"judge": "tracemalloc peak (Python allocations; excludes native Torch allocations)",
                      "mlx": "MLX peak allocated bytes after reset; process RSS high-water also reported",
                      "metal": "fresh worker RUSAGE_CHILDREN ru_maxrss delta (macOS bytes), Metal child process RSS"}[path]
            row = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
                   "mem_method": method}
            if rss:
                row["process_peak_rss_mb"] = max(rss)
            results[path][size] = row
    medium_outputs["metal"] = np.asarray(call("metal", "rtm", **inputs("medium"))["image"], dtype=np.float64)
    ref = medium_outputs["judge"]
    scale = np.max(np.abs(ref))
    diffs = {path: float(np.max(np.abs(medium_outputs[path] - ref)) / scale)
             if scale else float(np.max(np.abs(medium_outputs[path])))
             for path in ("mlx", "metal")}
    payload = {"project": "17-rtm", "sizes": SIZES, "results": results,
               "medium_max_rel_diff": diffs,
               "notes": "RTM image; median of three calls after one warm-up; fixed seed 170101. "
                        "Metal timing includes harness file I/O and process startup. Memory is the maximum "
                        "of three peaks; memory methods differ and do not measure the same allocations. "
                        "MLX process RSS is a lifetime high-water mark; Metal RSS includes shared buffers "
                        "only as reflected in OS process accounting. Relative difference is max absolute "
                        "error divided by max absolute reference image."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-worker":
        metal_worker(sys.argv[2])
    else:
        main()
