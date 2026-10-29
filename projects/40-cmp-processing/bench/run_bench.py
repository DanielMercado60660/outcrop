"""Benchmark CMP velocity semblance through the locked interfaces."""
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
    "small": {"n_offsets": 32, "n_time": 384, "n_velocities": 16, "dt_s": 0.002, "half_window": 5},
    "medium": {"n_offsets": 64, "n_time": 1024, "n_velocities": 64, "dt_s": 0.002, "half_window": 5},
    "large": {"n_offsets": 192, "n_time": 3072, "n_velocities": 256, "dt_s": 0.002, "half_window": 5},
}


def inputs(size):
    rng = np.random.default_rng(40)
    dt = size["dt_s"]
    times = np.arange(size["n_time"], dtype=np.float64) * dt
    offsets = np.linspace(0, 1500, size["n_offsets"], dtype=np.float32)
    gather = rng.normal(0, 0.01, (offsets.size, times.size))
    for t0, velocity, amplitude in [(0.35, 1800, 1.0), (0.72, 2400, -0.8), (1.08, 3000, 0.7)]:
        arrival = np.sqrt(t0**2 + (offsets.astype(np.float64) / velocity) ** 2)
        u = np.pi * 25 * (times[None, :] - arrival[:, None])
        gather += amplitude * (1 - 2 * u * u) * np.exp(-u * u)
    return {"gather": gather.astype(np.float32), "offsets": offsets, "dt": dt,
            "velocities": np.linspace(1500, 3300, size["n_velocities"], dtype=np.float32),
            "half_window": size["half_window"]}


def reference(gather, offsets, dt, velocities, half_window):
    """Float64 interpolation and clipped window sums, as in make_goldens.py."""
    traces = gather.astype(np.float64)
    x = offsets.astype(np.float64)
    t = np.arange(traces.shape[1], dtype=np.float64) * dt
    rows = []
    for velocity in velocities:
        query = np.sqrt(t[None, :] ** 2 + (x[:, None] / float(velocity)) ** 2)
        corrected = np.stack([np.interp(q, t, trace, left=0., right=0.)
                              for q, trace in zip(query, traces)])
        numerator = corrected.sum(axis=0) ** 2
        denominator = x.size * (corrected * corrected).sum(axis=0)
        num = np.array([numerator[max(0, j-half_window):min(t.size, j+half_window+1)].sum()
                        for j in range(t.size)])
        den = np.array([denominator[max(0, j-half_window):min(t.size, j+half_window+1)].sum()
                        for j in range(t.size)])
        rows.append(np.divide(num, den, out=np.zeros_like(num), where=den != 0))
    return {"semblance": np.asarray(rows)}


def run(path, args):
    return reference(**args) if path == "judge" else call(path, "semblance", **args)


def benchmark(path, args):
    run(path, args)
    elapsed, memory, rss = [], [], []
    if path == "mlx":
        import mlx.core as mx
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
            tracemalloc.reset_peak()
        elif path == "mlx":
            mx.reset_peak_memory()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        run(path, args)
        elapsed.append(time.perf_counter() - start)
        if path == "judge":
            memory.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        elif path == "mlx":
            memory.append(mx.get_peak_memory() / 1e6)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            memory.append(max(0, after - before) / 1e6)
    methods = {
        "judge": "tracemalloc peak traced allocations during call (MB)",
        "mlx": "MLX reset_peak_memory/get_peak_memory (MB); process RSS is lifetime high-water mark",
        "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta in macOS bytes (MB); may be zero after warmup",
    }
    result = {"wall_s": statistics.median(elapsed), "peak_mem_mb": max(memory), "mem_method": methods[path]}
    if rss:
        result["process_peak_rss_mb"] = max(rss)
    return result


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {name}", flush=True)
            path_results[name] = benchmark(path, args)
    args = inputs(SIZES["medium"])
    truth = reference(**args)["semblance"]
    scale = float(np.max(np.abs(truth)))
    diffs = {path: float(np.max(np.abs(run(path, args)["semblance"] - truth)) / scale)
             for path in ("mlx", "metal")}
    data = {"project": "40-cmp-processing", "sizes": SIZES, "results": results,
            "medium_max_rel_diff": diffs,
            "notes": "Semblance, including harness I/O and synchronization; median of three calls after one warmup. Relative difference is max absolute difference divided by max absolute reference value. RSS and MLX memory are high-water measurements, not additive; child RSS deltas can be zero after warmup. MB = 10^6 bytes. Fixed RNG seed 40."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
