"""Benchmark the reference, MLX, and Metal acoustic propagators."""

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
from scipy.ndimage import correlate1d

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"n": 4096, "steps": 400},
    "medium": {"n": 8192, "steps": 1200},
    "large": {"n": 32768, "steps": 4096},
}
PATHS = ("judge", "mlx", "metal")


def inputs(size):
    n = size["n"]
    dx, c = 5.0, 1500.0
    dt = 0.5 * dx / c
    x = np.arange(n, dtype=np.float64) * dx
    rng = np.random.default_rng(1201)
    center = 0.5 * x[-1]
    width = 100.0

    def pulse(z):
        return np.exp(-((z - center) / width) ** 2)

    u0 = pulse(x) + rng.normal(0.0, 1e-5, n)
    previous = 0.5 * (pulse(x + c * dt) + pulse(x - c * dt))
    return {"u0": u0.astype(np.float32), "u_prev": previous.astype(np.float32),
            "c": c, "dx": dx, "dt": dt, "steps": size["steps"]}


def reference(u0, u_prev, c, dx, dt, steps):
    """The matched-stencil float64 reference in judge/make_goldens.py."""
    old = u0.astype(np.float64)
    prev = u_prev.astype(np.float64)
    rows = [old.copy()]
    r = c * dt / dx
    q = (r - 1) / (r + 1)
    stencil = np.array([1.0, -2.0, 1.0])
    for _ in range(steps):
        lap = correlate1d(old, stencil, mode="constant")
        new = 2 * old - prev + r * r * lap
        new[0] = old[1] + q * (new[1] - old[0])
        new[-1] = old[-2] + q * (new[-2] - old[-1])
        prev, old = old, new
        rows.append(old.copy())
    return {"wavefield": np.stack(rows)}


def run(path, args):
    return reference(**args) if path == "judge" else call(path, "propagate", **args)


def rss_mb(who):
    # macOS reports bytes; Linux reports KiB.
    factor = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(who).ru_maxrss * factor / (1024 ** 2)


def measure(path, size):
    args = inputs(SIZES[size])
    if path == "mlx":
        import mlx.core as mx
        mx.reset_peak_memory()
    elif path == "judge":
        tracemalloc.start()
    else:
        before = rss_mb(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    output = run(path, args)
    elapsed = time.perf_counter() - start
    if path == "mlx":
        mem = mx.get_peak_memory() / (1024 ** 2)
        method = "MLX reset_peak_memory/get_peak_memory; process RSS high-water also reported"
        extra = {"process_peak_rss_mb": rss_mb(resource.RUSAGE_SELF)}
    elif path == "metal":
        mem = max(0.0, rss_mb(resource.RUSAGE_CHILDREN) - before)
        method = "fresh worker RUSAGE_CHILDREN ru_maxrss delta (Metal child high-water RSS)"
        extra = {}
    else:
        mem = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
        tracemalloc.stop()
        method = "tracemalloc peak in fresh worker"
        extra = {}
    assert output["wavefield"].shape == (SIZES[size]["steps"] + 1, SIZES[size]["n"])
    return {"wall_s": elapsed, "peak_mem_mb": mem, "mem_method": method, **extra}


def worker(path, size):
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", path, size],
        check=True, capture_output=True, text=True,
    )
    return json.loads(proc.stdout)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        print(json.dumps(measure(sys.argv[2], sys.argv[3])))
        return
    results = {path: {} for path in PATHS}
    for path in PATHS:
        for size in SIZES:
            print(f"{path} {size}: warmup and three repeats", flush=True)
            worker(path, size)
            measurements = [worker(path, size) for _ in range(3)]
            result = dict(measurements[0])
            for key in ("wall_s", "peak_mem_mb", "process_peak_rss_mb"):
                if key in result:
                    result[key] = statistics.median(m[key] for m in measurements)
            results[path][size] = result
    args = inputs(SIZES["medium"])
    expected = run("judge", args)["wavefield"]
    scale = float(np.max(np.abs(expected)))
    differences = {}
    for path in ("mlx", "metal"):
        actual = run(path, args)["wavefield"]
        differences[path] = float(np.max(np.abs(actual - expected)) / scale)
        del actual
    report = {
        "project": "12-acoustic-1d", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": "Wall times include output materialization and, for Metal, binary launch and file I/O. "
                 "Each warmup and repeat uses a fresh Python process; startup is outside the timer. "
                 "Memory values are medians of three peaks, in MiB. Metal child ru_maxrss is a "
                 "high-water mark, not GPU allocation alone. MLX RSS is a process high-water mark. "
                 "Relative difference is max absolute error divided by reference peak amplitude. "
                 "Inputs use seed 1201, dx=5 m, c=1500 m/s and Courant number 0.5.",
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
