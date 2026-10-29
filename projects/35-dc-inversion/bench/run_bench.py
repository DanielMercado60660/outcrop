"""Benchmark the 2.5-D DC forward solve against the SimPEG oracle."""
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

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"nx": 6, "nz": 5, "electrodes": 4, "quadrature": 5},
    "medium": {"nx": 12, "nz": 7, "electrodes": 8, "quadrature": 13},
    "large": {"nx": 22, "nz": 11, "electrodes": 16, "quadrature": 25},
}


def make_problem(nx, nz, electrodes, quadrature):
    """Create a symmetric mesh, a seeded model, and a surface dipole survey."""
    hx = np.full(nx, 10, dtype=np.float32)
    hz = np.full(nz, 10, dtype=np.float32)
    origin = np.array([-5 * nx, -10 * nz], dtype=np.float32)
    ex = np.linspace(-5 * (electrodes - 1), 5 * (electrodes - 1), electrodes)
    locations = np.column_stack((ex, np.zeros(electrodes))).astype(np.float32)
    rows = [(a, a + 1, m, m + 1) for a in range(electrodes - 1)
            for m in range(electrodes - 1) if abs(m - a) >= 2]
    k = np.logspace(-5, np.log10(2), quadrature)
    dk = np.diff(k) / 2
    weights = np.r_[dk, 0] + np.r_[0, dk]
    weights[0] += k[0]
    rng = np.random.default_rng(350101)
    model = (2 + 0.1 * rng.standard_normal((nz, nx))).astype(np.float32)
    return {"hx": hx, "hz": hz, "origin": origin,
            "quadrature": np.column_stack((k, weights / np.pi)).astype(np.float32),
            "electrodes": locations, "abmn": np.asarray(rows, dtype=np.int32),
            "current": 1.0, "log10_rho": model}


def load_oracle():
    spec = importlib.util.spec_from_file_location("dc_oracle", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(oracle, inputs):
    # Include survey and simulator setup, just as the MLX API does on every call.
    geometry = {k: v for k, v in inputs.items() if k != "log10_rho"}
    _, sim, order = oracle.simulator(geometry)
    # The oracle constructor defaults to 25 nodes; field storage follows nT.
    sim.nky = len(inputs["quadrature"])
    sim.nT = sim.nky - 1
    sim.Ainv = [None] * sim.nky
    predicted, _ = oracle.predict(sim, order, inputs["log10_rho"].astype(np.float64))
    return predicted


def rss_mb():
    # macOS ru_maxrss is bytes and is a process-lifetime high-water mark.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def measure(path, inputs, oracle):
    run = (lambda: reference(oracle, inputs)) if path == "judge" else (
        lambda: call("mlx", "dc_predict", **inputs)["predicted"])
    run()  # Warm up and force materialization through the harness.
    if path == "judge":
        tracemalloc.start()
        tracemalloc.reset_peak()
    else:
        mx.reset_peak_memory()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        output = run()
        times.append(time.perf_counter() - start)
    if path == "judge":
        peak = tracemalloc.get_traced_memory()[1] / 1024**2
        tracemalloc.stop()
        method = "tracemalloc peak of traced Python allocations over three repeats"
    else:
        peak = mx.get_peak_memory() / 1024**2
        method = "MLX peak allocated bytes after reset over three repeats"
    result = {"wall_s": statistics.median(times), "peak_mem_mb": peak,
              "mem_method": method, "process_peak_rss_mb": rss_mb()}
    return result, np.asarray(output)


def main():
    oracle = load_oracle()
    results = {"judge": {}, "mlx": {}}
    medium = {}
    for name, size in SIZES.items():
        inputs = make_problem(**size)
        for path, path_results in results.items():
            print(f"{path}: {name}", flush=True)
            path_results[name], output = measure(path, inputs, oracle)
            if name == "medium":
                medium[path] = output.astype(np.float64)
    denominator = np.maximum(np.abs(medium["judge"]),
                             np.max(np.abs(medium["judge"])) * 1e-12)
    difference = float(np.max(np.abs(medium["mlx"] - medium["judge"]) / denominator))
    report = {"project": "35-dc-inversion", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": {"mlx": difference},
              "notes": "Forward voltage prediction; median of three timed calls after one "
                       "warmup, including per-call problem setup. Seed 350101. The judge "
                       "uses float64 SimPEG, MLX uses float32. RSS is the process lifetime "
                       "high-water mark on macOS; tracemalloc excludes native allocations. "
                       "Relative error is elementwise with a 1e-12 peak-voltage floor."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
