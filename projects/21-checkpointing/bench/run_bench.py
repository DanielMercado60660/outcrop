"""Benchmark the acoustic checkpointed gradient on deterministic synthetic data."""
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
    "small": {"nz": 16, "nx": 24, "nt": 16, "nshot": 1, "nreceiver": 8, "checkpoint_budget": 7},
    "medium": {"nz": 48, "nx": 64, "nt": 64, "nshot": 1, "nreceiver": 16, "checkpoint_budget": 7},
    "large": {"nz": 96, "nx": 128, "nt": 192, "nshot": 1, "nreceiver": 32, "checkpoint_budget": 7},
}


def make_inputs(nz, nx, nt, nshot, nreceiver, checkpoint_budget):
    rng = np.random.default_rng(210021)
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    velocity = (1800 + 0.8 * z + 55 * np.exp(-((z - .6 * nz) ** 2 / (.18 * nz) ** 2
                                                   + (x - .55 * nx) ** 2 / (.2 * nx) ** 2))
                + rng.normal(0, 2, (nz, nx))).astype(np.float32)
    dt = 0.001
    phase = np.pi * 25 * (np.arange(nt) * dt - 0.04)
    wavelet = (0.03 * (1 - 2 * phase**2) * np.exp(-phase**2)).astype(np.float32)
    source = np.broadcast_to(wavelet, (nshot, 1, nt)).copy()
    sources = np.array([[[nz // 3, nx // 2]] for _ in range(nshot)], dtype=np.int64)
    rx = np.linspace(2, nx - 3, nreceiver, dtype=np.int64)
    receivers = np.array([[[nz // 3, int(i)] for i in rx] for _ in range(nshot)], dtype=np.int64)
    return {"velocity": velocity, "source_amplitudes": source, "source_locations": sources,
            "receiver_locations": receivers, "observed": np.zeros((nshot, nreceiver, nt), np.float32),
            "grid_spacing": [8.0, 10.0], "dt": dt, "accuracy": 2, "pml_width": [8] * 4,
            "pml_freq": 25.0, "max_vel": 2600.0, "checkpoint_budget": checkpoint_budget}


def reference_module():
    spec = importlib.util.spec_from_file_location("checkpoint_reference", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(module, inputs):
    import torch

    case = dict(inputs)
    case["grid_spacing"] = np.asarray(case["grid_spacing"])
    case["pml_width"] = np.asarray(case["pml_width"])
    return module.propagate(case, inputs["velocity"], torch.float64)


def relative(out, ref):
    errors = []
    for key in ("objective", "predicted", "gradient"):
        a, b = np.asarray(out[key], np.float64), np.asarray(ref[key], np.float64)
        scale = float(np.max(np.abs(b)))
        error = float(np.max(np.abs(a - b)))
        errors.append(error / scale if scale else error)
    return max(errors)


def main():
    import mlx.core as mx

    module = reference_module()
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium_outputs = {}
    for size, params in SIZES.items():
        inputs = make_inputs(**params)
        for path, path_results in results.items():
            print(f"{path}: {size}", flush=True)
            invoke = (lambda inputs=inputs: reference(module, inputs)) if path == "judge" else (
                lambda path=path, inputs=inputs: call(path, "misfit_and_grad", **inputs))
            invoke()
            if path == "mlx":
                mx.reset_peak_memory()
            elif path == "judge":
                tracemalloc.start()
            before_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            times = []
            for _ in range(3):
                start = time.perf_counter()
                out = invoke()
                times.append(time.perf_counter() - start)
            if path == "judge":
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                method = "tracemalloc peak Python allocations; excludes native PyTorch storage"
            elif path == "mlx":
                peak = mx.get_peak_memory()
                method = "MLX allocator peak after reset; process RSS high-water mark also reported"
            else:
                peak = max(0, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss - before_children)
                method = "RUSAGE_CHILDREN ru_maxrss high-water delta in macOS bytes; cumulative and may be zero"
            row = {"wall_s": statistics.median(times), "peak_mem_mb": peak / 2**20, "mem_method": method}
            if path == "mlx":
                row["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            if path == "metal":
                row["children_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
            path_results[size] = row
            if size == "medium":
                medium_outputs[path] = out
    differences = {p: relative(medium_outputs[p], medium_outputs["judge"]) for p in ("mlx", "metal")}
    payload = {"project": "21-checkpointing", "sizes": SIZES, "results": results,
               "medium_max_rel_diff": differences,
               "notes": "Median of three calls after one warmup; includes harness serialization and Metal process startup. "
                        "Memory values are MiB. Child ru_maxrss is a cumulative high-water mark, not an individual-call peak; "
                        "the warmup can make its measured delta zero. The reference uses float64 Deepwave on CPU. "
                        "Timings and sizes are estimates until run on the target machine."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
