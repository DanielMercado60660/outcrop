"""Benchmark neural-parameterized acoustic objective and gradient."""
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

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"nz": 12, "nx": 16, "shots": 1, "receivers": 6, "nt": 32},
    "medium": {"nz": 24, "nx": 32, "shots": 3, "receivers": 15, "nt": 280},
    "large": {"nz": 64, "nx": 96, "shots": 4, "receivers": 32, "nt": 1200},
}


def inputs(size):
    rng = np.random.default_rng(20261020)
    nz, nx, shots, receivers, nt = (size[k] for k in
                                    ("nz", "nx", "shots", "receivers", "nt"))
    latent = rng.normal(size=4).astype(np.float32)
    parameters = rng.normal(scale=0.1, size=355).astype(np.float32)
    times = np.arange(nt) * 0.001
    a = (np.pi * 18 * (times - 0.06)) ** 2
    wavelet = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)
    source = np.tile(wavelet, (shots, 1, 1))
    source_locations = np.zeros((shots, 1, 2), dtype=np.int64)
    source_locations[:, 0, 0] = 2
    source_locations[:, 0, 1] = np.linspace(2, nx - 3, shots, dtype=np.int64)
    receiver_locations = np.zeros((shots, receivers, 2), dtype=np.int64)
    receiver_locations[:, :, 0] = nz - 3
    receiver_locations[:, :, 1] = np.linspace(1, nx - 2, receivers, dtype=np.int64)
    return {"parameters": parameters, "latent": latent,
            "initial_velocity": np.full((nz, nx), 2000, dtype=np.float32),
            "observed": rng.normal(scale=0.1, size=(shots, receivers, nt)).astype(np.float32),
            "source_amplitudes": source, "source_locations": source_locations,
            "receiver_locations": receiver_locations, "spacing": 10.0, "dt": 0.001,
            "data_scale": 1.0, "velocity_scale": 500.0, "coarse_shape": [5, 7],
            "hidden_size": 8, "parameterization": "neural"}


def load_judge():
    spec = importlib.util.spec_from_file_location("nnfwi_judge", PROJECT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def judge_call(judge, kwargs):
    import torch

    g = {k: torch.tensor(v, dtype=torch.int64 if v.dtype.kind in "iu" else torch.float64)
         if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
         if k != "parameters"}
    p = torch.tensor(kwargs["parameters"], dtype=torch.float64, requires_grad=True)
    loss, velocity, predicted = judge.evaluate(p, g, kwargs["parameterization"])
    gradient, = torch.autograd.grad(loss, p)
    return {"objective": loss.detach().numpy(), "gradient": gradient.detach().numpy(),
            "predicted": predicted.detach().numpy(), "velocity": velocity.detach().numpy()}


def measure(path, kwargs, judge):
    import mlx.core as mx

    def run():
        return judge_call(judge, kwargs) if path == "judge" else call(path, "misfit_and_grad", **kwargs)

    run()
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
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        method = "tracemalloc peak, Python allocations only; excludes native Torch allocations"
        result = {"wall_s": statistics.median(times), "peak_mem_mb": peak, "mem_method": method}
    else:
        result = {"wall_s": statistics.median(times), "peak_mem_mb": mx.get_peak_memory() / 1e6,
                  "mem_method": "MLX peak device allocation since reset; process RSS is lifetime high-water mark",
                  "process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    return result, output


def main():
    judge = load_judge()
    results = {"judge": {}, "mlx": {}}
    medium_outputs = {}
    for name, size in SIZES.items():
        kwargs = inputs(size)
        for path, path_results in results.items():
            print(f"{path}: {name}", flush=True)
            path_results[name], output = measure(path, kwargs, judge)
            if name == "medium":
                medium_outputs[path] = output
    reference, actual = medium_outputs["judge"], medium_outputs["mlx"]
    errors = []
    for key in reference:
        ref = np.asarray(reference[key], dtype=np.float64)
        got = np.asarray(actual[key], dtype=np.float64)
        denominator = float(np.max(np.abs(ref)))
        errors.append(float(np.max(np.abs(got - ref))) / denominator if denominator else
                      float(np.max(np.abs(got - ref))))
    report = {"project": "20-nnfwi", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": {"mlx": max(errors)},
              "notes": "Neural misfit_and_grad, 18 Hz source, dt=0.001 s, spacing=10 m; "
                       "fixed-seed synthetic observations. Three repeats after one warm-up. "
                       "Relative error is the maximum across outputs of max absolute difference "
                       "divided by each reference output's maximum absolute value. "
                       "Judge uses float64 CPU; MLX uses float32. MB means 10^6 bytes. "
                       "Process RSS is macOS ru_maxrss and includes earlier work. "
                       "Large-size runtime is an unmeasured estimate."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
