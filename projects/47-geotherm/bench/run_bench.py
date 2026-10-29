"""Benchmark the cooling half-space forward model through the project harness."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

PARAMS = {"surface_temp_c": 0.0, "mantle_temp_c": 1350.0,
          "diffusivity_m2_s": 1e-6, "conductivity_w_m_k": 3.3}
SIZES = {"small": {"n_age": 1200, "n_depth": 1200},
         "medium": {"n_age": 4000, "n_depth": 4000},
         "large": {"n_age": 12000, "n_depth": 12000}}


def inputs(size):
    return {"depth_m": np.linspace(0, 200_000, size["n_depth"], dtype=np.float32),
            "age_myr": np.linspace(1, 200, size["n_age"], dtype=np.float32),
            **PARAMS}


def judge(args):
    spec = importlib.util.spec_from_file_location("geotherm_reference", ROOT / "judge" / "make_goldens.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    temperature, flux = mod.forward(args["depth_m"], args["age_myr"], PARAMS)
    return {"temperature_c": temperature, "heat_flow_w_m2": flux}


def metal_worker(size_name):
    args = inputs(SIZES[size_name])
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    out = call("metal", "geotherm", **args)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    del out
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": max(0, after - before) / 1e6}))


def run_metal(size_name):
    # Each worker has a fresh RUSAGE_CHILDREN high-water mark.
    def once():
        result = subprocess.run([sys.executable, __file__, "--metal-worker", size_name],
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    once()
    samples = [once() for _ in range(3)]
    return {"wall_s": statistics.median(s["wall_s"] for s in samples),
            "peak_mem_mb": max(s["peak_mem_mb"] for s in samples),
            "mem_method": "Metal child RUSAGE_CHILDREN ru_maxrss delta in a fresh worker, bytes on macOS; maximum of repeats"}


def run_in_process(path, args):
    if path == "mlx":
        import mlx.core as mx
        invoke = lambda: call("mlx", "geotherm", **args)
    else:
        invoke = lambda: judge(args)
    warm = invoke()
    del warm
    if path == "mlx":
        mx.reset_peak_memory()
    else:
        tracemalloc.start()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        out = invoke()
        samples.append(time.perf_counter() - start)
        del out
    if path == "mlx":
        peak = mx.get_peak_memory() / 1e6
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        method = f"MLX get_peak_memory after reset; process peak RSS {rss:.1f} MB (ru_maxrss, macOS bytes)"
    else:
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        method = "tracemalloc peak during repeats, including traced NumPy allocations"
    return {"wall_s": statistics.median(samples), "peak_mem_mb": peak, "mem_method": method}


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {name}", flush=True)
            path_results[name] = run_metal(name) if path == "metal" else run_in_process(path, args)
    args = inputs(SIZES["medium"])
    ref = judge(args)
    differences = {}
    for path in ("mlx", "metal"):
        out = call(path, "geotherm", **args)
        # Scale each output by its reference maximum, avoiding ratios at zero.
        differences[path] = max(float(np.max(np.abs(out[k] - ref[k])) / np.max(np.abs(ref[k])))
                                for k in ref)
        del out
    report = {"project": "47-geotherm", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": differences,
              "notes": "Forward geotherm: deterministic 1–200 Myr and 0–200 km grids. "
                       "Median of three wall times after one warmup; harness conversion and Metal "
                       "process and file I/O are included. Medium difference is the maximum over "
                       "outputs of max absolute error divided by that output's reference max magnitude. "
                       "Memory measures differ by path and are not directly comparable. "
                       "Large runtime is hardware and storage dependent."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-worker":
        metal_worker(sys.argv[2])
    else:
        main()
