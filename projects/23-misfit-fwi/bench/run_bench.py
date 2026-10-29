"""Benchmark deterministic envelope objectives and gradients."""
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
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

spec = importlib.util.spec_from_file_location("misfit_oracle", ROOT / "judge" / "make_goldens.py")
oracle_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_module)

SIZES = {
    "small": {"ntrace": 128, "ntime": 1024, "dt": 0.002, "kind": "envelope", "epsilon": 0.01},
    "medium": {"ntrace": 512, "ntime": 4096, "dt": 0.002, "kind": "envelope", "epsilon": 0.01},
    "large": {"ntrace": 2048, "ntime": 8192, "dt": 0.002, "kind": "envelope", "epsilon": 0.01},
}


def inputs(params):
    rng = np.random.default_rng(2301)
    shape = (params["ntrace"], params["ntime"])
    return {"predicted": rng.standard_normal(shape, dtype=np.float32),
            "observed": rng.standard_normal(shape, dtype=np.float32),
            "dt": params["dt"], "kind": params["kind"], "epsilon": params["epsilon"]}


def run(path, kwargs):
    if path == "judge":
        return oracle_module.oracle(**kwargs)
    return call(path, "misfit_and_grad", **kwargs)


def measure(path, kwargs):
    run(path, kwargs)
    times, peaks, rss_peaks = [], [], []
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
            tracemalloc.reset_peak()
        else:
            mx.reset_peak_memory()
        start = time.perf_counter()
        result = run(path, kwargs)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
            tracemalloc.stop()
        else:
            peaks.append(mx.get_peak_memory() / 1024**2)
            rss_peaks.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        del result
    method = ("tracemalloc peak, MiB, including traced NumPy allocations during call"
              if path == "judge" else
              "MLX reset_peak_memory/get_peak_memory, MiB; RSS is process lifetime peak")
    entry = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks), "mem_method": method}
    if rss_peaks:
        entry["process_peak_rss_mb"] = max(rss_peaks)
    return entry


def relative_difference(actual, expected):
    errors = []
    for key in ("objective", "gradient"):
        a = np.asarray(actual[key], dtype=np.float64)
        b = np.asarray(expected[key], dtype=np.float64)
        denominator = np.max(np.abs(b))
        difference = np.max(np.abs(a - b))
        errors.append(float(difference / denominator) if denominator else float(difference))
    return max(errors)


def main():
    results = {"judge": {}, "mlx": {}}
    diffs = {}
    for size, params in SIZES.items():
        kwargs = inputs(params)
        for path, entries in results.items():
            entries[size] = measure(path, kwargs)
        if size == "medium":
            diffs["mlx"] = relative_difference(run("mlx", kwargs), run("judge", kwargs))
    report = {
        "project": "23-misfit-fwi", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": diffs,
        "notes": ("Envelope objective and gradient, three timed calls after one warm-up; "
                  "median wall time includes input conversion and output synchronization. "
                  "Memory values are MiB. Inputs are allocated before measurement. "
                  "RSS is the process lifetime peak, not an isolated per-call delta. "
                  "Medium error is the larger objective or gradient global infinity-norm "
                  "relative difference. Runtimes depend on hardware."),
    }
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
