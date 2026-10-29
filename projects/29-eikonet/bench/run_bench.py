"""Benchmark EikoNet evaluation against the locked NumPy reference."""
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
    "small": {"pairs": 5243, "dimension": 2, "width": 32},
    "medium": {"pairs": 65536, "dimension": 2, "width": 32},
    "large": {"pairs": 524288, "dimension": 2, "width": 32},
}


def inputs(size):
    rng = np.random.default_rng(29012020)
    n, d, h = size["pairs"], size["dimension"], size["width"]
    p = rng.normal(0, 0.08, 2 * d * h + h * h + 3 * h + 1).astype(np.float32)
    return {"parameters": p, "sources": rng.uniform(0, 700, (n, d)).astype(np.float32),
            "receivers": rng.uniform(800, 2000, (n, d)).astype(np.float32),
            "origin": np.zeros(d, dtype=np.float32), "length_scale": 2000.0,
            "reference_velocity": 3000.0, "width": h}


def load_reference():
    spec = importlib.util.spec_from_file_location("eikonet_reference", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.numpy_evaluate


def main():
    reference = load_reference()
    results = {"judge": {}, "mlx": {}}
    medium_outputs = {}
    for label, size in SIZES.items():
        args = inputs(size)
        # The reference operates in float64 on the exact float32 API inputs.
        ref_args = {k: v.astype(np.float64) if isinstance(v, np.ndarray) else v
                    for k, v in args.items()}

        def judge(ref_args=ref_args):
            t, g = reference(ref_args["parameters"], ref_args["sources"],
                             ref_args["receivers"], ref_args["origin"],
                             ref_args["length_scale"], ref_args["reference_velocity"],
                             ref_args["width"])
            return {"time": t, "receiver_gradient": g}

        for path, fn in (("judge", judge), ("mlx", lambda args=args: call("mlx", "evaluate", **args))):
            fn()  # Warm up, including lazy evaluation and host conversion for MLX.
            times, peaks = [], []
            output = None
            for _ in range(3):
                if path == "judge":
                    tracemalloc.start()
                else:
                    mx.reset_peak_memory()
                start = time.perf_counter()
                output = fn()
                times.append(time.perf_counter() - start)
                if path == "judge":
                    peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
                    tracemalloc.stop()
                else:
                    peaks.append(mx.get_peak_memory() / 1e6)
            result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
                      "mem_method": ("tracemalloc peak during call (includes NumPy allocations)"
                                     if path == "judge" else "MLX peak allocated memory after reset")}
            if path == "mlx":
                result["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            results[path][label] = result
            if label == "medium":
                medium_outputs[path] = output
            print(f"{path} {label}: {result}", flush=True)

    diffs = []
    for key in ("time", "receiver_gradient"):
        actual = np.asarray(medium_outputs["mlx"][key], dtype=np.float64)
        expected = medium_outputs["judge"][key]
        diffs.append(float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected))))
    report = {"project": "29-eikonet", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": {"mlx": max(diffs)},
              "notes": "Evaluation and receiver gradient; fixed seed, one warmup and median of three calls. "
                       "Relative difference is the maximum across outputs of max absolute error divided by "
                       "the maximum absolute reference value for that output. MB uses 10^6 bytes. "
                       "MLX RSS is the macOS process lifetime high-water mark, not a per-call delta; "
                       "MLX peak includes live allocations at reset. Timings include host output conversion."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
