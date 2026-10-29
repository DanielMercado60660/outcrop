"""Benchmark the phase loss and its gradient."""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"batch": 8, "samples": 3000, "calls": 16},
    "medium": {"batch": 32, "samples": 3000, "calls": 64},
    "large": {"batch": 128, "samples": 3000, "calls": 128},
}


def inputs(size):
    rng = np.random.default_rng(43003)
    shape = (size["batch"], size["samples"], 3)
    logits = rng.normal(0, 1.5, shape).astype(np.float32)
    targets = rng.random(shape, dtype=np.float32)
    targets /= targets.sum(axis=-1, keepdims=True)
    return {"logits": logits, "targets": targets}


def main():
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    loss64 = runpy.run_path(str(ROOT / "judge" / "make_goldens.py"))["loss64"]

    def judge(logits, targets):
        x = logits.astype(np.float64)
        y = targets.astype(np.float64)
        loss = loss64(x, y)
        shifted = x - x.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probability = exp / exp.sum(axis=-1, keepdims=True)
        return {"loss": np.asarray(loss),
                "gradient": (probability - y) / np.prod(x.shape[:-1])}

    results = {"judge": {}, "mlx": {}}
    diffs = {}
    for name, size in SIZES.items():
        args = inputs(size)
        for path in ("judge", "mlx"):
            def invoke(path=path, args=args):
                if path == "judge":
                    return judge(**args)
                return call("mlx", "phase_loss_and_grad", **args)

            invoke()  # NumPy outputs synchronize MLX.
            if path == "judge":
                tracemalloc.start()
            else:
                mx.reset_peak_memory()
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(size["calls"]):
                    out = invoke()
                timings.append(time.perf_counter() - start)
            if path == "judge":
                peak = tracemalloc.get_traced_memory()[1] / 2**20
                tracemalloc.stop()
                method = "tracemalloc peak during timed calls, including NumPy allocations"
            else:
                peak = mx.get_peak_memory() / 2**20
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_mb = rss / (2**20 if sys.platform == "darwin" else 1024)
                method = ("MLX peak allocated bytes since reset; process lifetime peak RSS "
                          f"{rss_mb:.1f} MiB (includes prior work)")
            results[path][name] = {"wall_s": statistics.median(timings),
                                   "peak_mem_mb": peak, "mem_method": method}
            if name == "medium" and path == "mlx":
                ref = judge(**args)
                diffs[path] = max(float(np.max(np.abs(out[k] - ref[k])) /
                                        max(float(np.max(np.abs(ref[k]))), 1e-30))
                                  for k in ref)
            print(f"{path} {name}: {results[path][name]}", flush=True)
    report = {"project": "43-phase-picker", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": diffs,
              "notes": "Median total wall time for the stated number of identical "
              "phase_loss_and_grad calls; input generation and warmup excluded. "
              "Judge uses loss64 and the analytic gradient from judge/make_goldens.py. "
              "Relative difference is the maximum over loss and gradient of max "
              "absolute error divided by the maximum absolute reference value. "
              "Training and inference are not benchmarked. MiB are reported as MB."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
