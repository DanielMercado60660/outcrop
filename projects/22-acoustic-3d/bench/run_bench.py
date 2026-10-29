"""Benchmark the Deepwave reference and the release Metal executable."""
import ast
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"shape": [48, 48, 48], "steps": 320, "spacing": 10.0, "dt": 0.0015},
    "medium": {"shape": [96, 96, 96], "steps": 200, "spacing": 10.0, "dt": 0.0015},
    "large": {"shape": [192, 192, 192], "steps": 500, "spacing": 10.0, "dt": 0.0015},
}


def inputs(size):
    rng = np.random.default_rng(22001)
    shape = tuple(size["shape"])
    velocity = rng.uniform(1800, 2200, shape).astype(np.float32)
    initial = rng.normal(0, 0.1, shape).astype(np.float32)
    previous = initial.copy()
    return {"velocity": velocity, "initial": initial, "previous": previous,
            "spacing": size["spacing"], "dt": size["dt"], "steps": size["steps"]}


def reference_function():
    # Importing make_goldens directly would overwrite the locked fixtures.
    import deepwave
    import torch

    torch.set_num_threads(1)
    source = (PROJECT / "judge" / "make_goldens.py").read_text()
    tree = ast.parse(source)
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "propagate")
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {"deepwave": deepwave, "torch": torch, "np": np, "warnings": warnings}
    exec(compile(module, str(PROJECT / "judge" / "make_goldens.py"), "exec"), namespace)  # noqa: S102
    return namespace["propagate"]


def judge(args, ref):
    final, last = ref(args["velocity"].astype(np.float64),
                      args["initial"].astype(np.float64),
                      args["previous"].astype(np.float64),
                      args["spacing"], args["dt"], args["steps"])
    return {"final": final, "last": last}


def metal_sample(size_name):
    # A fresh worker has no earlier children, so the high-water mark is local.
    args = inputs(SIZES[size_name])
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    call("metal", "propagate", **args)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": (after - before) / 1024**2}))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-sample":
        metal_sample(sys.argv[2])
        return
    ref = reference_function()
    results = {"judge": {}, "metal": {}}
    medium_reference = None
    for name, size in SIZES.items():
        args = inputs(size)
        print(f"Benchmarking {name}", flush=True)
        if name == "large":
            results["judge"][name] = {"skipped": "Single-threaded Deepwave reference may exceed 120 s at this size."}
        else:
            judge(args, ref)  # warm-up
            times, peaks = [], []
            for _ in range(3):
                tracemalloc.start()
                start = time.perf_counter()
                output = judge(args, ref)
                times.append(time.perf_counter() - start)
                peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
                tracemalloc.stop()
            if name == "medium":
                medium_reference = output
            results["judge"][name] = {
                "wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
                "mem_method": "tracemalloc peak; excludes most native PyTorch/Deepwave allocations",
            }
        call("metal", "propagate", **args)  # warm-up
        samples = []
        for _ in range(3):
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--metal-sample", name], check=True,
                                  capture_output=True, text=True)
            samples.append(json.loads(proc.stdout))
        results["metal"][name] = {
            "wall_s": statistics.median(s["wall_s"] for s in samples),
            "peak_mem_mb": max(s["peak_mem_mb"] for s in samples),
            "mem_method": "fresh worker RUSAGE_CHILDREN ru_maxrss delta, macOS bytes; child RSS, not GPU allocation",
        }
    medium_output = call("metal", "propagate", **inputs(SIZES["medium"]))
    diff = max(float(np.max(np.abs(medium_output[k] - medium_reference[k])) /
                     max(np.max(np.abs(medium_reference[k])), 1e-30))
               for k in ("final", "last"))
    report = {
        "project": PROJECT.name, "sizes": SIZES, "results": results,
        "medium_max_rel_diff": {"metal": diff},
        "notes": "Fixed seed 22001; float64 Deepwave reference and float32 Metal. "
                 "Times include harness serialization, process startup, shader compilation and output loading. "
                 "Memory is the maximum of three repeats; judge tracemalloc omits native allocations. "
                 "Relative difference is max absolute error divided by max reference amplitude, "
                 "maximized over final and last. Periodic boundaries; no source or damping.",
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
