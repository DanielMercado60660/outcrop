"""Benchmark the ObsPy reference and Metal implementation."""
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
from obspy.signal.cross_correlation import correlate_template

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"stations": 2, "samples": 8192, "templates": 4, "template_samples": 128},
    "medium": {"stations": 4, "samples": 16384, "templates": 8, "template_samples": 192},
    "large": {"stations": 8, "samples": 32768, "templates": 16, "template_samples": 256},
}


def inputs(size):
    rng = np.random.default_rng(3801)
    s, n, t, m = (size[k] for k in ("stations", "samples", "templates", "template_samples"))
    data = rng.normal(size=(s, n)).astype(np.float32)
    templates = rng.normal(size=(t, s, m)).astype(np.float32)
    data[:, n // 3:n // 3 + m] += templates[0] * 2
    return {"data": data, "templates": templates, "threshold": 0.8}


def reference(data, templates, threshold):
    t, s, m = templates.shape
    corr = np.empty((t, s, data.shape[1] - m + 1), dtype=np.float64)
    for ti in range(t):
        for si in range(s):
            corr[ti, si] = correlate_template(
                data[si].astype(np.float64), templates[ti, si].astype(np.float64),
                mode="valid", normalize="full", demean=True, method="direct",
            )
    return {"correlation": corr, "detections": (corr >= threshold).astype(np.float32)}


def worker(path, size_name):
    args = inputs(SIZES[size_name])
    run = (lambda: reference(**args)) if path == "judge" else (
        lambda: call("metal", "template_match", **args)
    )
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    run()
    if path == "judge":
        tracemalloc.start()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        output = run()
        times.append(time.perf_counter() - start)
    if path == "judge":
        peak = tracemalloc.get_traced_memory()[1] / 1024**2
        tracemalloc.stop()
        method = "tracemalloc peak during timed calls (Python and traced NumPy allocations)"
    else:
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        divisor = 1024**2 if sys.platform == "darwin" else 1024
        peak = max(0, child_after - child_before) / divisor
        method = "RUSAGE_CHILDREN ru_maxrss high-water delta, fresh worker; includes warm-up child"
    result = {"wall_s": statistics.median(times), "peak_mem_mb": peak, "mem_method": method}
    if size_name == "medium" and path == "metal":
        truth = reference(**args)["correlation"]
        result["max_rel_diff"] = float(np.max(np.abs(output["correlation"] - truth)) / np.max(np.abs(truth)))
    return result


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        print(json.dumps(worker(sys.argv[2], sys.argv[3])))
        return
    results = {"judge": {}, "metal": {}}
    differences = {}
    for path, path_results in results.items():
        for name in SIZES:
            if path == "judge" and name == "large":
                path_results[name] = {"skipped": "Direct float64 correlation may exceed 120 seconds at this size."}
                continue
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", path, name],
                check=True, capture_output=True, text=True,
            )
            result = json.loads(proc.stdout)
            if "max_rel_diff" in result:
                differences[path] = result.pop("max_rel_diff")
            path_results[name] = result
            print(f"{path} {name}: {result['wall_s']:.3f} s", flush=True)
    report = {
        "project": PROJECT.name, "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": "Median of three calls after one warm-up; Metal timing includes harness file I/O, process startup and shader compilation. Relative difference is maximum absolute correlation error divided by maximum absolute reference correlation. ru_maxrss is a child high-water mark, not GPU allocation; memory is in MiB. Large runtimes depend on hardware and have not been calibrated on M5 Max.",
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
