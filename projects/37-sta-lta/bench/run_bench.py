"""Benchmark ObsPy and the Metal classic STA/LTA implementation."""
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
from obspy.signal.trigger import classic_sta_lta

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"samples": 640_000, "nsta": 40, "nlta": 400},
    "medium": {"samples": 6_400_000, "nsta": 40, "nlta": 400},
    "large": {"samples": 64_000_000, "nsta": 40, "nlta": 400},
}


def make_trace(size):
    return np.random.default_rng(3701).standard_normal(size["samples"], dtype=np.float32)


def judge(trace, size):
    return classic_sta_lta(trace.astype(np.float64), size["nsta"], size["nlta"])


def metal(trace, size):
    return call("metal", "classic_sta_lta", trace=trace,
                nsta=size["nsta"], nlta=size["nlta"])["characteristic"]


def metal_worker(size_name):
    # A fresh process has no prior children to contaminate the high-water mark.
    size = SIZES[size_name]
    trace = make_trace(size)
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    result = metal(trace, size)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    assert result.shape == trace.shape
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": max(0, after - before) / 1024**2}))


def measure_metal(size_name):
    def run():
        output = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 "--metal-worker", size_name],
                                check=True, capture_output=True, text=True)
        return json.loads(output.stdout)

    run()
    samples = [run() for _ in range(3)]
    return {"wall_s": statistics.median(x["wall_s"] for x in samples),
            "peak_mem_mb": max(x["peak_mem_mb"] for x in samples),
            "mem_method": "max child ru_maxrss delta across three fresh workers (macOS bytes)"}


def measure_judge(trace, size):
    judge(trace, size)
    times = []
    peaks = []
    for _ in range(3):
        tracemalloc.start()
        start = time.perf_counter()
        result = judge(trace, size)
        times.append(time.perf_counter() - start)
        peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
        tracemalloc.stop()
        del result
    return {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
            "mem_method": "tracemalloc peak during call; input allocated before tracing"}


def main():
    results = {"judge": {}, "metal": {}}
    for name, size in SIZES.items():
        trace = make_trace(size)
        results["judge"][name] = measure_judge(trace, size)
        del trace
        results["metal"][name] = measure_metal(name)
        print(f"{name}: {results['judge'][name]['wall_s']:.3f}s judge, "
              f"{results['metal'][name]['wall_s']:.3f}s metal", flush=True)

    size = SIZES["medium"]
    trace = make_trace(size)
    reference = judge(trace, size)
    actual = metal(trace, size)
    rel_diff = float(np.max(np.abs(actual - reference)) / np.max(np.abs(reference)))
    report = {
        "project": "37-sta-lta", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": {"metal": rel_diff},
        "notes": ("Classic STA/LTA only; fixed-seed float32 Gaussian trace. Wall time "
                  "includes harness file I/O, subprocess startup, Metal compilation and "
                  "output loading for Metal, and float64 conversion for ObsPy. Metal "
                  "memory is the child process RSS high-water mark, not GPU allocations "
                  "or parent RSS; fresh workers avoid prior-child high-water marks. "
                  "Judge tracemalloc includes NumPy allocations tracked by Python, "
                  "but may omit native allocations. Sizes and runtimes are estimates "
                  "until measured on the target machine. Relative difference is max "
                  "absolute error divided by max absolute reference value."),
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-worker":
        metal_worker(sys.argv[2])
    else:
        main()
