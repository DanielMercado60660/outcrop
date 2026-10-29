"""Benchmark tesseroid gravity from the project root."""
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
    "small": {"observations": 8, "sources": 1},
    "medium": {"observations": 64, "sources": 2},
    "large": {"observations": 256, "sources": 4},
}


def inputs(size):
    rng = np.random.default_rng(20261028)
    count = size["observations"]
    coordinates = np.column_stack((rng.uniform(-2.5, 2.5, count),
                                   rng.uniform(-2.5, 2.5, count),
                                   np.full(count, 6_391_000))).astype(np.float32)
    centers = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
    tesseroids = np.array([[x - 0.45, x + 0.45, y - 0.45, y + 0.45,
                            6_351_000, 6_371_000]
                           for x, y in centers[:size["sources"]]], dtype=np.float32)
    density = np.full(size["sources"], 2670, dtype=np.float32)
    return {"coordinates": coordinates, "tesseroids": tesseroids, "density": density,
            "field": "g_z", "radial_adaptive": True}


def load_reference():
    spec = importlib.util.spec_from_file_location("outcrop_reference", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference


def main():
    import mlx.core as mx

    reference = load_reference()

    def execute(path, args):
        if path == "judge":
            return reference(args["coordinates"], args["tesseroids"], args["density"],
                             args["field"], args["radial_adaptive"])
        return call(path, "tesseroid_gravity", **args)["field"]

    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium = {}
    for name, size in SIZES.items():
        args = inputs(size)
        for path, path_results in results.items():
            execute(path, args)
            times, peaks, rss_values = [], [], []
            for _ in range(3):
                if path == "judge":
                    tracemalloc.start()
                    tracemalloc.reset_peak()
                elif path == "mlx":
                    mx.reset_peak_memory()
                else:
                    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                start = time.perf_counter()
                output = execute(path, args)
                times.append(time.perf_counter() - start)
                if path == "judge":
                    peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
                    tracemalloc.stop()
                elif path == "mlx":
                    peaks.append(mx.get_peak_memory() / 1e6)
                    rss_values.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
                else:
                    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                    peaks.append(max(0, after - before) / 1e6)
            methods = {
                "judge": "tracemalloc peak Python allocations during call",
                "mlx": "MLX reset_peak_memory/get_peak_memory; process peak RSS also reported",
                "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta (macOS bytes)",
            }
            entry = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks),
                     "mem_method": methods[path]}
            if rss_values:
                entry["process_peak_rss_mb"] = max(rss_values)
            path_results[name] = entry
            if name == "medium":
                medium[path] = np.asarray(output, dtype=np.float64)
            print(f"{path:5s} {name:6s}: {entry['wall_s']:.3f} s", flush=True)
    reference_values = medium["judge"]
    diff = {path: float(np.max(np.abs(medium[path] - reference_values)
                               / np.maximum(np.abs(reference_values), 1e-30)))
            for path in ("mlx", "metal")}
    report = {"project": "04-tesseroid-gravity", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": diff,
              "notes": "g_z in mGal; radial adaptive subdivision; seed 20261028. "
              "Wall time includes harness I/O and Metal process startup. Memory values are decimal MB. "
              "Child ru_maxrss is a cumulative high-water mark, so its delta can be zero after "
              "warmup and is not the child's actual peak. tracemalloc excludes native allocations; "
              "process RSS is a lifetime high-water mark. MLX peak includes its live allocation baseline."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
