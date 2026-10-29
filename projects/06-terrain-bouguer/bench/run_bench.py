"""Benchmark the signed terrain prism layer from the project directory."""
import json
import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import harmonica as hm
import mlx.core as mx
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"nx": 32, "ny": 32, "observations": 80},
    "medium": {"nx": 64, "ny": 64, "observations": 256},
    "large": {"nx": 128, "ny": 128, "observations": 512},
}
G_MGAL = 6.67430e-6


def make_inputs(nx, ny, observations):
    rng = np.random.default_rng(20261028)
    x = np.arange(nx, dtype=np.float32) * 200
    y = np.arange(ny, dtype=np.float32) * 200
    xx, yy = np.meshgrid(x, y)
    surface = (180 + 80 * np.sin(xx / 1100) * np.cos(yy / 1300)).astype(np.float32)
    reference = np.zeros_like(surface)
    density = (2670 + 100 * np.cos(xx / 1700)).astype(np.float32)
    coords = np.column_stack((rng.uniform(x[0], x[-1], observations),
                              rng.uniform(y[0], y[-1], observations),
                              np.full(observations, 700.0))).astype(np.float32)
    return {"easting": x, "northing": y, "surface": surface, "reference": reference,
            "density": density, "coordinates": coords}


def judge(easting, northing, surface, reference, density, coordinates):
    layer = hm.prism_layer((easting.astype(np.float64), northing.astype(np.float64)),
                           surface.astype(np.float64), reference.astype(np.float64),
                           properties={"density": density.astype(np.float64) *
                                       np.sign(surface - reference)})
    gz = layer.prism_layer.gravity(tuple(coordinates.astype(np.float64).T),
                                   field="g_z", parallel=False, dtype="float64")
    ix = np.clip(np.floor((coordinates[:, 0] - easting[0]) /
                          (easting[1] - easting[0]) + 0.5).astype(int), 0, len(easting)-1)
    iy = np.clip(np.floor((coordinates[:, 1] - northing[0]) /
                          (northing[1] - northing[0]) + 0.5).astype(int), 0, len(northing)-1)
    slab = 2 * np.pi * G_MGAL * density[iy, ix].astype(np.float64) * (
        surface[iy, ix].astype(np.float64) - reference[iy, ix].astype(np.float64))
    return {"layer_gz": gz, "slab_gz": slab, "terrain_gz": gz-slab, "correction_gz": -gz}


def rss_mb(who):
    value = resource.getrusage(who).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def benchmark(path, inputs):
    run = (lambda: judge(**inputs)) if path == "judge" else (
        lambda: call(path, "terrain_bouguer", **inputs))
    run()  # Materializes MLX arrays and waits for the Metal child.
    if path == "mlx":
        mx.reset_peak_memory()
    child_before = rss_mb(resource.RUSAGE_CHILDREN)
    if path == "judge":
        tracemalloc.start()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        output = run()
        times.append(time.perf_counter() - start)
    result = {"wall_s": float(np.median(times))}
    if path == "judge":
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.update(peak_mem_mb=peak / 1024**2,
                      mem_method="tracemalloc peak during timed repeats; excludes native allocations")
    elif path == "mlx":
        result.update(peak_mem_mb=mx.get_peak_memory() / 1024**2,
                      process_peak_rss_mb=rss_mb(resource.RUSAGE_SELF),
                      mem_method="MLX peak allocator after reset; process_peak_rss_mb is lifetime RSS high-water")
    else:
        child_after = rss_mb(resource.RUSAGE_CHILDREN)
        result.update(peak_mem_mb=max(0.0, child_after - child_before),
                      child_peak_rss_mb=child_after,
                      mem_method="RUSAGE_CHILDREN ru_maxrss high-water delta after warmup; child_peak_rss_mb is cumulative")
    return result, output


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium = {}
    for size, params in SIZES.items():
        inputs = make_inputs(**params)
        for path, path_results in results.items():
            print(f"{size}: {path}", flush=True)
            path_results[size], output = benchmark(path, inputs)
            if size == "medium":
                medium[path] = output
    differences = {}
    for path in ("mlx", "metal"):
        differences[path] = max(float(np.max(np.abs(medium[path][key] - ref))) /
                                max(float(np.max(np.abs(ref))), 1e-12)
                                for key, ref in medium["judge"].items())
    payload = {"project": PROJECT.name, "sizes": SIZES, "results": results,
               "medium_max_rel_diff": differences,
               "notes": "Fixed seed 20261028; 200 m cells; 700 m observation altitude. "
                        "Median of three calls after one warmup; includes harness transfer and "
                        "Metal process startup and shader compilation. Relative differences "
                        "are max absolute errors normalized by each reference output's maximum "
                        "absolute value. Memory measures are not directly comparable; Metal "
                        "child high-water delta can be zero after warmup."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
