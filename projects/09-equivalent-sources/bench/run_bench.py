"""Benchmark deterministic damped equivalent-source fits through the locked API."""
from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
sys.path.insert(0, str(WORKSPACE / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)

from outcrop_harness import call

SIZES = {
    "small": {"observations": 768, "sources": 64, "evaluation": 1024},
    "medium": {"observations": 2048, "sources": 160, "evaluation": 4096},
    "large": {"observations": 4096, "sources": 320, "evaluation": 16384},
}


def make_inputs(observations: int, sources: int, evaluation: int) -> dict:
    """Create a reproducible survey over a two-source synthetic field."""
    rng = np.random.default_rng(93421)
    xy = rng.uniform([0, 0], [1000, 800], (observations, 2))
    coordinates = np.column_stack((xy, rng.uniform(65, 95, observations)))
    nx = int(np.ceil(np.sqrt(sources * 1000 / 800)))
    ny = int(np.ceil(sources / nx))
    sx, sy = np.meshgrid(np.linspace(0, 1000, nx), np.linspace(0, 800, ny))
    points = np.column_stack((sx.ravel()[:sources], sy.ravel()[:sources],
                              np.full(sources, -240.0)))
    side = int(np.ceil(np.sqrt(evaluation)))
    ex, ey = np.meshgrid(np.linspace(0, 1000, side), np.linspace(0, 800, side))
    eval_xyz = np.column_stack((ex.ravel()[:evaluation], ey.ravel()[:evaluation],
                                np.full(evaluation, 120.0)))
    truth = np.array([[300, 280, -350], [760, 570, -420]])
    data = (1 / np.linalg.norm(coordinates[:, None] - truth, axis=2)) @ [1800, -1050]
    return {
        "coordinates": coordinates.astype(np.float32),
        "data": data.astype(np.float32),
        "points": points.astype(np.float32),
        "evaluation": eval_xyz.astype(np.float32),
        "weights": rng.uniform(0.6, 1.8, observations).astype(np.float32),
        "damping": 0.03,
        "method": "standard",
        "data_indices": np.empty(0, dtype=np.int32),
        "data_offsets": np.zeros(1, dtype=np.int32),
        "source_indices": np.empty(0, dtype=np.int32),
        "source_offsets": np.zeros(1, dtype=np.int32),
    }


def judge(args: dict) -> dict:
    """Use the same Harmonica reference and float64 promotion as the goldens."""
    import harmonica as hm

    model = hm.EquivalentSources(
        points=tuple(args["points"].astype(np.float64).T),
        damping=args["damping"], parallel=False, dtype="float64",
    )
    model.fit(tuple(args["coordinates"].astype(np.float64).T),
              args["data"].astype(np.float64),
              weights=args["weights"].astype(np.float64))
    return {"prediction": model.predict(tuple(args["evaluation"].astype(np.float64).T))}


def run(path: str, args: dict) -> dict:
    return judge(args) if path == "judge" else call(path, "equivalent_sources", **args)


def measure(path: str, args: dict) -> tuple[dict, dict]:
    # The harness materializes MLX arrays and waits for Metal before returning.
    run(path, args)
    times = []
    peaks = []
    rss = []
    output = None
    if path == "mlx":
        import mlx.core as mx
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
        elif path == "mlx":
            mx.reset_peak_memory()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        output = run(path, args)
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory() / 1024**2)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, after - before) / 1024**2)
    methods = {
        "judge": "tracemalloc peak during call; excludes untracked native allocations",
        "mlx": "MLX reset_peak_memory/get_peak_memory; process RSS high-water also reported",
        "metal": "RUSAGE_CHILDREN ru_maxrss high-water delta, macOS bytes; may be zero after earlier larger children",
    }
    result = {"wall_s": float(statistics.median(times)),
              "peak_mem_mb": float(max(peaks)), "mem_method": methods[path]}
    if rss:
        result["process_peak_rss_mb"] = float(max(rss))
    return result, output


def main() -> None:
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium_outputs = {}
    for size, params in SIZES.items():
        args = make_inputs(**params)
        for path, path_results in results.items():
            print(f"Benchmarking {path} / {size}", flush=True)
            result, output = measure(path, args)
            path_results[size] = result
            if size == "medium":
                medium_outputs[path] = output["prediction"].astype(np.float64)
    reference = medium_outputs["judge"]
    denominator = float(np.max(np.abs(reference)))
    differences = {
        path: float(np.max(np.abs(medium_outputs[path] - reference)) / denominator)
        for path in ("mlx", "metal")
    }
    report = {
        "project": PROJECT.name, "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": ("Positive damping 0.03; deterministic standard fit. Wall times include "
                  "harness conversion and Metal process startup and file I/O. Memory is "
                  "MiB; methods differ and are not directly comparable. Child RSS is a "
                  "cumulative high-water mark, so a delta is not a per-child peak. "
                  "Relative difference is max absolute error divided by max absolute "
                  "reference prediction. Large runtime is an unverified estimate."),
    }
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
