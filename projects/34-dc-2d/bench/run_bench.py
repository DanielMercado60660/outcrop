"""Benchmark the 2.5D DC solver from the project directory."""
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
sys.path.insert(0, str(PROJECT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(PROJECT)
from outcrop_harness import call

SIZES = {
    "small": {"nx": 25, "nz": 16, "observations": 2, "wavenumbers": 5},
    "medium": {"nx": 61, "nz": 32, "observations": 8, "wavenumbers": 12},
    "large": {"nx": 121, "nz": 60, "observations": 16, "wavenumbers": 24},
}


def inputs(size):
    nx, nz, n, nk = (size[k] for k in ("nx", "nz", "observations", "wavenumbers"))
    hx = np.full(nx, 0.5, dtype=np.float32)
    hz = np.full(nz, 0.5, dtype=np.float32)
    origin = np.array([-0.25 * nx, -0.5 * nz], dtype=np.float32)
    x = origin[0] + (np.arange(nx) + 0.5) * 0.5
    z = origin[1] + (np.arange(nz) + 0.5) * 0.5
    conductivity = np.full((nx, nz), 0.01, dtype=np.float32)
    conductivity[(np.abs(x[:, None] - 0.5) < 1.5) & (z[None, :] < -1) & (z[None, :] > -4)] = 0.04
    centers = np.linspace(-2, 2, n)
    spacing = np.linspace(0.75, 1.25, n)
    ab = np.column_stack((centers - 1.5 * spacing, centers + 1.5 * spacing))
    mn = np.column_stack((centers - 0.5 * spacing, centers + 0.5 * spacing))
    ky = np.logspace(-4, 1, nk)
    delta = np.diff(ky) / 2
    weights = np.r_[delta, 0] + np.r_[0, delta]
    weights[0] += ky[0]
    return {"hx": hx, "hz": hz, "origin": origin, "conductivity": conductivity,
            "ab": ab.astype(np.float32), "mn": mn.astype(np.float32),
            "currents": np.ones(n, dtype=np.float32), "ky": ky.astype(np.float32),
            "weights": (weights / np.pi).astype(np.float32)}


def reference(data):
    import discretize
    from simpeg.electromagnetics.static import resistivity as dc

    mesh = discretize.TensorMesh([data["hx"], data["hz"]], origin=data["origin"])
    sources = []
    for (a, b), (m, n), current in zip(data["ab"], data["mn"], data["currents"]):
        receiver = dc.receivers.Dipole(np.array([[m, 0.0]]), np.array([[n, 0.0]]))
        sources.append(dc.sources.Dipole([receiver], np.array([a, 0.0]),
                                         np.array([b, 0.0]), current=float(current)))
    sim = dc.Simulation2DCellCentered(mesh, survey=dc.Survey(sources), bc_type="Robin",
                                     nky=len(data["ky"]), do_trap=True)
    sim._quad_points = data["ky"].astype(np.float64)
    sim._quad_weights = data["weights"].astype(np.float64)
    sim.sigma = data["conductivity"].ravel(order="F").astype(np.float64)
    voltage = sim.dpred()
    a, b = data["ab"].T.astype(np.float64)
    m, n = data["mn"].T.astype(np.float64)
    geom = (1 / abs(a - m) - 1 / abs(b - m) - 1 / abs(a - n) + 1 / abs(b - n)) / (2 * np.pi)
    return {"voltage": voltage, "apparent_resistivity": voltage / data["currents"] / geom}


def measure(path, data):
    if path == "mlx":
        import mlx.core as mx
    run = (lambda: reference(data)) if path == "judge" else (lambda: call(path, "dc_forward", **data))
    run()
    times, peaks, rss = [], [], []
    out = None
    for _ in range(3):
        if path == "judge":
            tracemalloc.start()
        elif path == "mlx":
            mx.reset_peak_memory()
        else:
            child_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        out = run()
        times.append(time.perf_counter() - start)
        if path == "judge":
            peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
            tracemalloc.stop()
        elif path == "mlx":
            peaks.append(mx.get_peak_memory() / 1024**2)
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        else:
            child_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, child_after - child_before) / 1024**2)
    methods = {"judge": "tracemalloc peak during call (Python allocations)",
               "mlx": "MLX peak allocated memory after reset; process RSS high-water also reported",
               "metal": "RUSAGE_CHILDREN ru_maxrss delta, macOS bytes; cumulative child high-water mark"}
    result = {"wall_s": statistics.median(times), "peak_mem_mb": max(peaks), "mem_method": methods[path]}
    if rss:
        result["process_peak_rss_mb"] = max(rss)
    return result, out


def main():
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium = {}
    for name, size in SIZES.items():
        data = inputs(size)
        for path, path_results in results.items():
            if path == "judge" and name == "large":
                path_results[name] = {"skipped": "Potentially exceeds 120 s for 24 modes and 16 sources"}
                continue
            print(f"{path} {name}", flush=True)
            path_results[name], out = measure(path, data)
            if name == "medium":
                medium[path] = out
    diffs = {}
    for path in ("mlx", "metal"):
        diffs[path] = max(float(np.max(np.abs(medium[path][key] - medium["judge"][key]) /
                                       np.maximum(np.abs(medium["judge"][key]), 1e-30)))
                          for key in ("voltage", "apparent_resistivity"))
    report = {"project": "34-dc-2d", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": diffs,
              "notes": "Deterministic synthetic block; 0.5 m cells, trapezoidal Fourier weights. "
              "Times include setup and harness serialization. Memory uses macOS ru_maxrss bytes. "
              "Child ru_maxrss is a cumulative high-water mark, so its delta can be zero even "
              "when a child uses substantial memory; tracemalloc excludes most native allocations. "
              "Sizes and timing targets are estimates and have not been run on the target GPU."}
    (PROJECT / "bench" / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
