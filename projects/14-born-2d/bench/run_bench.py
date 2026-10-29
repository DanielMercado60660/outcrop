"""Benchmark forward 2-D Born modeling on the reference, MLX and Metal paths."""
import json
import os
import resource
import runpy
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"ny": 48, "nx": 48, "shots": 1, "receivers": 24, "nt": 100},
    "medium": {"ny": 128, "nx": 128, "shots": 2, "receivers": 96, "nt": 240},
    "large": {"ny": 256, "nx": 256, "shots": 4, "receivers": 192, "nt": 400},
}
MIB = 1024 * 1024


def make_case(size):
    ny, nx, shots, nr, nt = (size[k] for k in ("ny", "nx", "shots", "receivers", "nt"))
    rng = np.random.default_rng(1402)
    y, x = np.mgrid[:ny, :nx]
    v = (1500 + 100 * np.sin(y / ny * 2) * np.cos(x / nx * 2)).astype(np.float32)
    dv = (45 * np.exp(-((y - 0.58 * ny) ** 2 / (0.12 * ny) ** 2
                        + (x - 0.5 * nx) ** 2 / (0.15 * nx) ** 2))).astype(np.float32)
    t = np.arange(nt) * 0.001 - 0.06
    q = (np.pi * 15 * t) ** 2
    pulse = ((1 - 2 * q) * np.exp(-q)).astype(np.float32)
    amplitudes = np.broadcast_to(pulse, (shots, 1, nt)).copy()
    source_locations = np.zeros((shots, 1, 2), dtype=np.int64)
    source_locations[:, 0, 0] = max(4, ny // 8)
    source_locations[:, 0, 1] = np.linspace(nx // 4, 3 * nx // 4, shots, dtype=np.int64)
    receiver_locations = np.empty((shots, nr, 2), dtype=np.int64)
    receiver_locations[:, :, 0] = max(4, ny // 8)
    receiver_locations[:, :, 1] = np.linspace(2, nx - 3, nr, dtype=np.int64)
    v += rng.normal(0, 0.5, (ny, nx)).astype(np.float32)
    zero = np.zeros((shots, ny, nx), dtype=np.float32)
    return {"v": v, "dv": dv, "source_amplitudes": amplitudes,
            "source_locations": source_locations, "receiver_locations": receiver_locations,
            "wavefield_0": zero.copy(), "wavefield_m1": zero.copy(),
            "wavefield_sc_0": zero.copy(), "wavefield_sc_m1": zero.copy(),
            "dx": 10.0, "dt": 0.001, "accuracy": 4, "boundary": "pml", "pml_width": 10,
            "pml_freq": 15.0, "max_vel": 1800.0}


def reference(case):
    # This is the float64 Deepwave function used to generate the locked goldens.
    result = runpy.run_path(str(ROOT / "judge" / "make_goldens.py"))["run"](case)
    return {"background": result.bg_receiver_amplitudes.detach().numpy(),
            "scattered": result.sc_receiver_amplitudes.detach().numpy()}


def metal_worker(size_name):
    case = make_case(SIZES[size_name])
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    call("metal", "born", **case)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(json.dumps({"wall_s": elapsed, "peak_mem_mb": max(0, after - before) / MIB}))


def run_metal(size_name):
    # Each helper has no prior children, so the high-water delta belongs to Metal.
    output = subprocess.check_output([sys.executable, __file__, "--metal-worker", size_name], text=True)
    return json.loads(output)


def main():
    import mlx.core as mx
    import torch

    torch.set_num_threads(1)
    results = {name: {} for name in ("judge", "mlx", "metal")}
    medium_outputs = {}
    for size_name, size in SIZES.items():
        case = make_case(size)
        for path in ("judge", "mlx", "metal"):
            print(f"{path} / {size_name}", flush=True)
            if path == "metal":
                run_metal(size_name)
                measurements = [run_metal(size_name) for _ in range(3)]
                entry = {"wall_s": statistics.median(m["wall_s"] for m in measurements),
                         "peak_mem_mb": max(m["peak_mem_mb"] for m in measurements),
                         "mem_method": "Metal child ru_maxrss delta in a fresh helper process (macOS bytes)"}
                if size_name == "medium":
                    medium_outputs[path] = call(path, "born", **case)
            else:
                fn = (lambda case=case: reference(case)) if path == "judge" else (
                    lambda path=path, case=case: call(path, "born", **case))
                fn()
                times, peaks, rss = [], [], []
                for _ in range(3):
                    if path == "mlx":
                        mx.reset_peak_memory()
                    start = time.perf_counter()
                    output = fn()
                    times.append(time.perf_counter() - start)
                    rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MIB)
                    if path == "mlx":
                        peaks.append(mx.get_peak_memory() / MIB)
                if size_name == "medium":
                    medium_outputs[path] = output
                entry = {"wall_s": statistics.median(times),
                         "peak_mem_mb": max(peaks) if peaks else max(rss),
                         "mem_method": ("MLX reset_peak_memory/get_peak_memory; process peak RSS separately"
                                        if path == "mlx" else "process peak RSS (ru_maxrss, macOS bytes; includes native Torch allocations)")}
                if path == "mlx":
                    entry["process_peak_rss_mb"] = max(rss)
            results[path][size_name] = entry
    diffs = {}
    for path in ("mlx", "metal"):
        errors = []
        for field in ("background", "scattered"):
            ref = medium_outputs["judge"][field].astype(np.float64)
            actual = medium_outputs[path][field].astype(np.float64)
            denom = float(np.max(np.abs(ref)))
            errors.append(float(np.max(np.abs(actual - ref))) / denom if denom else
                          float(np.max(np.abs(actual - ref))))
        diffs[path] = max(errors)
    report = {"project": "14-born-2d", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": diffs,
              "notes": "Forward born, accuracy 4, 10-cell PML, dx 10 m, dt 1 ms; median of three after one warm-up. Relative error is the maximum over both outputs of max absolute error divided by each reference field's peak absolute amplitude. Metal includes harness serialization, process startup and shader compilation; MLX and judge run in process. RSS is a process lifetime high-water mark, not GPU allocation; Metal's child RSS may omit device memory. Timings and memory are uncalibrated until run on the target machine."}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-worker":
        metal_worker(sys.argv[2])
    else:
        main()
