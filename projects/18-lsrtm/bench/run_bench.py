"""Benchmark one LSRTM update on deterministic synthetic observations."""
from __future__ import annotations

import importlib.util
import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call

SIZES = {
    "small": {"nz": 20, "nx": 24, "nt": 128, "nshot": 1, "nrec": 8, "iterations": 1},
    "medium": {"nz": 40, "nx": 48, "nt": 320, "nshot": 2, "nrec": 20, "iterations": 1},
    "large": {"nz": 64, "nx": 80, "nt": 640, "nshot": 3, "nrec": 36, "iterations": 1},
}


def survey(nz, nx, nt, nshot, nrec, iterations):
    rng = np.random.default_rng(18017)
    v = np.full((nz, nx), 1800.0, dtype=np.float32)
    mask = np.zeros_like(v)
    mask[4:-3, 3:-3] = 1
    sx = np.linspace(3, nx - 4, nshot).round().astype(np.int64)
    rx = np.linspace(2, nx - 3, nrec).round().astype(np.int64)
    sloc = np.stack((np.full(nshot, 2), sx), axis=-1)[:, None, :]
    rloc = np.broadcast_to(np.stack((np.full(nrec, 2), rx), axis=-1),
                           (nshot, nrec, 2)).copy()
    dt, freq = 0.001, 18.0
    t = np.arange(nt, dtype=np.float64) * dt
    a = np.pi * freq * (t - 1.5 / freq)
    wavelet = ((1 - 2 * a * a) * np.exp(-a * a)).astype(np.float32)
    sources = np.broadcast_to(wavelet, (nshot, 1, nt)).copy()
    delays = rng.uniform(0.045, 0.075, size=(nshot, nrec, 1))
    phase = np.pi * freq * (t[None, None, :] - delays)
    observed = ((1 - 2 * phase**2) * np.exp(-phase**2)).astype(np.float32)
    return {"v": v, "mask": mask, "observed": observed, "source_amplitudes": sources,
            "source_locations": sloc, "receiver_locations": rloc, "spacing": [10.0, 10.0],
            "dt": dt, "pml_width": 10, "pml_freq": freq, "accuracy": 4,
            "iterations": iterations}


def load_reference():
    sys.path.insert(0, str(ROOT / "judge"))
    spec = importlib.util.spec_from_file_location("lsrtm_judge", ROOT / "judge" / "make_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def judge_solve(ref, inputs):
    ref_inputs = dict(inputs, spacing=np.asarray(inputs["spacing"]))
    op = ref.Born(ref_inputs)
    data = ref.tensor(inputs["observed"])
    rtm = op.adjoint(data)
    image, history, _ = ref.cgls(op, data, inputs["iterations"])
    predicted = op.forward(image).detach()
    return {"image": image.detach().numpy(), "rtm": rtm.detach().numpy(),
            "predicted": predicted.numpy(), "residual_history": np.asarray(history)}


def rss_mb(who):
    # Darwin reports bytes; Linux reports KiB.
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(who).ru_maxrss / scale


def max_rel(out, reference):
    differences = []
    for name, expected in reference.items():
        a = np.asarray(out[name], dtype=np.float64)
        b = np.asarray(expected, dtype=np.float64)
        denom = float(np.max(np.abs(b)))
        error = float(np.max(np.abs(a - b)))
        differences.append(error / denom if denom else error)
    return max(differences)


def main():
    import mlx.core as mx

    ref = load_reference()
    results = {path: {} for path in ("judge", "mlx", "metal")}
    medium = {}
    for size, params in SIZES.items():
        inputs = survey(**params)
        for path, path_results in results.items():
            run = (partial(judge_solve, ref, inputs) if path == "judge" else
                   partial(call, path, "lsrtm", **inputs))
            print(f"{path} {size}: warmup", flush=True)
            run()
            times, memories, process_rss = [], [], []
            for _ in range(3):
                if path == "judge":
                    tracemalloc.start()
                    tracemalloc.reset_peak()
                elif path == "mlx":
                    mx.reset_peak_memory()
                else:
                    child_before = rss_mb(resource.RUSAGE_CHILDREN)
                start = time.perf_counter()
                output = run()
                times.append(time.perf_counter() - start)
                if path == "judge":
                    memories.append(tracemalloc.get_traced_memory()[1] / 1024**2)
                    tracemalloc.stop()
                elif path == "mlx":
                    memories.append(mx.get_peak_memory() / 1024**2)
                    process_rss.append(rss_mb(resource.RUSAGE_SELF))
                else:
                    memories.append(max(0.0, rss_mb(resource.RUSAGE_CHILDREN) - child_before))
            methods = {
                "judge": "tracemalloc peak Python allocations; excludes native Torch allocations",
                "mlx": "MLX reset_peak_memory/get_peak_memory in MiB",
                "metal": "RUSAGE_CHILDREN ru_maxrss high-water mark delta; bytes on macOS",
            }
            entry = {"wall_s": statistics.median(times), "peak_mem_mb": max(memories),
                     "mem_method": methods[path]}
            if process_rss:
                entry["process_peak_rss_mb"] = max(process_rss)
            path_results[size] = entry
            if size == "medium":
                medium[path] = output
            print(f"{path} {size}: {entry['wall_s']:.3f} s", flush=True)
    comparisons = {path: max_rel(medium[path], medium["judge"]) for path in ("mlx", "metal")}
    report = {"project": "18-lsrtm", "sizes": SIZES, "results": results,
              "medium_max_rel_diff": comparisons,
              "notes": ("One CGLS update; deterministic synthetic observations. Times include harness "
                        "and Metal process/serialization overhead. MLX retains its warmed operator "
                        "and background cache. Memory is MiB. Child ru_maxrss is a cumulative "
                        "high-water mark, so its delta can be zero even when the child uses memory. "
                        "Relative difference is the maximum across outputs of max absolute error "
                        "divided by each reference output's maximum magnitude.")}
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
