"""Benchmark the frozen FWI API on a synthetic case and licensed Marmousi2.

Run from the project folder. No GPU work occurs on import. Surveys are
evaluated one shot at a time, identically for all paths, to bound memory.
"""
from __future__ import annotations

import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE / "common" / "py"))
os.environ["OUTCROP_PROJECT"] = str(ROOT)
from outcrop_harness import call, metal_binary

SEED = 1601
SETTINGS_KEYS = ("grid_spacing", "dt", "accuracy", "pml_width", "pml_freq", "max_vel")
SIZES = {
    "small": {
        "model": "synthetic tilted layers, fault and lens (stand-in)",
        "nz": 48, "nx": 64, "nt": 800, "shots": 4, "receivers": 64,
        "grid_spacing": 10.0, "dt": 0.001, "accuracy": 4,
        "pml_width": 10, "pml_freq": 15.0, "max_vel": 2600.0,
        "record_length_s": 0.799, "shot_batch_size": 1,
        "source_depth_index": 2, "receiver_depth_index": 2,
    },
    "medium": {
        "model": "data/marmousi2/vp_70x340_d50m.npy",
        "nz": 70, "nx": 340, "nt": 1001, "shots": 8, "receivers": 340,
        "grid_spacing": 50.0, "dt": 0.004, "accuracy": 4,
        "pml_width": 12, "pml_freq": 4.5, "max_vel": 4700.0,
        "record_length_s": 4.0, "shot_batch_size": 1,
        "source_depth_index": 1, "receiver_depth_index": 1,
        "starting_model_gaussian_sigma_m": 150.0,
    },
    "large": {
        "model": "data/marmousi2/vp_140x680_d25m.npy",
        "nz": 140, "nx": 680, "nt": 2001, "shots": 8, "receivers": 680,
        "grid_spacing": 25.0, "dt": 0.002, "accuracy": 4,
        "pml_width": 24, "pml_freq": 8.0, "max_vel": 4700.0,
        "record_length_s": 4.0, "shot_batch_size": 1,
        "source_depth_index": 1, "receiver_depth_index": 1,
        "starting_model_gaussian_sigma_m": 150.0,
    },
}
ATTRIBUTION = (
    "Velocity model: Marmousi2 (AGL Elastic Marmousi), (c) 2004 Allied "
    "Geophysical Laboratory, University of Houston; Martin, Wiley & Marfurt "
    "(2006); CC BY 4.0. Block-averaged."
)


def smooth(model, sigma_cells):
    """Separable Gaussian with edge extension; no extra dependencies."""
    radius = max(1, int(np.ceil(3 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel /= kernel.sum()
    result = np.asarray(model, dtype=np.float64)
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        result = np.apply_along_axis(
            lambda line: np.convolve(line, kernel, mode="valid"),
            axis, np.pad(result, padding, mode="edge"),
        )
    return np.ascontiguousarray(result, dtype=np.float32)


def build_inputs(size_name, *, shots=None):
    """Return API inputs (zero targets initially) and the independent truth."""
    size = SIZES[size_name]
    nz, nx, nt = size["nz"], size["nx"], size["nt"]
    shots = size["shots"] if shots is None else shots
    spacing = size["grid_spacing"]
    if size_name == "small":
        # The judge's stand-in construction expanded to 48 x 64. Rescaling
        # retains its tilted layers, fault and smooth lens on the larger grid.
        z, x = np.mgrid[:nz, :nx].astype(np.float64)
        z *= 23 / (nz - 1)
        x *= 31 / (nx - 1)
        background = 1800 + 7 * z + 1.5 * x
        truth = background + 70 * np.tanh((z - 11 - 0.12*x - 2*(x > 18))/2)
        truth += 75 * np.exp(-((z-16)**2/24 + (x-12)**2/50))
        truth = truth.astype(np.float32)
        initial = (background - 75).astype(np.float32)
    else:
        filename = WORKSPACE / size["model"]
        truth = np.ascontiguousarray(np.load(filename, allow_pickle=False), dtype=np.float32)
        if truth.shape != (nz, nx) or not np.isfinite(truth).all():
            raise ValueError(f"Invalid Marmousi2 array: {filename}")
        # Preserve the gas-sand velocities below 1500 m/s in the true model.
        initial = smooth(truth, size["starting_model_gaussian_sigma_m"] / spacing)
        initial[:int(450 / spacing)] = 1500.0  # Known, fixed water layer.
    if truth.min() <= 0 or truth.max() > size["max_vel"]:
        raise ValueError("Model exceeds the fixed propagation speed bound")
    # Fourth-order 2-D leapfrog CFL limit is sqrt(3/8) ~ 0.612. Both
    # Marmousi2 cases use max_vel*dt/dx = 0.376, without substepping.
    if size["max_vel"] * size["dt"] / spacing >= 0.6:
        raise ValueError("Time step exceeds the conservative CFL limit")
    frequency = size["pml_freq"]
    t = np.arange(nt, dtype=np.float64) * size["dt"]
    a = np.pi * frequency * (t - 1.5 / frequency)
    wavelet = ((1 - 2*a*a) * np.exp(-a*a) / spacing**2).astype(np.float32)
    amplitudes = np.broadcast_to(wavelet, (shots, 1, nt)).copy()
    amplitudes *= np.random.default_rng(SEED).uniform(0.95, 1.05, (shots, 1, 1))
    source_x = np.linspace(2, nx - 3, shots).round().astype(np.int64)
    sl = np.stack((np.full(shots, size["source_depth_index"]), source_x), axis=-1)[:, None]
    positions = np.stack((np.full(nx, size["receiver_depth_index"]), np.arange(nx)), axis=-1)
    rl = np.broadcast_to(positions, (shots, nx, 2)).copy().astype(np.int64)
    args = {
        "velocity": initial, "source_amplitudes": amplitudes,
        "source_locations": np.ascontiguousarray(sl, dtype=np.int64),
        "receiver_locations": rl, "observed": np.zeros((shots, nx, nt), dtype=np.float32),
        **{key: size[key] for key in SETTINGS_KEYS},
    }
    return args, truth


def judge_call(args):
    # Reuse the actual float64 judge evaluator, with per-case settings. This
    # changes its in-memory configuration only; no judge file is modified.
    sys.path.insert(0, str(ROOT / "judge"))
    import make_goldens
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    make_goldens.SETTINGS = {key: args[key] for key in SETTINGS_KEYS}
    return make_goldens.evaluate(
        args["velocity"], args["source_amplitudes"], args["source_locations"],
        args["receiver_locations"], args["observed"], dtype=torch.float64,
    )


def invoke(path, args):
    """Assemble a full survey from one-shot frozen-API evaluations.

    Equal receiver/sample counts make the mean of shot objectives/gradients
    exactly the full-survey mean loss. Accumulate in float64 for roundoff.
    """
    shots = len(args["source_amplitudes"])
    dtype = np.float64 if path == "judge" else np.float32
    predicted = np.empty(args["observed"].shape, dtype=dtype)
    gradient = np.zeros(args["velocity"].shape, dtype=np.float64)
    objective = 0.0
    shot_arrays = ("source_amplitudes", "source_locations", "receiver_locations", "observed")
    for index in range(shots):
        shot = dict(args)
        shot.update({key: np.ascontiguousarray(args[key][index:index+1]) for key in shot_arrays})
        out = judge_call(shot) if path == "judge" else call(path, "misfit_and_grad", **shot)
        if not all(np.isfinite(out[key]).all() for key in ("predicted", "objective", "gradient")):
            raise RuntimeError(f"{path}: nonfinite FWI output")
        predicted[index:index+1] = out["predicted"]
        gradient += out["gradient"] / shots
        objective += float(out["objective"]) / shots
    return {"predicted": predicted, "objective": np.asarray(objective, dtype=dtype),
            "gradient": gradient.astype(dtype)}


def add_observations(args, truth):
    """Generate noiseless truth data once, outside all benchmark timers."""
    generated = invoke("metal", {**args, "velocity": truth})
    args["observed"] = np.ascontiguousarray(generated["predicted"], dtype=np.float32)
    return args


def rss_mb(who):
    # macOS returns bytes; Linux returns KiB. Report decimal MB throughout.
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return resource.getrusage(who).ru_maxrss * scale / 1e6


def metal_sample(input_file):
    # A fresh worker has never spawned a child, so its before value is zero.
    # Across sequential one-shot children, ru_maxrss is their maximum, not
    # their sum. A global-parent delta after warm-up would often return zero.
    with np.load(input_file, allow_pickle=False) as archive:
        args = {key: archive[key] for key in archive.files}
    for key in SETTINGS_KEYS:
        args[key] = args[key].item()
    before = rss_mb(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    invoke("metal", args)
    elapsed = time.perf_counter() - start
    after = rss_mb(resource.RUSAGE_CHILDREN)
    return {"wall_s": elapsed, "peak_mem_mb": max(0.0, after - before),
            "child_max_rss_mb": after}


def measure(path, args, input_file):
    invoke(path, args)  # One full-survey warm-up, excluded from timing.
    samples = []
    for _ in range(3):
        if path == "metal":
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--metal-sample", str(input_file)],
                check=True, capture_output=True, text=True,
            )
            samples.append(json.loads(proc.stdout))
            continue
        if path == "mlx":
            import mlx.core as mx
            mx.reset_peak_memory()
        else:
            tracemalloc.start()
            tracemalloc.reset_peak()
        start = time.perf_counter()
        invoke(path, args)
        elapsed = time.perf_counter() - start
        if path == "mlx":
            samples.append({"wall_s": elapsed, "peak_mem_mb": mx.get_peak_memory()/1e6,
                            "process_max_rss_mb": rss_mb(resource.RUSAGE_SELF)})
        else:
            peak = tracemalloc.get_traced_memory()[1]/1e6
            tracemalloc.stop()
            samples.append({"wall_s": elapsed, "peak_mem_mb": peak,
                            "process_max_rss_mb": rss_mb(resource.RUSAGE_SELF)})
    methods = {
        "judge": "tracemalloc peak; native PyTorch memory excluded; process lifetime max RSS also reported",
        "mlx": "mx.reset_peak_memory/get_peak_memory; process lifetime max RSS also reported",
        "metal": "fresh worker RUSAGE_CHILDREN ru_maxrss delta; maximum of one-shot child peaks; macOS bytes",
    }
    result = {"wall_s": statistics.median(s["wall_s"] for s in samples),
              "peak_mem_mb": max(s["peak_mem_mb"] for s in samples),
              "mem_method": methods[path], "repeat_wall_s": [s["wall_s"] for s in samples]}
    for extra in ("process_max_rss_mb", "child_max_rss_mb"):
        if extra in samples[0]:
            result[extra] = max(s[extra] for s in samples)
    return result


def work(size):
    pad = 2 * (size["pml_width"] + 2)
    return (size["nz"] + pad) * (size["nx"] + pad) * size["nt"] * size["shots"]


def max_rel_difference(actual, reference):
    errors = []
    for key in ("predicted", "objective", "gradient"):
        ref = np.asarray(reference[key], dtype=np.float64)
        value = np.asarray(actual[key], dtype=np.float64)
        errors.append(float(np.max(np.abs(value - ref))) / max(float(np.max(np.abs(ref))), 1e-30))
    return max(errors)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--metal-sample":
        print(json.dumps(metal_sample(sys.argv[2]), allow_nan=False))
        return
    if not metal_binary(ROOT).is_file():
        raise FileNotFoundError(f"Build the release binary first: {metal_binary(ROOT)}")
    results = {path: {} for path in ("judge", "mlx", "metal")}
    differences = {}
    for size_name, size in SIZES.items():
        print(f"Preparing {size_name}: {size['model']}", flush=True)
        args, truth = build_inputs(size_name)
        add_observations(args, truth)
        medium_outputs = {}
        with tempfile.TemporaryDirectory(prefix="outcrop16-bench-") as directory:
            input_file = Path(directory) / "inputs.npz"
            np.savez(input_file, **args)
            for path, path_results in results.items():
                if path == "judge" and size_name == "large":
                    estimate = results["judge"]["medium"]["wall_s"] * work(size)/work(SIZES["medium"])
                    if estimate > 120:
                        path_results[size_name] = {
                            "skipped": f"One-thread Deepwave f64 estimated at {estimate:.1f} s per survey (>120 s), scaled from medium padded cells x shots x steps"
                        }
                        continue
                print(f"Benchmarking {path} {size_name} (one warm-up + three repeats)", flush=True)
                path_results[size_name] = measure(path, args, input_file)
                if size_name == "medium":
                    medium_outputs[path] = invoke(path, args)
        if size_name == "medium":
            for path in ("mlx", "metal"):
                differences[path] = max_rel_difference(medium_outputs[path], medium_outputs["judge"])
        del medium_outputs
    report = {
        "project": "16-fwi-marmousi", "sizes": SIZES, "results": results,
        "medium_max_rel_diff": differences,
        "notes": (
            "Seed 1601. Small is an expanded synthetic stand-in; medium/large are Marmousi2. "
            "Marmousi2 surveys have eight surface shots, receivers at every horizontal cell, "
            "4 s records, peak frequencies 4.5/8 Hz, and max_vel*dt/dx=0.376. "
            "All paths process shots sequentially, then average per-shot objectives/gradients; "
            "wall_s covers the full survey. Large padded cell-shot-step work is about 111x small. "
            "Noiseless targets are generated on the true velocity with Metal before timing. "
            "Median of three repeats after one warm-up; includes harness I/O, process launch, "
            "Metal shader compilation, and shot assembly. Fresh Metal measurement worker setup "
            "and input loading are outside its timer. Memory is the maximum over repeats, in decimal MB. "
            "Metal child RSS may omit GPU allocations; MLX process RSS is a lifetime high-water mark; "
            "tracemalloc excludes native PyTorch allocations. One large Metal shot history is about "
            "1.12 GB; MLX autodiff retains more intermediates. Large target 5-60 s/path is unverified. "
            "The medium sanity metric is the maximum over predicted/objective/gradient of "
            "max(abs(actual-reference))/max(abs(reference)); reference is Deepwave CPU float64. "
            + ATTRIBUTION + " Licence: https://creativecommons.org/licenses/by/4.0/"
        ),
    }
    (ROOT / "bench" / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
