"""Reproducible CPU/float64 Deepwave oracle; run from the project folder.

The spec's independent physical oracle is the known depth of synthetic
reflecting sheets. Deepwave supplies the matched discrete forward/adjoint
oracle, building on projects 13 and 14. No implementation code is imported.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from itertools import pairwise
from pathlib import Path

import deepwave
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
DEST = PROJECT / "judge" / "golden"
SEED = 170101
# Read the one tolerance source by filename; no sys.path modifications.
_spec = importlib.util.spec_from_file_location(
    "rtm_tolerances", PROJECT / "tests" / "tolerances.py"
)
TOL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TOL)
PINS = {"numpy": "2.5.4", "torch": "2.15.0", "deepwave": "0.1.0"}


def forward(g, scatter, dtype=torch.float64):
    """J_v scatter; all non-location inputs rounded to API float32 first."""
    def tensor(a):
        return torch.as_tensor(np.asarray(a, np.float32), dtype=dtype)

    result = deepwave.scalar_born(
        tensor(g["velocity"]),
        g["grid_spacing"].tolist(),
        float(g["dt"]),
        dv=scatter,
        source_amplitudes=tensor(g["source_amplitudes"]),
        source_locations=torch.as_tensor(g["source_locations"], dtype=torch.int64),
        sc_receiver_locations=torch.as_tensor(g["receiver_locations"], dtype=torch.int64),
        accuracy=4,
        pml_width=int(g["pml_width"]),
        pml_freq=float(g["pml_freq"]),
        max_vel=float(np.max(g["velocity"])),
        model_gradient_sampling_interval=1,
        storage_mode="device",
        storage_compression=False,
        location_interpolation="none",
    )
    assert result.info.step_ratio == 1, "API assumes no internal time resampling"
    return result.sc_receiver_amplitudes


def evaluate(g, scatter, data, dtype=torch.float64):
    s = torch.tensor(np.asarray(scatter, np.float32), dtype=dtype, requires_grad=True)
    pred = forward(g, s, dtype)
    residual = pred - torch.tensor(np.asarray(data, np.float32), dtype=dtype)
    loss = 0.5 * residual.square().sum()
    grad, = torch.autograd.grad(loss, s)
    return {"predicted": pred.detach().numpy(), "loss": loss.detach().numpy(),
            "gradient": grad.detach().numpy()}


def scalar_forward(g, velocity, max_vel):
    """Full acoustic solve for the independent piecewise-layered example."""
    result = deepwave.scalar(
        torch.tensor(velocity, dtype=torch.float64),
        g["grid_spacing"].tolist(), float(g["dt"]),
        source_amplitudes=torch.tensor(g["source_amplitudes"], dtype=torch.float64),
        source_locations=torch.tensor(g["source_locations"], dtype=torch.int64),
        receiver_locations=torch.tensor(g["receiver_locations"], dtype=torch.int64),
        accuracy=4, pml_width=int(g["pml_width"]), pml_freq=float(g["pml_freq"]),
        max_vel=max_vel, location_interpolation="none",
    )
    assert result.info.step_ratio == 1, "API assumes no internal time resampling"
    return result.receiver_amplitudes.detach().numpy()


def migrate(g, data, dtype=torch.float64):
    return -evaluate(g, np.zeros_like(g["velocity"]), data, dtype)["gradient"]


def relative(a, b):
    scale = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b))) / scale if scale else float(np.max(np.abs(a)))


def ricker(nt, dt, freq, peak):
    a = (np.pi * freq * (np.arange(nt, dtype=np.float64) * dt - peak)) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def geometry(nz, nx, dz, dx, nt, freq, shots, nsrc=1):
    g = {
        "velocity": np.full((nz, nx), 2000, dtype=np.float32),
        "source_amplitudes": np.tile(ricker(nt, 0.001, freq, 1.5 / freq),
                                     (len(shots), nsrc, 1)),
        "source_locations": np.array([[[3, x + k * 3] for k in range(nsrc)]
                                       for x in shots], dtype=np.int64),
        "receiver_locations": np.array([[[3, x] for x in range(5, nx - 4, 2)]
                                         for _ in shots], dtype=np.int64),
        "grid_spacing": np.array([dz, dx], dtype=np.float64),
        "dt": np.array(0.001), "pml_width": np.array(12, dtype=np.int64),
        "pml_freq": np.array(freq, dtype=np.float64),
    }
    return g


def describe(g):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                "min": float(v.min()), "max": float(v.max())}
            for k, v in g.items()}


def layer_case(name, *, heterogeneous):
    if heterogeneous:
        g = geometry(69, 61, 10.0, 12.0, 680, 25.0, [13, 29, 45])
        z, x = np.indices(g["velocity"].shape)
        # A smooth migration background avoids additional unknown interfaces.
        g["velocity"] = (1850 + 4 * z + 35 * np.cos(x / 12)).astype(np.float32)
        depths, strengths = [19, 35, 50], [30.0, -25.0, 35.0]
    else:
        g = geometry(65, 65, 8.0, 10.0, 600, 30.0, [16, 32, 48])
        depths, strengths = [22, 42], [60.0, -70.0]
    truth = np.zeros_like(g["velocity"])
    for depth, strength in zip(depths, strengths, strict=True):
        if heterogeneous:
            truth[depth, :] = strength
        else:
            truth[depth:, :] += strength
    # Observed data is rounded ONCE to exactly what float32 APIs receive.
    if heterogeneous:
        data = forward(g, torch.tensor(truth, dtype=torch.float64)).detach().numpy()
    else:
        # Full nonlinear layered data minus the homogeneous direct wave.
        # This case is independent of the Born forward used by migration.
        true_velocity = g["velocity"] + truth
        bound = float(np.max(true_velocity))
        # Identical PML coefficients in the two data-generation solves.
        data = scalar_forward(g, true_velocity, bound) - scalar_forward(g, g["velocity"], bound)
    g["data"] = data.astype(np.float32)
    image = migrate(g, g["data"])
    f32_image = migrate(g, g["data"], torch.float32)
    parity = relative(f32_image, image)
    assert parity <= TOL.REFERENCE_RTOL, (name, "float32 parity", parity)
    # Broad, disjoint depth zones: no searching only inside the allowed +/-2.
    bounds = [8] + [(a + b) // 2 for a, b in pairwise(depths)]
    bounds.append(g["velocity"].shape[0] - 5)
    zones = np.array(list(pairwise(bounds)), dtype=np.int64)
    nx = truth.shape[1]
    lateral = np.array([nx // 3, 2 * nx // 3 + 1], dtype=np.int64)
    profile = np.mean(image[:, lateral[0]:lateral[1]] ** 2, axis=1)
    peaks = [int(lo + np.argmax(profile[lo:hi])) for lo, hi in zones]
    f32_profile = np.mean(f32_image.astype(np.float64)[:, lateral[0]:lateral[1]] ** 2, axis=1)
    f32_peaks = [int(lo + np.argmax(f32_profile[lo:hi])) for lo, hi in zones]
    assert np.all(np.abs(np.array(peaks) - depths) <= TOL.DEPTH_CELLS), (name, peaks)
    assert np.all(np.abs(np.array(f32_peaks) - depths) <= TOL.DEPTH_CELLS), (name, f32_peaks)
    assert all(profile[p] > TOL.MIN_LAYER_FRACTION ** 2 * profile.max() for p in peaks)
    g.update(truth=truth, image=image, reflector_rows=np.array(depths, np.int64),
             depth_zones=zones, lateral_roi=lateral, reference_peaks=np.array(peaks))
    print(f"{name}: depths={depths}, peaks={peaks}, data=[{data.min():.6g}, "
          f"{data.max():.6g}], image=[{image.min():.6g}, {image.max():.6g}], "
          f"float32 rel={parity:.3g}", flush=True)
    return g, {"float32_image_relative_error": parity, "peak_rows": peaks,
               "float32_peak_rows": f32_peaks,
               "true_depths_m": (np.array(depths) * float(g["grid_spacing"][0])).tolist(),
               "depth_zones_half_open": zones.tolist(), "lateral_roi": lateral.tolist(),
               "data_generation": "Born thin sheets" if heterogeneous else "full scalar layered minus full scalar background",
               "layer_velocity_jumps_or_sheet_dv_m_s": strengths}


def adjoint_case():
    rng = np.random.default_rng(SEED)
    g = geometry(27, 31, 10.0, 12.0, 320, 25.0, [7, 20], nsrc=2)
    z, x = np.indices(g["velocity"].shape)
    g["velocity"] = (1900 + z * 3 + 20 * np.sin(x / 6)).astype(np.float32)
    g["source_amplitudes"][0, 1] *= -0.5
    g["source_amplitudes"][1, 0] *= 0.75
    g["source_amplitudes"][1, 1] = np.roll(g["source_amplitudes"][1, 1], 7)
    # Nonhorizontal receiver geometry catches z/x mistakes and shot mixing.
    g["receiver_locations"][1, :, 0] = 4 + np.arange(g["receiver_locations"].shape[1]) % 3
    truth = np.zeros_like(g["velocity"])
    truth[12, :] = 20
    truth[20, :] = -15
    data = forward(g, torch.tensor(truth, dtype=torch.float64)).detach().numpy()
    # Fixed noise keeps the adjoint test from seeing only range(J) data.
    g["data"] = (data + 0.02 * abs(data).max() * rng.standard_normal(data.shape)).astype(np.float32)
    scatter = rng.integers(-4, 5, size=truth.shape).astype(np.float32)
    ref = evaluate(g, scatter, g["data"])
    img = migrate(g, g["data"])
    parity = {k: relative(v, ref[k]) for k, v in
              evaluate(g, scatter, g["data"], torch.float32).items()}
    parity["image"] = relative(migrate(g, g["data"], torch.float32), img)
    assert max(parity.values()) <= TOL.REFERENCE_RTOL, parity
    # Dense seeded dyadic direction, sparse largest-gradient direction, and
    # a gradient-aligned dyadic direction give complementary derivative checks.
    dirs = np.zeros((3, *truth.shape), dtype=np.float32)
    dirs[0] = rng.choice([-1.0, 1.0], size=truth.shape)
    dirs[1].flat[np.argmax(np.abs(ref["gradient"]))] = 1
    dirs[2] = np.round(8 * ref["gradient"] / np.abs(ref["gradient"]).max()) / 8
    fd, fd32 = [], []
    losses = []
    for direction in dirs:
        endpoints = []
        endpoints32 = []
        for sign in [-1, 1]:
            pert = scatter + sign * TOL.FD_STEP * direction
            pred = forward(g, torch.tensor(pert, dtype=torch.float64)).detach().numpy()
            endpoints.append(float(0.5 * np.sum((pred - g["data"]) ** 2)))
            pred32 = forward(g, torch.tensor(pert, dtype=torch.float32), torch.float32)
            endpoints32.append(float(0.5 * (pred32 - torch.tensor(g["data"])).square().sum()))
        losses.append(endpoints)
        fd.append((endpoints[1] - endpoints[0]) / (2 * TOL.FD_STEP))
        fd32.append((endpoints32[1] - endpoints32[0]) / (2 * TOL.FD_STEP))
    analytical = np.sum(dirs.astype(np.float64) * ref["gradient"], axis=(1, 2))
    fd_errors = np.abs((np.array(fd) - analytical) / analytical)
    fd32_errors = np.abs((np.array(fd32) - analytical) / analytical)
    assert max(fd_errors) <= TOL.ORACLE_RTOL, fd_errors
    assert max(fd32_errors) <= TOL.GRADIENT_RTOL, fd32_errors
    # Dot product is independent of the least-squares loss and its gradient.
    jdirection = forward(g, torch.tensor(dirs[0], dtype=torch.float64)).detach().numpy()
    left = float(np.sum(img * dirs[0]))
    right = float(np.sum(g["data"].astype(np.float64) * jdirection))
    dot_error = abs(left - right) / max(abs(left), abs(right))
    assert dot_error <= TOL.ORACLE_RTOL, dot_error
    g.update(scatter=scatter, image=img, **ref, directions=dirs,
             fd_derivatives=np.array(fd), fd_losses=np.array(losses),
             jdirection=jdirection, dot_product=np.array(right))
    print(f"adjoint: loss={ref['loss']:.6g}, dot rel={dot_error:.3g}, "
          f"float32 parity={parity}, FD rel={fd_errors}, float32 FD rel={fd32_errors}", flush=True)
    return g, {"float32_relative_errors": parity, "float64_fd_relative_errors": fd_errors.tolist(),
               "float32_fd_relative_errors": fd32_errors.tolist(), "dot_relative_error": dot_error}


def main():
    versions = {k: importlib.metadata.version(k) for k in PINS}
    if versions != PINS:
        raise RuntimeError(f"Oracle dependencies changed: expected {PINS}, got {versions}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    DEST.mkdir(parents=True, exist_ok=True)
    meta = {
        "project": "17-rtm", "seed": SEED, "versions": versions,
        "python": platform.python_version(), "device": "cpu", "reference_dtype": "float64",
        "reference": "Known-depth synthetic layers; Deepwave scalar full data, scalar_born forward and autograd adjoint",
        "discretization": {"space_order": 4, "time_order": 2, "pml": "Deepwave CFS-PML",
                           "max_vel": "max(velocity)", "substeps": 1,
                           "scatter_padding": "edge replicate, adjoint folds padding",
                           "normalization": "Euclidean transpose, summed shots, no dt or area weights"},
        "tolerances": {k: getattr(TOL, k) for k in vars(TOL) if k.isupper()},
        "tolerance_reasons": (PROJECT / "tests" / "tolerances.py").read_text(),
        "inputs": "All source, velocity, scatter, data arrays quantized once to float32 before the float64 solve; int64 locations; float64 Python scalars",
        "synthetic": "layers_uniform: full scalar piecewise-constant layers (2000/2060/1990 m/s), direct-wave subtracted, migrated in homogeneous 2000 m/s; layers_variable: Born thin signed sheets in smooth heterogeneous background",
        "fd": "3 saved dyadic directions, centered difference, step 0.5 m/s; quadratic loss has zero truncation error",
        "interface_depth_convention": "First grid node of the deeper piecewise-constant layer; sheet depths are their grid rows",
        "full_scalar_data_pml_max_vel_m_s": 2060.0,
        "cases": {},
    }
    for name, make in [
        ("layers_uniform", lambda: layer_case("layers_uniform", heterogeneous=False)),
        ("layers_variable", lambda: layer_case("layers_variable", heterogeneous=True)),
        ("adjoint", adjoint_case),
    ]:
        g, sanity = make()
        assert all(np.all(np.isfinite(v)) for v in g.values())
        target = DEST / f"{name}.npz"
        np.savez_compressed(target, **g)
        meta["cases"][name] = {"arrays": describe(g), "sanity": sanity,
                               "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    # The installed Deepwave reports 0.1.0; hashes disambiguate local builds.
    root = Path(deepwave.__file__).parent
    meta["deepwave_file_hashes"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ["scalar.py", "scalar_kernels.h", "scalar_born.py",
                     "scalar_born_kernels.h", "libdeepwave_C.dylib"]
    }
    (DEST / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = sum(p.stat().st_size for p in DEST.glob("*.npz"))
    assert size < 20_000_000, size
    print(f"Saved {size:,} bytes of goldens to {DEST}", flush=True)


if __name__ == "__main__":
    main()
