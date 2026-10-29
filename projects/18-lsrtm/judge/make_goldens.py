"""Float64 Deepwave Born/adjoint oracle on project 17's layered Born survey.

No implementation code is imported. Run from the project directory with the
repository's pinned venv. The exact project-17 input archive is vendored in judge/input/
by the first run so regeneration also works from a project-only checkout.

The locked tests/SCHEME.md is independently transcribed in NumPy and checked
against both bundles, including complete CGLS solves, before they are written.
Widths use route (a) from review: full-column analytic-signal envelope
FWHM measures the reflected wave packet instead of its near-Nyquist raw carrier
lobe. The survey and 20% threshold are unchanged. Generation asserts that the
reference passes every width and residual check before writing its golden.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import runpy
import time
from pathlib import Path

import deepwave
import numpy as np
import torch
from scheme_reference import validate as validate_scheme

ROOT = Path(__file__).resolve().parents[1]
TOL = runpy.run_path(str(ROOT / "tests/tolerances.py"))
METRICS = runpy.run_path(str(ROOT / "tests/metrics.py"))
reflector_metrics = METRICS["reflector_metrics"]
SEED = 18017
ITERATIONS = 40
VERSIONS = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "deepwave", "pytest")}
EXPECTED_VERSIONS = {"numpy": "2.5.4", "torch": "2.15.0", "deepwave": "0.1.0", "pytest": "9.2.0"}
SCHEME_DIAGNOSTICS = {}


def tensor(a, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)


def survey(nz=40, nx=48, nt=640, velocity=1800.0, frequency=12.0):
    """All floating inputs are rounded once to float32, then judged in float64."""
    dt = 0.001
    v = np.full((nz, nx), velocity, np.float32)
    source_locations = np.array([[[3, nx // 5]], [[3, nx // 2]], [[3, 4 * nx // 5]]], np.int64)
    xs = np.arange(4, nx - 4, 2)
    receiver_locations = np.tile(np.stack([np.full_like(xs, 3), xs], -1)[None], (3, 1, 1))
    t = np.arange(nt, dtype=np.float64) * dt
    a = np.pi * frequency * (t - 1.5 / frequency)
    wavelet = ((1.0 - 2.0 * a**2) * np.exp(-a**2)).astype(np.float32)
    sources = np.tile(wavelet[None, None, :], (3, 1, 1))
    mask = np.zeros_like(v)
    mask[7:-4, 4:-4] = 1.0
    return {"v": v, "source_amplitudes": sources, "source_locations": source_locations,
            "receiver_locations": receiver_locations, "mask": mask,
            "spacing": np.array([10.0, 10.0]), "dt": np.array(dt),
            "pml_width": np.array(12), "pml_freq": np.array(frequency), "accuracy": np.array(4)}


class Born:
    def __init__(self, inputs, dtype=torch.float64):
        self.v = tensor(inputs["v"], dtype)
        self.mask = tensor(inputs["mask"], dtype)
        self.kw = {"grid_spacing": inputs["spacing"].tolist(), "dt": float(inputs["dt"]),
                   "source_amplitudes": tensor(inputs["source_amplitudes"], dtype),
                   "source_locations": tensor(inputs["source_locations"], torch.long),
                   "sc_receiver_locations": tensor(inputs["receiver_locations"], torch.long),
                   "accuracy": int(inputs["accuracy"]), "pml_width": int(inputs["pml_width"]),
                   "pml_freq": float(inputs["pml_freq"]), "max_vel": float(self.v.max()),
                   "survey_pad": None, "model_gradient_sampling_interval": 1,
                   "freq_taper_frac": 0.0, "time_pad_frac": 0.0, "time_taper": False,
                   "storage_mode": "device", "storage_compression": False,
                   "location_interpolation": "none"}

    def forward(self, model):
        result = deepwave.scalar_born(self.v, dv=self.mask * model, **self.kw)
        assert result.info.step_ratio == 1, "Contract requires one integration step per sample"
        return result.sc_receiver_amplitudes

    def adjoint(self, data):
        model = torch.zeros_like(self.v, requires_grad=True)
        result = self.forward(model)
        return torch.autograd.grad(result, model, data)[0].detach()

    def objective(self, model, observed):
        m = model.detach().clone().requires_grad_(True)
        predicted = self.forward(m)
        loss = 0.5 * torch.sum((predicted - observed)**2)
        gradient = torch.autograd.grad(loss, m)[0]
        return loss.detach(), gradient.detach(), predicted.detach()


def cgls(op, observed, iterations):
    """CGLS with two-pass full A-conjugate reorthogonalization, zero start.

    In exact arithmetic this produces the CGLS iterates. Retaining normalized
    (p, A p) pairs prevents float32 loss of conjugacy, without regularization.
    """
    model = torch.zeros_like(op.v)
    residual = observed.clone()
    norm0 = torch.linalg.vector_norm(observed)
    if float(norm0) == 0.0:
        return model, np.zeros(iterations + 1), model.clone()
    history = [1.0]
    first = None
    basis = []
    for _ in range(iterations):
        p = op.adjoint(residual)
        q = op.forward(p).detach()
        for _pass in range(2):
            for old_p, old_q in basis:
                beta = torch.sum(old_q * q)
                p = p - beta * old_p
                q = q - beta * old_q
        scale = torch.linalg.vector_norm(q)
        p, q = p / scale, q / scale
        alpha = torch.sum(residual * q)
        model = model + alpha * p
        basis.append((p, q))
        if first is None:
            first = model.clone()
        residual = observed - op.forward(model).detach()
        history.append(float(torch.linalg.vector_norm(residual) / norm0))
    return model, np.asarray(history), first


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_case(outdir, name, arrays):
    print(f"Checking {name} against the independent locked-scheme reference", flush=True)
    SCHEME_DIAGNOSTICS[name] = validate_scheme(arrays, TOL["ORACLE_RTOL"], TOL["FD_STEP"])
    print(f"{name} scheme check:", SCHEME_DIAGNOSTICS[name], flush=True)
    np.savez_compressed(outdir / f"{name}.npz", **arrays)
    return {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype),
                  "sha256": digest(value),
                  "min": float(np.min(value)), "max": float(np.max(value)),
                  **({"value": np.asarray(value).item()} if np.asarray(value).ndim == 0 else {})}
            for key, value in arrays.items()}


def layered(outdir):
    input_dir = ROOT / "judge/input"
    input_dir.mkdir(exist_ok=True)
    shared_path = input_dir / "rtm17_layers_variable.npz"
    if not shared_path.exists():
        upstream = ROOT.parent / "17-rtm/judge/golden/layers_variable.npz"
        if not upstream.exists():
            raise RuntimeError("BLOCKED: project 17 layered Born synthetic unavailable")
        shared_path.write_bytes(upstream.read_bytes())
    with np.load(shared_path, allow_pickle=False) as archive:
        shared = {k: archive[k] for k in archive.files}
    inputs = {"v": shared["velocity"], "source_amplitudes": shared["source_amplitudes"],
              "source_locations": shared["source_locations"], "receiver_locations": shared["receiver_locations"],
              "spacing": shared["grid_spacing"], "dt": shared["dt"],
              "pml_width": shared["pml_width"], "pml_freq": shared["pml_freq"],
              "accuracy": np.array(4), "mask": np.ones_like(shared["velocity"])}
    truth = shared["truth"]
    depths = shared["reflector_rows"]
    op = Born(inputs)
    observed = tensor(shared["data"])
    predicted_truth = op.forward(tensor(truth)).detach()
    assert float(torch.max(torch.abs(predicted_truth-observed)) / torch.max(torch.abs(observed))) <= TOL["ORACLE_RTOL"]
    rtm = op.adjoint(observed)
    assert np.max(np.abs(rtm.numpy()-shared["image"])) / np.max(np.abs(shared["image"])) <= TOL["ORACLE_RTOL"]
    model, history, rtm_scaled = cgls(op, observed, ITERATIONS)
    nx = truth.shape[1]
    columns = np.array([nx // 3, nx // 2, 2 * nx // 3], np.int64)
    windows = shared["depth_zones"]
    rtm_depth, rtm_width = reflector_metrics(rtm.numpy(), columns, windows)
    ls_depth, ls_width = reflector_metrics(model.numpy(), columns, windows)
    truth_depth, truth_width = reflector_metrics(truth, columns, windows)
    ratios = ls_width / rtm_width
    print("layered residual history:", history.tolist(), flush=True)
    print("RTM widths:", rtm_width.tolist(), "LS widths:", ls_width.tolist(), flush=True)
    print("width ratios:", ratios.tolist(), "peak depths:", ls_depth.tolist(), flush=True)
    assert history[-1] <= TOL["MAX_RESIDUAL_RATIO"]
    assert np.all(ratios <= TOL["MAX_WIDTH_RATIO"]), (
        "Reference fails the envelope-width criterion; refusing golden", ratios)
    assert np.all(np.abs(ls_depth - depths[:, None]) <= TOL["MAX_DEPTH_ERROR_CELLS"])
    assert np.all(np.abs(rtm_depth - depths[:, None]) <= TOL["MAX_DEPTH_ERROR_CELLS"])
    pred = op.forward(model).detach()
    rtm_pred = op.forward(rtm_scaled).detach()
    loss, grad, _ = op.objective(torch.zeros_like(op.v), observed)
    np.testing.assert_allclose(grad.numpy(), -rtm.numpy(), rtol=TOL["ORACLE_RTOL"], atol=0)
    assert history[-1] <= TOL["MAX_RESIDUAL_RATIO"] * float(torch.linalg.vector_norm(rtm_pred - observed) / torch.linalg.vector_norm(observed))
    scaled_observed = (observed.float() * -0.625).double()
    scaled_image, scaled_history, _ = cgls(op, scaled_observed, ITERATIONS)
    arrays = dict(inputs, truth=truth, observed=observed.numpy(), rtm=rtm.numpy(),
                  rtm_scaled=rtm_scaled.numpy(), image=model.numpy(), predicted=pred.numpy(),
                  residual_history=history, iterations=np.array(ITERATIONS),
                  columns=columns, windows=windows, reflector_depths=depths,
                  rtm_depths=rtm_depth, rtm_widths=rtm_width, ls_depths=ls_depth,
                  ls_widths=ls_width, truth_depths=truth_depth, truth_widths=truth_width,
                  zero_objective=loss.numpy(),
                  zero_model=np.zeros_like(truth), rtm_scaled_predicted=rtm_pred.numpy(),
                  scaled_observed=scaled_observed.float().numpy(),
                  scaled_image=scaled_image.numpy(), scaled_rtm=op.adjoint(scaled_observed).numpy(),
                  scaled_predicted=op.forward(scaled_image).detach().numpy(),
                  scaled_residual_history=scaled_history)
    manifest = write_case(outdir, "layered", arrays)
    # Float32 feasibility diagnostic is recorded, never used to relax criteria.
    op32 = Born(inputs, torch.float32)
    m32, h32, _ = cgls(op32, observed.float(), ITERATIONS)
    pred32 = op32.forward(m32).detach().numpy()
    rtm32 = op32.adjoint(observed.float()).numpy()
    float32_errors = {
        "image_max_relative": float(np.max(np.abs(m32.numpy() - model.numpy())) / np.max(np.abs(model.numpy()))),
        "predicted_max_relative": float(np.max(np.abs(pred32 - pred.numpy())) / np.max(np.abs(pred.numpy()))),
        "rtm_max_relative": float(np.max(np.abs(rtm32 - rtm.numpy())) / np.max(np.abs(rtm.numpy()))),
        "history_max_absolute": float(np.max(np.abs(h32 - history))),
    }
    print("float32 diagnostic:", float32_errors, flush=True)
    assert max(float32_errors.values()) <= TOL["PARITY_RTOL"]
    depth32, width32 = reflector_metrics(m32.numpy(), columns, windows)
    _, rtm_width32 = reflector_metrics(rtm32, columns, windows)
    assert np.all(width32 <= TOL["MAX_WIDTH_RATIO"] * rtm_width32)
    assert np.all(width32 <= TOL["MAX_WIDTH_RATIO"] * rtm_width)
    assert np.all(np.abs(depth32 - depths[:, None]) <= TOL["MAX_DEPTH_ERROR_CELLS"])
    return manifest, {"residual_ratio": float(history[-1]),
                      "residual_vs_scaled_rtm": float(torch.linalg.vector_norm(pred - observed) / torch.linalg.vector_norm(rtm_pred - observed)),
                      "rtm_widths_cells": rtm_width.tolist(), "ls_widths_cells": ls_width.tolist(),
                      "width_ratios": ratios.tolist(), "width_criterion_passes": True,
                      "minimum_width_ratio_margin": float(TOL["MAX_WIDTH_RATIO"] - np.max(ratios)),
                      "rtm_peak_depths_cells": rtm_depth.tolist(), "ls_peak_depths_cells": ls_depth.tolist(),
                      "truth_widths_cells": truth_width.tolist(),
                      "exact_truth_width_ratios": (truth_width/rtm_width).tolist(),
                      "float32_width_ratios": (width32/rtm_width32).tolist(),
                      "float32_diagnostic": float32_errors}


def small(outdir):
    rng = np.random.default_rng(SEED)
    inputs = survey(nz=18, nx=22, nt=280, velocity=1900.0, frequency=30.0)
    # Nonuniform background and nonzero signed model challenge axis/sign errors.
    z, x = np.indices(inputs["v"].shape)
    inputs["v"] = (inputs["v"] + 1.5 * z + 0.5 * x).astype(np.float32)
    truth = (rng.normal(size=z.shape) * inputs["mask"] * 25).astype(np.float32)
    model = (rng.normal(size=z.shape) * inputs["mask"] * 8).astype(np.float32)
    op = Born(inputs)
    observed = op.forward(tensor(truth)).detach().float().double()
    loss, grad, pred = op.objective(tensor(model), observed)
    directions = []
    # One dense direction and two coordinates with the largest nonzero gradients.
    direction = (rng.normal(size=z.shape) * inputs["mask"]).astype(np.float32)
    direction /= np.max(np.abs(direction))
    directions.append(direction)
    for i in np.argsort(np.abs(grad.numpy()).ravel())[-2:]:
        d = np.zeros_like(model)
        d.flat[i] = 1.0
        directions.append(d)
    fd_values, dot_values, fd_models_plus, fd_models_minus = [], [], [], []
    for direction in directions:
        d = tensor(direction)
        h = TOL["FD_STEP"]
        plus_model = (tensor(model) + h * d).float()
        minus_model = (tensor(model) - h * d).float()
        plus = op.objective(plus_model.double(), observed)[0]
        minus = op.objective(minus_model.double(), observed)[0]
        fd = float((plus - minus) / (2 * h))
        dot = float(torch.sum(grad * d))
        assert abs(fd - dot) / max(abs(fd), abs(dot)) < TOL["ORACLE_RTOL"]
        fd_values.append(fd)
        dot_values.append(dot)
        fd_models_plus.append(plus_model.numpy())
        fd_models_minus.append(minus_model.numpy())
    probe_data = rng.normal(size=observed.shape).astype(np.float32)
    adjoint = op.adjoint(tensor(probe_data))
    lhs = float(torch.sum(op.forward(tensor(model)) * tensor(probe_data)))
    rhs = float(torch.sum(tensor(model) * adjoint))
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < TOL["ORACLE_RTOL"]
    scalar_kw = dict(op.kw)
    scalar_kw["receiver_locations"] = scalar_kw.pop("sc_receiver_locations")
    eps = TOL["BORN_LINEARIZATION_STEP"]
    perturbation = op.mask * tensor(model)
    plus = deepwave.scalar(op.v + eps * perturbation, **scalar_kw)[-1]
    minus = deepwave.scalar(op.v - eps * perturbation, **scalar_kw)[-1]
    linearization_error = float(torch.max(torch.abs((plus - minus) / (2 * eps) - pred)) / torch.max(torch.abs(pred)))
    assert linearization_error <= TOL["ORACLE_RTOL"]
    print("small objective:", float(loss), "FD:", fd_values, "adjoint dot:", [lhs, rhs],
          "Born vs scalar FD:", linearization_error, flush=True)
    zero_loss, zero_grad, _ = op.objective(torch.zeros_like(op.v), observed)
    op32 = Born(inputs, torch.float32)
    loss32, grad32, pred32 = op32.objective(tensor(model, torch.float32), observed.float())
    f32_errors = {key: float(np.max(np.abs(a.numpy()-b.numpy())) / np.max(np.abs(b.numpy())))
                  for key, a, b in [("objective", loss32, loss), ("gradient", grad32, grad),
                                    ("predicted", pred32, pred)]}
    assert max(f32_errors.values()) <= TOL["PARITY_RTOL"]
    f32_fd_errors = []
    for d, plus_model, minus_model in zip(directions, fd_models_plus, fd_models_minus, strict=True):
        plus = op32.objective(tensor(plus_model, torch.float32), observed.float())[0]
        minus = op32.objective(tensor(minus_model, torch.float32), observed.float())[0]
        fd32 = (float(plus)-float(minus))/(2*TOL["FD_STEP"])
        dot32 = float(np.sum(grad32.numpy().astype(np.float64)*d.astype(np.float64)))
        f32_fd_errors.append(abs(fd32-dot32)/max(abs(fd32), abs(dot32)))
    assert max(f32_fd_errors) <= TOL["GRADIENT_RTOL"]
    model_outside = model + (1.0 - inputs["mask"]) * rng.normal(size=z.shape).astype(np.float32) * 100.0
    arrays = dict(inputs, truth=truth, model=model, observed=observed.numpy(),
                  objective=loss.numpy(), gradient=grad.numpy(), predicted=pred.numpy(),
                  directions=np.stack(directions), fd_values=np.asarray(fd_values),
                  probe_data=probe_data, adjoint=adjoint.numpy(), rtm=-zero_grad.numpy(),
                  zero_objective=zero_loss.numpy(), zero_model=np.zeros_like(model),
                  zero_observed=np.zeros(observed.shape, np.float32), model_outside=model_outside,
                  fd_models_plus=np.stack(fd_models_plus), fd_models_minus=np.stack(fd_models_minus))
    return write_case(outdir, "small", arrays), {"objective": float(loss),
            "directional_finite_differences": fd_values, "directional_gradient_products": dot_values,
            "adjoint_dot_lhs": lhs, "adjoint_dot_rhs": rhs,
            "born_vs_nonlinear_scalar_relative_error": linearization_error,
            "float32_max_relative_errors": f32_errors, "float32_fd_relative_errors": f32_fd_errors}


def main():
    start = time.monotonic()
    if VERSIONS != EXPECTED_VERSIONS:
        raise RuntimeError(f"Reference environment changed: {VERSIONS}; expected {EXPECTED_VERSIONS}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    outdir = ROOT / "judge/golden"
    outdir.mkdir(exist_ok=True)
    deepwave_dir = Path(deepwave.__file__).parent
    reference_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(deepwave_dir.glob("*"))
                        if p.suffix in {".py", ".h", ".c", ".dylib"}}
    # Deepwave here reports 0.1.0. Pin its actual source/native build as well,
    # so a different package with the same version cannot silently rejudge us.
    meta_path = outdir / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous["deepwave_source_sha256"] != reference_hashes:
            raise RuntimeError("Deepwave source/native build changed despite version pin; review required")
    cases, diagnostics = {}, {}
    cases["layered"], diagnostics["layered"] = layered(outdir)
    cases["small"], diagnostics["small"] = small(outdir)
    meta = {"schema_version": 3, "seed": SEED, "precision": "float64 CPU; supplied floats quantized to float32 first",
            "python": platform.python_version(), "library_versions": VERSIONS,
            "deepwave_source_sha256": reference_hashes,
            "lock_status": "REFERENCE_PASS: local locked scheme independently verified; full-column analytic-envelope FWHM; implementation parity not yet evaluated",
            "scheme": {"specification": "tests/SCHEME.md", "reference": "judge/scheme_reference.py",
                       "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("tests/SCHEME.md", "judge/scheme_reference.py")},
                       "validation": SCHEME_DIAGNOSTICS},
            "synthetic": "Exact project 17 layers_variable Born sheets, survey, velocity and data; no synthetic modifications",
            "project17_provenance": {"case": "layers_variable", "file": "judge/input/rtm17_layers_variable.npz",
                                     "sha256": hashlib.sha256((ROOT / "judge/input/rtm17_layers_variable.npz").read_bytes()).hexdigest()},
            "inputs": cases, "tolerances": {key: {"value": TOL[key], "reason": reason}
                                          for key, reason in TOL["REASONS"].items()},
            "conventions": "axes (z,x), z down, metres; scatter is velocity perturbation m/s; data are scattered field only",
            "objective": "0.5 sum((A m - observed)^2), A m = scalar_born(v, mask*m); no dt/dx weights or damping",
            "solver": f"{ITERATIONS} CGLS steps with two-pass full reorthogonalization; true residual recomputed each step",
            "rtm": "A^T d, Euclidean adjoint; scaled RTM is best-fit alpha*A^T d (first CGLS iterate)",
            "width_metric": METRICS["WIDTH_DEFINITION"],
            "width_transform_length_layered": int(cases["layered"]["v"]["shape"][0]),
            "width_definition_reason": "Route (a) from review: wave-packet envelope captures RTM side-lobe ringing; raw-amplitude central lobes saturate at the sampled one-cell floor on the unchanged project-17 survey.",
            "residual": "L2 norm vs zero start AND best-fit scaled RTM; both must fall at least 50%",
            "test_collection_count": 55,
            "diagnostics": diagnostics, "generation_seconds": time.monotonic()-start}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    assert sum(p.stat().st_size for p in outdir.iterdir()) < 20_000_000
    print(f"Goldens written in {time.monotonic()-start:.2f} s; exact project-17 Born synthetic used.")
    print(meta["lock_status"], flush=True)


if __name__ == "__main__":
    main()
