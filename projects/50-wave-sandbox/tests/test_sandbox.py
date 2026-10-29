"""Metal-only project: every collected case is explicitly a Metal case; no optional skips."""
from functools import cache
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from outcrop_harness import call, golden, golden_meta

from .tolerances import (
    CONVERGENCE_ORDER,
    CONVERGENCE_ORDER_SLACK,
    DESCENT_RATIO_MAX,
    EXACT_ABS,
    FD_STEP_MPS,
    GRAD_COS_MIN,
    GRAD_FD_REL,
    MIN_STEPS_PER_SECOND,
    PARITY_REL,
    PERFORMANCE_GRID_SIZE,
    PERFORMANCE_STEPS,
    RENDER_AUDIT_MARGIN_MIN,
    render_abs_tolerance,
)

PATHS = ["metal"]  # Metal only; project 50 has no MLX reference implementation.
CASES = [f"{mode}_o{o}" for mode in ("one_step", "evolved") for o in (2, 4, 6, 8)]
INPUTS = ("velocity", "state", "source_amplitudes", "source_locations",
          "receiver_locations", "grid_spacing", "dt", "accuracy", "pml_width",
          "pml_freq", "max_velocity", "boundary", "render_scale")


def arguments(g, *, fwi=False):
    keys = tuple(k for k in INPUTS if not (fwi and k == "render_scale"))
    result = {k: g[k].item() if g[k].ndim == 0 else g[k] for k in keys}
    result["grid_spacing"] = g["grid_spacing"].tolist()
    if fwi:
        result["observed"] = g["observed"]
    return result


def array_contract(out, name, shape):
    assert name in out, f"missing output {name}"
    a = out[name]
    assert isinstance(a, np.ndarray)
    assert a.shape == shape, (name, a.shape, shape)
    assert a.dtype == np.float32, (name, a.dtype)
    assert np.all(np.isfinite(a)), name
    return a


def parity(actual, expected, tolerance=PARITY_REL):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert a.shape == b.shape
    assert np.all(np.isfinite(a))
    scale = np.max(np.abs(b))
    error = np.max(np.abs(a-b))
    if scale == 0:
        assert error <= EXACT_ABS
    else:
        assert error/scale <= tolerance, f"relative max error {error/scale:.9g} > {tolerance}"


def check_physics(out, g):
    state = array_contract(out, "state", g["expected_state"].shape)
    # Normalize each memory separately; a large pressure cannot hide a bad PML.
    for channel in range(6):
        parity(state[channel], g["expected_state"][channel])
    # The normative PML projection is exact; small leakage into these slabs
    # must not be hidden by channel-wide relative normalization.
    p = int(g["pml_width"])
    ny, nx = g["velocity"].shape
    for channel in (2, 4):
        assert np.max(np.abs(state[channel, p:p+ny, :])) <= EXACT_ABS
    for channel in (3, 5):
        assert np.max(np.abs(state[channel, :, p:p+nx])) <= EXACT_ABS
    receivers = array_contract(out, "receiver_amplitudes", g["expected_receivers"].shape)
    parity(receivers, g["expected_receivers"])


def check_render(out, g):
    p = int(g["pml_width"])
    ny, nx = g["velocity"].shape
    physical = g["expected_state"][0, p:p+ny, p:p+nx]
    loc = g["receiver_locations"]
    # Match precisely the pressure scope represented by each output. Receiver
    # traces are pre-update, so append the final field and discard sample 0.
    probes = np.concatenate((g["expected_receivers"][:, 1:].T,
                             physical[loc[:, 0], loc[:, 1]][None]), axis=0)
    for name in ("rgba", "texture_samples"):
        a = array_contract(out, name, g[f"expected_{name}"].shape)
        pressure = physical if name == "rgba" else probes
        bound = render_abs_tolerance(float(np.max(np.abs(pressure))), float(g["render_scale"]))
        error = float(np.max(np.abs(a.astype(np.float64)-g[f"expected_{name}"])))
        assert error <= bound, (name, error, bound)
        np.testing.assert_array_equal(a[..., 1], 0)
        np.testing.assert_array_equal(a[..., 3], 1)
        assert np.all((a >= 0) & (a <= 1))
        assert np.all(a[..., 0]*a[..., 2] == 0)
    # Final probes and complete texture must agree, including the origin/axes.
    np.testing.assert_array_equal(out["texture_samples"][-1], out["rgba"][loc[:, 0], loc[:, 1]])


@cache
def frame_result(path, name):
    g = golden(name)
    return g, call(path, "run_frames", **arguments(g))


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_deepwave_physics_parity(path, case):
    g, out = frame_result(path, case)
    check_physics(out, g)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_every_frame_render_and_texture_orientation(path, case):
    g, out = frame_result(path, case)
    check_render(out, g)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", ["rest", "source_impulse"])
def test_rest_and_source_sign_calibration(path, case):
    g, out = frame_result(path, case)
    check_physics(out, g)
    check_render(out, g)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_analytic_refinement_second_order_in_time(path):
    errors = []
    for n in (16, 32, 64):
        g, out = frame_result(path, f"convergence_n{n}")
        check_physics(out, g)
        check_render(out, g)
        # Independent continuum eigenmode, not the finite-difference eigenvalue.
        errors.append(float(np.sqrt(np.mean((out["state"][0].astype(np.float64)-g["analytic_final"])**2))))
    observed = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all(np.isfinite(observed))
    assert np.all(np.abs(observed-CONVERGENCE_ORDER) <= CONVERGENCE_ORDER_SLACK), (errors, observed)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_1000_square_simulation_and_render_at_60_steps_per_second(path):
    g = golden("performance")
    assert g["velocity"].shape == (PERFORMANCE_GRID_SIZE, PERFORMANCE_GRID_SIZE)
    assert g["source_amplitudes"].shape[1] == PERFORMANCE_STEPS
    kwargs = arguments(g)  # Disk loading is outside the timer; transport is inside.
    started = perf_counter()
    out = call(path, "run_frames", **kwargs)
    elapsed = perf_counter()-started
    # Check the actual timed workload, not a separate small simulation.
    check_physics(out, g)
    check_render(out, g)
    rate = PERFORMANCE_STEPS/elapsed
    assert rate >= MIN_STEPS_PER_SECOND, f"{rate:.3f} steps/s, {elapsed:.6f}s total"


@cache
def fwi_result(path):
    g = golden("fwi")
    return g, call(path, "misfit_and_grad", **arguments(g, fwi=True))


def check_objective_gradient(out, g):
    j = array_contract(out, "objective", ())
    gradient = array_contract(out, "gradient", g["velocity"].shape)
    assert j >= 0
    return j, gradient


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_live_fwi_objective_gradient_and_direction_vs_deepwave(path):
    g, out = fwi_result(path)
    j, gradient = check_objective_gradient(out, g)
    parity(j, g["expected_objective"])
    parity(gradient, g["expected_gradient"])
    a, b = gradient.astype(np.float64).ravel(), g["expected_gradient"].ravel()
    cosine = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    assert cosine >= GRAD_COS_MIN


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("direction", [0, 1, 2], ids=["global", "anomaly", "gradient"])
def test_centered_velocity_gradient(path, direction):
    g, out = fwi_result(path)
    _, gradient = check_objective_gradient(out, g)
    d = g["directions"][direction]
    plus, minus = arguments(g, fwi=True), arguments(g, fwi=True)
    plus["velocity"] = (g["velocity"]+FD_STEP_MPS*d).astype(np.float32)
    minus["velocity"] = (g["velocity"]-FD_STEP_MPS*d).astype(np.float32)
    jp, _ = check_objective_gradient(call(path, "misfit_and_grad", **plus), g)
    jm, _ = check_objective_gradient(call(path, "misfit_and_grad", **minus), g)
    fd = (float(jp)-float(jm))/(2*FD_STEP_MPS)
    actual_direction = (plus["velocity"].astype(np.float64)-minus["velocity"])/(2*FD_STEP_MPS)
    adjoint = float(np.sum(gradient.astype(np.float64)*actual_direction))
    parity(np.array(fd), g["expected_fd"][direction], GRAD_FD_REL)
    parity(np.array(adjoint), np.array(fd), GRAD_FD_REL)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_live_fwi_gradient_step_reduces_misfit(path):
    g, out = fwi_result(path)
    j, gradient = check_objective_gradient(out, g)
    args = arguments(g, fwi=True)
    args["velocity"] = (g["velocity"]-float(g["descent_step"])*gradient).astype(np.float32)
    trial_j, _ = check_objective_gradient(call(path, "misfit_and_grad", **args), g)
    assert float(trial_j)/float(j) <= DESCENT_RATIO_MAX


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_locked_metadata_matches_thresholds(path):
    # Fail explicitly if metadata and test thresholds drift after regenerating.
    meta = golden_meta()
    assert meta["paths"] == PATHS and path in PATHS
    scheme = Path(__file__).with_name("SCHEME.md")
    assert sha256(scheme.read_bytes()).hexdigest() == meta["scheme_sha256"]
    assert meta["render_target_format"] == (
        "MTLPixelFormatRGBA32Float; linear float32; no intermediate quantization")
    from . import tolerances
    for name, entry in meta["tolerances"].items():
        assert getattr(tolerances, name) == entry["value"]
    for name, case in meta["render_float32_audit"].items():
        assert set(case) == {"deepwave_float32", "numpy_recurrence_float32"}
        for reference in case.values():
            for output in reference.values():
                scale = meta["inputs"][name]["inputs"]["render_scale"]["value"]
                bound = render_abs_tolerance(output["pressure_peak_au"], scale)
                assert output["absolute_bound"] == bound
                assert output["absolute_error"]*RENDER_AUDIT_MARGIN_MIN <= bound
    for key in ("expected_rgba", "expected_texture_samples"):
        for defect in ("swapped_red_blue", "doubled_render_scale", "omitted_clip"):
            assert any(case[key][defect] for case in meta["render_mutation_rejections"].values())
