"""Locked MLX parity, gradient, optimizer and noisy-model recovery checks."""
from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from outcrop_harness import call, golden, rel_err
from tolerances import (
    FD_RTOL,
    FD_STEP,
    MAX_ERROR_RATIO,
    MAX_INITIAL_ERROR_RATIO,
    PARITY_RTOL,
)

# MLX-only project. Metal and MLX-vs-Metal tests are inapplicable, not skipped.
PATHS = ["mlx"]
CASES = ["noise_a", "noise_b"]
MODES = ["grid", "neural"]
DIRECTIONS = [
    pytest.param("grid", 0, id="grid-even"),
    pytest.param("grid", 1, id="grid-odd"),
    pytest.param("neural", 0, id="neural-W1"),
    pytest.param("neural", 1, id="neural-b1"),
    pytest.param("neural", 2, id="neural-W2"),
    pytest.param("neural", 3, id="neural-b2"),
]


@pytest.fixture(scope="module", params=PATHS, ids=PATHS)
def path(request):
    return request.param


@pytest.fixture(scope="module", params=CASES, ids=CASES)
def case(request):
    return request.param


@cache
def _golden(case):
    return golden(case)


def _kwargs(g, mode, parameters):
    # Deliberate allowlist: truth, clean gathers and reference outputs never go
    # to an implementation. It receives only saved acquisition/training inputs.
    kwargs = {key: g[key].copy() for key in (
        "latent", "initial_velocity", "observed", "source_amplitudes",
        "source_locations", "receiver_locations",
    )}
    kwargs.update(parameters=parameters.copy(), parameterization=mode,
                  spacing=float(g["spacing"]), dt=float(g["dt"]),
                  data_scale=float(g["data_scale"]), velocity_scale=float(g["velocity_scale"]),
                  coarse_shape=g["coarse_shape"].tolist(), hidden_size=int(g["hidden_size"]))
    return kwargs


def _call(path, fn, kwargs):
    before = {key: value.copy() for key, value in kwargs.items() if isinstance(value, np.ndarray)}
    out = call(path, fn, **kwargs)
    for key, value in before.items():
        np.testing.assert_array_equal(kwargs[key], value, err_msg=f"{fn} mutated input {key}")
    return out


def _check_output(out, shapes):
    assert set(out) == set(shapes), f"expected {set(shapes)}, got {set(out)}"
    for key, shape in shapes.items():
        assert isinstance(out[key], np.ndarray), key
        assert out[key].shape == shape, (key, out[key].shape, shape)
        assert out[key].dtype == np.dtype("float32"), (key, out[key].dtype)
        assert np.isfinite(out[key]).all(), key


def _check_misfit(out, g, p):
    _check_output(out, {"objective": (), "gradient": p.shape,
                        "predicted": g["observed"].shape,
                        "velocity": g["initial_velocity"].shape})
    # A self-consistency check independent of the stored reference objective.
    residual = (out["predicted"].astype(np.float64)-g["observed"].astype(np.float64))/float(g["data_scale"])
    _parity(out["objective"], 0.5*np.mean(residual**2), "objective definition")
    assert np.all(out["velocity"] > g["initial_velocity"]-float(g["velocity_scale"]))
    assert np.all(out["velocity"] < g["initial_velocity"]+float(g["velocity_scale"]))


def _parity(actual, expected, label):
    error = rel_err(actual, expected)
    assert error <= PARITY_RTOL, f"{label}: relative max error {error} > {PARITY_RTOL}"


@cache
def _probe(path, case, mode):
    g = _golden(case)
    p = g[f"{mode}_probe_parameters"]
    out = _call(path, "misfit_and_grad", _kwargs(g, mode, p))
    _check_misfit(out, g, p)
    return out


@cache
def _inversion(path, case, mode, steps):
    g = _golden(case)
    p = g[f"{mode}_initial_parameters"]
    kwargs = _kwargs(g, mode, p)
    kwargs.update(steps=steps, learning_rate=float(g["learning_rate"]))
    out = _call(path, "invert", kwargs)
    _check_output(out, {"parameters": p.shape, "velocity": g["initial_velocity"].shape,
                        "loss_history": (steps+1,)})
    return out


@pytest.mark.parametrize("mode", MODES)
def test_objective_and_gradient_parity(path, case, mode):
    g, out = _golden(case), _probe(path, case, mode)
    for key in ("objective", "gradient", "predicted"):
        _parity(out[key], g[f"{mode}_probe_{key}"], f"{mode} {key}")
    # Normalize velocity parity by the perturbation, rather than the large v0.
    _parity(out["velocity"]-g["initial_velocity"],
            g[f"{mode}_probe_velocity"]-g["initial_velocity"], "decoded perturbation")


@pytest.mark.parametrize("mode,direction_index", DIRECTIONS)
def test_centered_parameter_gradient(path, case, mode, direction_index):
    g, out = _golden(case), _probe(path, case, mode)
    p = g[f"{mode}_probe_parameters"]
    direction = g[f"{mode}_probe_directions"][direction_index]
    delta = np.float32(FD_STEP)*direction
    upper = _call(path, "misfit_and_grad", _kwargs(g, mode, p+delta))
    lower = _call(path, "misfit_and_grad", _kwargs(g, mode, p-delta))
    _check_misfit(upper, g, p)
    _check_misfit(lower, g, p)
    fd = (float(upper["objective"])-float(lower["objective"]))/ (2*FD_STEP)
    analytic = float(out["gradient"].astype(np.float64) @ direction.astype(np.float64))
    reference_fd = float(g[f"{mode}_probe_directional_fd"][direction_index])
    # Check both the implementation's own derivative and the independent
    # float64 centered difference; a consistently wrong objective cannot pass.
    assert rel_err(analytic, fd) <= FD_RTOL, (mode, direction_index, analytic, fd)
    assert rel_err(fd, reference_fd) <= FD_RTOL, (mode, direction_index, fd, reference_fd)
    _parity(analytic, reference_fd, f"{mode} block {direction_index} derivative")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("steps,label", [pytest.param(1, "one", id="one-step"),
                                         pytest.param(160, "final", id="160-steps")])
def test_fixed_adam_trajectory(path, case, mode, steps, label):
    g, out = _golden(case), _inversion(path, case, mode, steps)
    for key in ("parameters", "loss_history"):
        _parity(out[key], g[f"{mode}_{label}_{key}"], f"{mode} {key}")
    _parity(out["velocity"]-g["initial_velocity"],
            g[f"{mode}_{label}_velocity"]-g["initial_velocity"], "inverted perturbation")
    _parity(out["loss_history"][0], g["initial_objective"], "initial loss")
    # Re-evaluate the returned parameters: returned models and losses must be
    # tied to the actual decoder, wave equation and input noisy data.
    endpoint = _call(path, "misfit_and_grad", _kwargs(g, mode, out["parameters"]))
    _check_misfit(endpoint, g, out["parameters"])
    _parity(endpoint["predicted"], g[f"{mode}_{label}_predicted"], "endpoint gathers")
    _parity(endpoint["velocity"]-g["initial_velocity"],
            out["velocity"]-g["initial_velocity"], "endpoint model consistency")
    _parity(endpoint["objective"], out["loss_history"][-1], "endpoint loss consistency")
    assert out["loss_history"][-1] < out["loss_history"][0]


def test_neural_model_error_beats_plain_fwi_by_ten_percent(path, case):
    g = _golden(case)
    steps = int(g["steps"])
    neural = _inversion(path, case, "neural", steps)["velocity"].astype(np.float64)
    plain = _inversion(path, case, "grid", steps)["velocity"].astype(np.float64)
    truth = g["truth"]
    denominator = np.linalg.norm(truth)
    neural_error = np.linalg.norm(neural-truth)/denominator
    plain_error = np.linalg.norm(plain-truth)/denominator
    reference_plain_error = np.linalg.norm(g["grid_final_velocity"]-truth)/denominator
    initial_error = np.linalg.norm(g["initial_velocity"].astype(np.float64)-truth)/denominator
    # Entire physical grid, no favorable mask or exclusion of boundary cells.
    assert neural_error <= MAX_ERROR_RATIO*reference_plain_error, (neural_error, reference_plain_error)
    assert neural_error <= MAX_ERROR_RATIO*plain_error, (neural_error, plain_error)
    assert neural_error < MAX_INITIAL_ERROR_RATIO*initial_error, (neural_error, initial_error)


def test_both_parameterizations_start_from_identical_physics(path, case):
    g = _golden(case)
    outputs = {}
    for mode in MODES:
        p = g[f"{mode}_initial_parameters"]
        out = _call(path, "misfit_and_grad", _kwargs(g, mode, p))
        _check_misfit(out, g, p)
        np.testing.assert_array_equal(out["velocity"], g["initial_velocity"])
        _parity(out["objective"], g["initial_objective"], "initial objective")
        _parity(out["predicted"], g["initial_prediction"], "initial gathers")
        outputs[mode] = out
    _parity(outputs["grid"]["predicted"], outputs["neural"]["predicted"], "common starting physics")


@pytest.mark.parametrize("mode", MODES)
def test_repeated_one_step_is_deterministic(path, case, mode):
    g = _golden(case)
    expected = _inversion(path, case, mode, 1)
    kwargs = _kwargs(g, mode, g[f"{mode}_initial_parameters"])
    kwargs.update(steps=1, learning_rate=float(g["learning_rate"]))
    repeated = _call(path, "invert", kwargs)
    _check_output(repeated, {key: value.shape for key, value in expected.items()})
    for key in expected:
        np.testing.assert_array_equal(repeated[key], expected[key], err_msg=f"non-deterministic {mode} {key}")
