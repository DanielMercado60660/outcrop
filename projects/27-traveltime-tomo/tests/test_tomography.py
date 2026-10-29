"""Locked fixed-ray tomography parity, recovery and derivative tests."""

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import (
    CHECKERBOARD_MIN_CORRELATION,
    GEOMETRY_ATOL_M,
    GRADIENT_FD_RTOL,
    PARITY_RTOL,
    SMALL_SYSTEM_RTOL,
)

GEOMETRIES = ["geometry_hand", "geometry_irregular", "geometry_checkerboard"]
INVERSES = ["small_hand", "small_damped", "checkerboard_clean", "checkerboard_noisy"]
CHECKERBOARDS = ["checkerboard_clean", "checkerboard_noisy"]
GRADIENTS = ["gradient_plain", "gradient_damped"]


def assert_output(out, shapes):
    assert set(out) == set(shapes), f"Expected keys {set(shapes)}, got {set(out)}"
    for key, shape in shapes.items():
        assert isinstance(out[key], np.ndarray), key
        assert out[key].dtype == np.dtype("float32"), (key, out[key].dtype)
        assert out[key].shape == shape, (key, out[key].shape, shape)
        assert np.all(np.isfinite(out[key])), f"Nonfinite {key}"


def assert_relative(actual, expected, limit=PARITY_RTOL):
    error = rel_err(actual, expected)
    assert error <= limit, f"relative max error {error:.9g} > {limit:.9g}"


def geometry_args(g):
    return {key: g[key] for key in ("x_edges", "z_edges", "sources", "receivers")}


def inverse_args(g):
    return {"matrix": g["matrix"], "travel_times": g["travel_times"],
            "reference_slowness": g["reference_slowness"], "damping": float(g["damping"])}


def objective_args(g, model=None):
    return {**inverse_args(g), "slowness": g["slowness"] if model is None else model}


def check_solution(out, g):
    assert_output(out, {"slowness": g["slowness"].shape, "predicted_times": g["travel_times"].shape})
    # All inverse fixtures, including the rank-deficient damped system, obey
    # the stricter small-system limit where their name starts with 'small'.
    limit = SMALL_SYSTEM_RTOL if g["matrix"].shape[1] <= 4 else PARITY_RTOL
    assert_relative(out["slowness"], g["slowness"], limit)
    assert_relative(out["predicted_times"], g["predicted_times"])
    assert_relative(out["predicted_times"], g["matrix"].astype(np.float64) @ out["slowness"].astype(np.float64))
    if "hand_slowness" in g:
        assert_relative(out["slowness"], g["hand_slowness"], SMALL_SYSTEM_RTOL)


def check_recovery(slowness, g):
    mask = g["mask"]
    recovered = slowness.astype(np.float64)[mask] - g["reference_slowness"].astype(np.float64)[mask]
    truth = g["true_slowness"].astype(np.float64)[mask] - g["reference_slowness"].astype(np.float64)[mask]
    # A constant image has undefined correlation and must fail explicitly.
    assert np.std(recovered) > 0
    correlation = float(np.corrcoef(recovered, truth)[0, 1])
    assert np.isfinite(correlation)
    assert correlation >= CHECKERBOARD_MIN_CORRELATION, correlation
    assert np.all(slowness > 0), "These physical fixtures must recover positive slowness"


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", GEOMETRIES)
def test_ray_geometry(path, case):
    g = golden(case)
    args = geometry_args(g)
    out = call(path, "ray_matrix", **args)
    assert_output(out, {"matrix": g["matrix"].shape})
    assert np.all(out["matrix"] >= 0)
    assert_relative(out["matrix"], g["matrix"])
    np.testing.assert_allclose(out["matrix"], g["matrix"], rtol=PARITY_RTOL, atol=GEOMETRY_ATOL_M)
    assert_relative(out["matrix"].astype(np.float64).sum(axis=1), g["path_lengths"])
    reverse = call(path, "ray_matrix", **{**args, "sources": g["receivers"], "receivers": g["sources"]})
    assert_output(reverse, {"matrix": g["matrix"].shape})
    assert_relative(reverse["matrix"], g["matrix"])
    np.testing.assert_allclose(reverse["matrix"], g["matrix"], rtol=PARITY_RTOL, atol=GEOMETRY_ATOL_M)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", INVERSES)
def test_least_squares(path, case):
    g = golden(case)
    out = call(path, "solve_tomography", **inverse_args(g))
    check_solution(out, g)
    if case in CHECKERBOARDS:
        check_recovery(out["slowness"], g)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CHECKERBOARDS)
def test_checkerboard_pipeline(path, case):
    g = golden(case)
    geometry = golden("geometry_checkerboard")
    lengths = call(path, "ray_matrix", **geometry_args(geometry))
    assert_output(lengths, {"matrix": g["matrix"].shape})
    assert_relative(lengths["matrix"], geometry["matrix"])
    out = call(path, "solve_tomography", **{**inverse_args(g), "matrix": np.ascontiguousarray(lengths["matrix"])})
    # Use the actual returned geometry for prediction consistency, and the
    # original float64 LS oracle for model parity and resolution.
    assert_output(out, {"slowness": g["slowness"].shape, "predicted_times": g["travel_times"].shape})
    assert_relative(out["slowness"], g["slowness"])
    assert_relative(out["predicted_times"], g["predicted_times"])
    assert_relative(out["predicted_times"], lengths["matrix"].astype(np.float64) @ out["slowness"].astype(np.float64))
    check_recovery(out["slowness"], g)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", GRADIENTS)
def test_objective_and_gradient(path, case):
    g = golden(case)
    out = call(path, "misfit_and_grad", **objective_args(g))
    assert_output(out, {"objective": (), "gradient": g["slowness"].shape})
    assert_relative(out["objective"], g["objective"])
    assert_relative(out["gradient"], g["gradient"])


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", GRADIENTS)
def test_centered_finite_difference(path, case):
    g = golden(case)
    out = call(path, "misfit_and_grad", **objective_args(g))
    assert_output(out, {"objective": (), "gradient": g["slowness"].shape})
    derivative = np.empty_like(g["gradient"], dtype=np.float64)
    for j, (plus, minus) in enumerate(zip(g["fd_plus"], g["fd_minus"], strict=True)):
        p = call(path, "misfit_and_grad", **objective_args(g, plus))
        m = call(path, "misfit_and_grad", **objective_args(g, minus))
        for probe, expected in ((p, g["fd_plus_objectives"][j]), (m, g["fd_minus_objectives"][j])):
            assert_output(probe, {"objective": (), "gradient": g["slowness"].shape})
            assert_relative(probe["objective"], expected)
        derivative[j] = (float(p["objective"]) - float(m["objective"])) / g["fd_separation"][j]
    assert_relative(derivative, g["fd_gradient"], GRADIENT_FD_RTOL)
    # Every coordinate, not just one favorable directional derivative.
    np.testing.assert_allclose(out["gradient"], derivative, rtol=GRADIENT_FD_RTOL, atol=0.0)


@pytest.mark.parametrize("case", GEOMETRIES)
def test_mlx_metal_geometry_crosscheck(case):
    g = golden(case)
    outputs = [call(path, "ray_matrix", **geometry_args(g)) for path in PATHS]
    for out in outputs:
        assert_output(out, {"matrix": g["matrix"].shape})
    assert_relative(outputs[0]["matrix"], outputs[1]["matrix"])
    np.testing.assert_allclose(outputs[0]["matrix"], outputs[1]["matrix"], rtol=PARITY_RTOL, atol=GEOMETRY_ATOL_M)


@pytest.mark.parametrize("case", INVERSES)
def test_mlx_metal_solution_crosscheck(case):
    g = golden(case)
    outputs = [call(path, "solve_tomography", **inverse_args(g)) for path in PATHS]
    for out in outputs:
        check_solution(out, g)
    for key in ("slowness", "predicted_times"):
        assert_relative(outputs[0][key], outputs[1][key])


@pytest.mark.parametrize("case", GRADIENTS)
def test_mlx_metal_gradient_crosscheck(case):
    g = golden(case)
    outputs = [call(path, "misfit_and_grad", **objective_args(g)) for path in PATHS]
    for out in outputs:
        assert_output(out, {"objective": (), "gradient": g["slowness"].shape})
    for key in ("objective", "gradient"):
        assert_relative(outputs[0][key], outputs[1][key])
