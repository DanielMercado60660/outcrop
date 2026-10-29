"""Identical-grid float64 parity, physical symmetries, and actual gradients."""
import numpy as np
import pytest
from outcrop_harness import call, golden

from .conftest import inputs, objective_inputs
from .tolerances import (
    FD_RTOL,
    FD_SLOPE_FLOOR,
    FD_STEP,
    GRADIENT_ATOL,
    OBJECTIVE_ATOL,
    PARITY_RTOL,
    VOLTAGE_ATOL,
)


@pytest.mark.parametrize("case", ["conductor", "resistor", "gradient_data"])
def test_forward_float64_parity(path, case):
    g = golden(case)
    model = g["log10_rho"] if case == "gradient_data" else g["truth"]
    expected = g["predicted"] if case == "gradient_data" else g["clean"]
    out = call(path, "dc_predict", **inputs(g), log10_rho=model.astype(np.float32))
    assert set(out) == {"predicted"}
    assert out["predicted"].dtype == np.float32
    assert out["predicted"].shape == expected.shape
    assert np.isfinite(out["predicted"]).all()
    np.testing.assert_allclose(out["predicted"], expected, rtol=PARITY_RTOL, atol=VOLTAGE_ATOL)


@pytest.mark.parametrize("symmetry", ["reciprocal", "source_sign", "receiver_sign",
                                      "current_scale", "resistivity_scale", "row_order"])
def test_physical_conventions(path, symmetry):
    g = golden("gradient_data")
    kw = inputs(g)
    model = g["log10_rho"].astype(np.float32)
    expected = g["predicted"].copy()
    if symmetry == "reciprocal":
        kw["abmn"] = kw["abmn"][:, [2, 3, 0, 1]].copy()
    elif symmetry == "source_sign":
        kw["abmn"] = kw["abmn"][:, [1, 0, 2, 3]].copy()
        expected *= -1
    elif symmetry == "receiver_sign":
        kw["abmn"] = kw["abmn"][:, [0, 1, 3, 2]].copy()
        expected *= -1
    elif symmetry == "current_scale":
        kw["current"] *= 2
        expected *= 2
    elif symmetry == "resistivity_scale":
        model = (model.astype(np.float64) + np.log10(2)).astype(np.float32)
        expected *= 2
    elif symmetry == "row_order":
        # Saved deterministic permutation also exercises repeated rows.
        ix = g["row_selection"]
        kw["abmn"] = kw["abmn"][ix].copy()
        expected = expected[ix]
    out = call(path, "dc_predict", **kw, log10_rho=model)
    assert set(out) == {"predicted"}
    assert out["predicted"].shape == expected.shape
    assert out["predicted"].dtype == np.float32
    np.testing.assert_allclose(out["predicted"], expected, rtol=PARITY_RTOL, atol=VOLTAGE_ATOL)


def test_objective_gradient_float64_parity(evaluated):
    g, out = evaluated
    np.testing.assert_allclose(out["predicted"], g["predicted"],
                               rtol=PARITY_RTOL, atol=VOLTAGE_ATOL)
    np.testing.assert_allclose(out["objective"], g["objective"],
                               rtol=PARITY_RTOL, atol=OBJECTIVE_ATOL)
    np.testing.assert_allclose(out["gradient"], g["gradient"],
                               rtol=PARITY_RTOL, atol=GRADIENT_ATOL)


def test_objective_definition(evaluated):
    g, out = evaluated
    residual = (out["predicted"].astype(np.float64) - g["observed"]) / g["standard_deviation"]
    delta = g["log10_rho"] - g["reference"]
    penalty = np.sum(delta**2) + np.sum(np.diff(delta, axis=0)**2) + np.sum(np.diff(delta, axis=1)**2)
    expected = np.mean(residual**2) / 2 + float(g["beta"]) * penalty / 2
    np.testing.assert_allclose(out["objective"], expected,
                               rtol=PARITY_RTOL, atol=OBJECTIVE_ATOL)


@pytest.mark.parametrize("direction", range(6), ids=lambda n: f"direction{n}")
def test_centered_finite_difference(path, evaluated, direction):
    g, out = evaluated
    v = g["directions"][direction]
    # h=0.01 dex resolves float32 objective noise at objectives ~1700;
    # the generator replays both assertions with float32 solves and autodiff.
    # Construct perturbations in float64, round once to implementation inputs.
    plus = (g["log10_rho"] + FD_STEP * v).astype(np.float32)
    minus = (g["log10_rho"] - FD_STEP * v).astype(np.float32)
    f_plus = call(path, "dc_misfit_and_grad", **objective_inputs(g, plus))
    f_minus = call(path, "dc_misfit_and_grad", **objective_inputs(g, minus))
    for result in (f_plus, f_minus):
        assert result["objective"].shape == ()
        assert result["objective"].dtype == np.float32
        assert np.isfinite(result["objective"])
    slope = (float(f_plus["objective"]) - float(f_minus["objective"])) / (2 * FD_STEP)
    # Account for the actual rounded perturbation in the direction contraction.
    actual_v = (plus.astype(np.float64) - minus.astype(np.float64)) / (2 * FD_STEP)
    adjoint = float(np.sum(out["gradient"].astype(np.float64) * actual_v))
    error = abs(slope - adjoint) / max(abs(adjoint), FD_SLOPE_FLOOR)
    assert error <= FD_RTOL, (slope, adjoint, error)
    # The saved float64 difference uses actual_v, including model-input
    # rounding, rather than the nominal v. The generator replays this assertion.
    oracle_slope = float(g["fd_slopes"][direction])
    assert abs(adjoint - oracle_slope) / max(abs(oracle_slope), FD_SLOPE_FLOOR) <= FD_RTOL
