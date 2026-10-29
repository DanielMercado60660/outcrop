"""Frozen parity, linearity, adjoint, and analytic convergence checks."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import (CONVERGENCE_FINE, CONVERGENCE_RATIO,
                        CONVERGENCE_RATIO_MAX, CROSS, GRADIENT, PARITY)

ARRAYS = ("v", "dv", "source_amplitudes", "source_locations", "receiver_locations",
          "wavefield_0", "wavefield_m1", "wavefield_sc_0", "wavefield_sc_m1")
SCALARS = ("dx", "dt", "accuracy", "boundary", "pml_width", "pml_freq", "max_vel")


def inputs(case):
    return {**{key: case[key].copy() for key in ARRAYS},
            **{key: case[key].item() for key in SCALARS}}


def check_output(actual, expected, tol):
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert rel_err(actual, expected) <= tol


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", ["pml2", "pml4", "mode24", "mode48", "mode96"])
def test_reference(path, name):
    c = golden(name)
    out = call(path, "born", **inputs(c))
    assert set(out) == {"background", "scattered"}
    for field in ("background", "scattered"):
        check_output(out[field], c[field], PARITY)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", ["pml2", "pml4"])
def test_gradient_and_centered_difference(path, name):
    c = golden(name)
    kw = inputs(c)
    out = call(path, "misfit_and_grad", **kw, observed=c["observed"])
    assert set(out) == {"objective", "gradient"}
    check_output(out["objective"], c["objective"], PARITY)
    check_output(out["gradient"], c["gradient"], PARITY)
    # 0.5 m/s balances float32 subtraction against the exact quadratic
    # directional dependence of this linear Born least-squares objective.
    step = 0.5
    objectives = []
    for sign in (-1, 1):
        d = kw["dv"] + np.float32(sign * step) * c["direction"]
        result = call(path, "born", **{**kw, "dv": d})["scattered"]
        residual = result.astype(np.float64) - c["observed"].astype(np.float64)
        objectives.append(.5 * np.sum(residual * residual))
    fd = (objectives[1] - objectives[0]) / (2 * step)
    adj = np.sum(out["gradient"].astype(np.float64) * c["direction"])
    assert abs(fd-adj) / abs(adj) <= GRADIENT


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_zero_and_linearity(path):
    c = golden("pml2")
    kw = inputs(c)
    zero = call(path, "born", **{**kw, "dv": np.zeros_like(kw["dv"])})
    assert np.count_nonzero(zero["scattered"]) == 0
    scaled = call(path, "born", **{**kw, "dv": -2 * kw["dv"]})
    check_output(scaled["scattered"], -2*c["scattered"], PARITY)
    check_output(zero["background"], c["background"], PARITY)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_analytic_convergence(path):
    errors = []
    for n in (24, 48, 96):
        c = golden(f"mode{n}")
        data = call(path, "born", **inputs(c))["scattered"]
        errors.append(abs(float(data[0, 0, -1]) - float(c["analytic_final"])))
    assert CONVERGENCE_RATIO < errors[0] / errors[1] < CONVERGENCE_RATIO_MAX
    assert CONVERGENCE_RATIO < errors[1] / errors[2] < CONVERGENCE_RATIO_MAX
    assert errors[2] < CONVERGENCE_FINE


@pytest.mark.parametrize("name", ["pml2", "pml4", "mode24", "mode48", "mode96"])
def test_mlx_metal_crosscheck(name):
    c = golden(name)
    kw = inputs(c)
    a, b = call("mlx", "born", **kw), call("metal", "born", **kw)
    for field in ("background", "scattered"):
        check_output(a[field], b[field], CROSS)


@pytest.mark.parametrize("name", ["pml2", "pml4"])
def test_mlx_metal_gradient_crosscheck(name):
    c = golden(name)
    kw = {**inputs(c), "observed": c["observed"]}
    a = call("mlx", "misfit_and_grad", **kw)
    b = call("metal", "misfit_and_grad", **kw)
    for field in ("objective", "gradient"):
        check_output(a[field], b[field], CROSS)
