"""Both paths: same-grid Deepwave parity and independent physical judges."""
import hashlib

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, golden_meta, project_dir

from . import tolerances as tol
from ._support import (
    CASES,
    COMPONENTS,
    FIELDS,
    arguments,
    arrival_times,
    compare,
    component_floor,
    forward,
    gradient,
    objective,
    reflection_error,
)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_deepwave_parity(path, case):
    g, out = forward(path, case)
    for k in COMPONENTS:
        compare(out[k], g["expected_"+k], floor=component_floor(case, g, k))


@pytest.mark.parametrize("paths", [("mlx", "metal")], ids=["mlx-metal"])
@pytest.mark.parametrize("case", CASES)
def test_cross_path_parity(paths, case):
    g, a = forward(paths[0], case)
    _, b = forward(paths[1], case)
    for k in COMPONENTS:
        # Normalize to the oracle, avoiding a broken/zero path defining scale.
        compare(a[k]-b[k]+g["expected_"+k], g["expected_"+k], tol.CROSS,
                floor=component_floor(case, g, k))


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("mode,component", [("P", "vy"), ("S", "vx")])
def test_second_order_continuum_convergence(path, mode, component):
    errors = []
    for level in range(3):
        g, out = forward(path, f"plane_{mode}_{level}")
        numerical = np.asarray(out[component][0, 0], np.float64)
        errors.append(np.linalg.norm(numerical-g["analytic"])/np.linalg.norm(g["analytic"]))
    orders = np.log2(np.asarray(errors[:-1])/errors[1:])
    assert np.all(np.isfinite(orders))
    assert np.all(orders >= tol.ORDER_MIN), (errors, orders)
    assert np.all(orders <= tol.ORDER_MAX), (errors, orders)
    assert errors[-1] <= tol.PLANE_FINE_ERROR, errors


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_lamb_rayleigh_arrival_within_one_percent(path):
    g, out = forward(path, "lamb")
    measured = arrival_times(out["vy"], g)
    errors = np.abs(measured-g["analytic_arrival"])/g["analytic_arrival"]
    assert np.all(errors <= tol.RAYLEIGH_TIME), (measured, errors)
    # Also test differential travel time: no source-time/offset cancellation.
    lags = np.diff(measured)
    analytic_lags = np.diff(g["offsets"])/float(g["rayleigh_speed"])
    assert np.all(np.abs(lags-analytic_lags)/analytic_lags <= tol.RAYLEIGH_TIME)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_cpml_absorbs_p_and_sv(path):
    g, out = forward(path, "pml")
    control_g, control = forward(path, "reflective")
    pml_error = reflection_error(out, g)
    control_error = reflection_error(control, control_g)
    assert np.all(pml_error <= tol.PML_REFLECTION), pml_error
    assert np.all(control_error >= tol.REFLECTIVE_MIN), control_error
    assert np.all(control_error >= tol.PML_IMPROVEMENT*pml_error), (pml_error, control_error)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", ["gradient", "gradient_surface"])
def test_discrete_adjoint_reference_and_objective(path, case):
    g, _args, out = gradient(path, case)
    for k, value in out.items():
        compare(value, g["expected_"+k])
    forward_out = call(path, "elastic", **arguments(g))
    compare(out["objective"], np.asarray(objective(forward_out, g)))
    assert float(out["objective"]) > tol.ZERO
    if case == "gradient_surface":
        for field in ("vp", "vs", "rho"):
            assert np.all(out["grad_"+field][:2] == tol.ZERO)


@pytest.mark.parametrize("paths", [("mlx", "metal")], ids=["mlx-metal"])
@pytest.mark.parametrize("case", ["gradient", "gradient_surface"])
def test_cross_path_gradients(paths, case):
    g, _, a = gradient(paths[0], case)
    _, _, b = gradient(paths[1], case)
    for k in a:
        compare(a[k]-b[k]+g["expected_"+k], g["expected_"+k], tol.CROSS)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", ["gradient", "gradient_surface"])
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("direction", [0, 1], ids=["global", "seeded-local"])
def test_centered_finite_difference_gradient(path, case, field, direction):
    g, args, out = gradient(path, case)
    d = np.asarray(g["direction_"+field][direction], np.float64)
    base = np.asarray(args[field], np.float64)
    plus = {**args, field: np.ascontiguousarray(base+tol.FD_STEP*d, dtype=np.float32)}
    minus = {**args, field: np.ascontiguousarray(base-tol.FD_STEP*d, dtype=np.float32)}
    for kwargs in (plus, minus):
        for k in ("observed_y", "observed_x", "data_scale"):
            kwargs.pop(k)
    fp = objective(call(path, "elastic", **plus), g)
    fm = objective(call(path, "elastic", **minus), g)
    fd = (fp-fm)/(2*tol.FD_STEP)
    # Use the actual representable symmetric displacement; this removes
    # quantization of a float32 input perturbation from the derivative budget.
    effective_d = (plus[field].astype(np.float64)-minus[field].astype(np.float64))/(2*tol.FD_STEP)
    adjoint = np.sum(out["grad_"+field].astype(np.float64)*effective_d)
    denominator = max(abs(fd), abs(adjoint))
    assert denominator > tol.ZERO, "unilluminated/degenerate gradient direction"
    error = abs(fd-adjoint)/denominator
    assert error <= tol.GRAD_FD, (field, direction, fd, adjoint, error)


def test_oracle_record_and_size():
    meta = golden_meta()
    assert meta["seed"] == 150127
    assert meta["tolerances"] == {k: getattr(tol, k) for k in meta["tolerances"]}
    assert meta["source_sha256"], "unfingerprinted Deepwave source build"
    assert set(meta["scheme_sha256"]) == {"tests/SCHEME.md", "tests/api.py"}
    for filename, digest in meta["scheme_sha256"].items():
        assert hashlib.sha256((project_dir()/filename).read_bytes()).hexdigest() == digest, filename
    assert sum(meta["golden_bytes"].values()) < tol.GOLDEN_BYTES_LIMIT
    for filename, digest in meta["golden_sha256"].items():
        file = project_dir()/"judge/golden"/filename
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest, filename
        assert file.stat().st_size == meta["golden_bytes"][filename]
    for case in (*CASES, "gradient", "gradient_surface"):
        g = golden(case)
        assert all(np.isfinite(a).all() for a in g.values() if a.dtype.kind in "fiu")
        assert g["expected_vy"].dtype == np.float64
    # The continuum threshold must accommodate every perturbation allowed by
    # same-grid parity, not just the oracle's own discretization error.
    for mode, component in (("P", "vy"), ("S", "vx")):
        g = golden(f"plane_{mode}_2")
        reference = g["expected_"+component][0, 0]
        norm = np.linalg.norm(g["analytic"])
        error = np.linalg.norm(reference-g["analytic"])/norm
        budget = tol.PARITY*np.max(np.abs(reference))*np.sqrt(reference.size)/norm
        assert error+budget <= tol.PLANE_FINE_ERROR, (mode, error, budget)
