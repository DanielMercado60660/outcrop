"""Locked case 1a, same-discretization parity, and independent FD order tests.

No reference solvers are imported here. All stimuli and exact solutions are
loaded from judge/golden. The few formulas here only check conservation and
diagnostic consistency of returned fields; they do not advance or solve a PDE.
"""

from functools import cache

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden

from .tolerances import (
    BENCHMARK_RTOL,
    CROSS_RTOL,
    CROSS_ZERO_ATOL,
    DIVERGENCE_RTOL,
    FINE_DIFFUSION_ERROR,
    FINE_STOKES_ERROR,
    HEAT_BALANCE_RTOL,
    MIN_SPATIAL_ORDER,
    MIN_TEMPORAL_ORDER,
    PARITY_RTOL,
    STEADY_RESIDUAL_RTOL,
    TEMPERATURE_ATOL,
    TRAJECTORY_RTOL,
    ZERO_ATOL,
)

STOKES_CASES = ["stokes_n12", "stokes_n17", "stokes_n32",
                "stokes_refine_n16", "stokes_refine_n32", "stokes_refine_n64"]
CONVECTION_CASES = ["conduction_zero", "conduction_evolved", "coupled_n12",
                    "coupled_n24", "coupled_refine_n8", "coupled_refine_n16",
                    "coupled_refine_n32", "diffusion_n8", "diffusion_n16",
                    "diffusion_n32", "time_s10", "time_s20", "time_s40", "case1a"]
STOKES_KEYS = {"psi", "u", "w"}
CONVECTION_KEYS = {"temperature", "u", "w", "nu_top", "nu_bottom", "vrms"}


@pytest.fixture(scope="module", params=PATHS, ids=PATHS)
def path(request):
    return request.param


@cache
def data(name):
    return golden(name)


def assert_close(actual, expected, rtol=PARITY_RTOL):
    """Project-wide infinity-norm relative criterion, with only zero-reference atol."""
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape
    assert np.isfinite(a).all()
    scale = float(np.max(np.abs(b)))
    error = float(np.max(np.abs(a-b)))
    if scale == 0:
        assert error <= ZERO_ATOL, f"zero-reference absolute error {error}"
    else:
        assert error/scale <= rtol, f"relative infinity error {error/scale} > {rtol}"


def assert_cross(a, b, reference):
    """Normalize both paths by the same oracle, including its exact zeros."""
    scale = float(np.max(np.abs(reference)))
    error = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    bound = CROSS_RTOL*scale if scale else CROSS_ZERO_ATOL
    assert error <= bound, (error, bound)


@cache
def result(path, fn, name):
    g = data(name)
    inp = g["input_temperature"].copy()
    kwargs = {"temperature": inp, "rayleigh": float(g["rayleigh"])}
    keys = STOKES_KEYS
    if fn == "convection":
        kwargs.update(dt=float(g["dt"]), steps=int(g["steps"]))
        keys = CONVECTION_KEYS
    out = call(path, fn, **kwargs)
    assert set(out) == keys
    for key in keys:
        value = out[key]
        assert isinstance(value, np.ndarray), key
        assert value.dtype == np.dtype("float32"), (key, value.dtype)
        assert value.shape == g[key].shape, (key, value.shape, g[key].shape)
        assert np.isfinite(value).all(), key
    return out


def check_velocity(v):
    """Impenetrability, streamfunction gauge, and centered discrete divergence."""
    u, w = v["u"].astype(np.float64), v["w"].astype(np.float64)
    n = u.shape[0]-1
    assert np.max(np.abs(u[:, [0, -1]])) <= ZERO_ATOL
    assert np.max(np.abs(w[[0, -1], :])) <= ZERO_ATOL
    if "psi" in v:
        psi = v["psi"]
        assert np.max(np.abs(psi[[0, -1], :])) <= ZERO_ATOL
        assert np.max(np.abs(psi[:, [0, -1]])) <= ZERO_ATOL
    div = (n/2)*(u[1:-1, 2:]-u[1:-1, :-2]+w[2:, 1:-1]-w[:-2, 1:-1])
    scale = n*max(float(np.max(np.abs(u))), float(np.max(np.abs(w))))
    if scale:
        assert np.max(np.abs(div))/scale <= DIVERGENCE_RTOL
    else:
        assert np.max(np.abs(div)) <= ZERO_ATOL


def check_diagnostics(out):
    """Recompute physical observables from returned fields, not target constants."""
    t = out["temperature"].astype(np.float64)
    u, w = out["u"].astype(np.float64), out["w"].astype(np.float64)
    n = t.shape[0]-1
    weights = np.ones(n+1)/n
    weights[[0, -1]] *= 0.5
    expected = {
        "nu_top": weights @ ((n/2)*(4*t[-2]-t[-3]-3*t[-1])),
        "nu_bottom": weights @ ((n/2)*(3*t[0]-4*t[1]+t[2])),
        "vrms": np.sqrt(weights @ (u*u+w*w) @ weights),
    }
    for key, value in expected.items():
        assert_close(out[key], value)


@pytest.mark.parametrize("name", STOKES_CASES, ids=STOKES_CASES)
def test_stokes_same_grid(path, name):
    g = data(name)
    out = result(path, "stokes", name)
    for key in STOKES_KEYS:
        assert_close(out[key], g[key])
    check_velocity(out)


@pytest.mark.parametrize("field", ["psi", "u", "w"])
def test_stokes_second_order_spatial_convergence(path, field):
    errors = []
    for n in (16, 32, 64):
        name = f"stokes_refine_n{n}"
        g = data(name)
        exact = g[f"exact_{field}"]
        actual = result(path, "stokes", name)[field].astype(np.float64)
        errors.append(float(np.max(np.abs(actual-exact))/np.max(np.abs(exact))))
    assert np.isfinite(errors).all() and min(errors) > 0
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all(orders >= MIN_SPATIAL_ORDER), (errors, orders)
    assert errors[-1] <= FINE_STOKES_ERROR


@pytest.mark.parametrize("name", CONVECTION_CASES, ids=CONVECTION_CASES)
def test_convection_same_grid_stencil_and_steps(path, name):
    g = data(name)
    out = result(path, "convection", name)
    for key in CONVECTION_KEYS:
        assert_close(out[key], g[key], rtol=TRAJECTORY_RTOL)
    t = out["temperature"]
    assert_close(t[0], g["input_temperature"][0])
    assert_close(t[-1], g["input_temperature"][-1])
    check_velocity(out)
    check_diagnostics(out)
    if name.startswith("conduction"):
        assert_close(t, g["input_temperature"])
        assert_close(out["nu_top"], np.array(1.0))
        assert_close(out["nu_bottom"], np.array(1.0))
        assert_close(out["vrms"], np.array(0.0))


def test_heat_second_order_spatial_convergence(path):
    errors = []
    for n in (8, 16, 32):
        name = f"diffusion_n{n}"
        g = data(name)
        actual = result(path, "convection", name)["temperature"].astype(np.float64)
        # Normalize by the evolving perturbation, not the O(1) conductive base.
        scale = np.max(np.abs(g["exact_temperature"]-g["base"]))
        errors.append(float(np.max(np.abs(actual-g["exact_temperature"]))/scale))
    assert np.isfinite(errors).all() and min(errors) > 0
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all(orders >= MIN_SPATIAL_ORDER), (errors, orders)
    assert errors[-1] <= FINE_DIFFUSION_ERROR


def test_rk2_second_order_temporal_convergence(path):
    errors = []
    for steps in (10, 20, 40):
        name = f"time_s{steps}"
        g = data(name)
        actual = result(path, "convection", name)["temperature"].astype(np.float64)
        scale = np.max(np.abs(g["exact_temperature"]-g["base"]))
        errors.append(float(np.max(np.abs(actual-g["exact_temperature"]))/scale))
    assert np.isfinite(errors).all() and min(errors) > 0
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all(orders >= MIN_TEMPORAL_ORDER), (errors, orders)


def test_coupled_local_second_order_spatial_convergence(path):
    """Analytic short-time rate isolates advection plus diffusion and Stokes."""
    errors = []
    for n in (8, 16, 32):
        name = f"coupled_refine_n{n}"
        g = data(name)
        actual = result(path, "convection", name)["temperature"].astype(np.float64)
        rate = (actual-g["input_temperature"])/float(g["dt"])
        exact = g["analytic_rate"]
        errors.append(float(np.max(np.abs(rate-exact))/np.max(np.abs(exact))))
    assert np.isfinite(errors).all() and min(errors) > 0
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all(orders >= MIN_SPATIAL_ORDER), (errors, orders)


def test_blankenbach_case1a_published_values_and_steady_state(path):
    g = data("case1a")
    out = result(path, "convection", "case1a")
    for key in ("nu_top", "nu_bottom"):
        assert_close(out[key], g["published_nu"], rtol=BENCHMARK_RTOL)
    assert_close(out["vrms"], g["published_vrms"], rtol=BENCHMARK_RTOL)
    assert abs(float(out["nu_top"])-float(out["nu_bottom"])) / float(g["published_nu"]) <= HEAT_BALANCE_RTOL
    t = out["temperature"].astype(np.float64)
    assert t.min() >= -TEMPERATURE_ATOL
    assert t.max() <= 1+TEMPERATURE_ATOL

    # A stopped transient can accidentally hit one published diagnostic. Demand
    # a small steady PDE residual throughout the interior, including sidewalls.
    n = t.shape[0]-1
    p = np.pad(t, ((0, 0), (1, 1)), mode="reflect")
    tx = (n/2)*(p[1:-1, 2:]-p[1:-1, :-2])
    tz = (n/2)*(t[2:]-t[:-2])
    lap = n*n*(p[1:-1, 2:]+p[1:-1, :-2]+t[2:]+t[:-2]-4*t[1:-1])
    residual = lap-out["u"][1:-1]*tx-out["w"][1:-1]*tz
    diffusion_rms = float(np.sqrt(np.mean(lap*lap)))
    assert diffusion_rms > 0
    relative_rms = float(np.sqrt(np.mean(residual*residual))/diffusion_rms)
    assert relative_rms <= STEADY_RESIDUAL_RTOL, relative_rms


@pytest.mark.parametrize("paths", [("mlx", "metal")], ids=["mlx-metal"])
@pytest.mark.parametrize("name", STOKES_CASES, ids=STOKES_CASES)
def test_cross_stokes(paths, name):
    a = result(paths[0], "stokes", name)
    b = result(paths[1], "stokes", name)
    for key in STOKES_KEYS:
        assert_cross(a[key], b[key], data(name)[key])


@pytest.mark.parametrize("paths", [("mlx", "metal")], ids=["mlx-metal"])
@pytest.mark.parametrize("name", CONVECTION_CASES, ids=CONVECTION_CASES)
def test_cross_convection(paths, name):
    a = result(paths[0], "convection", name)
    b = result(paths[1], "convection", name)
    for key in CONVECTION_KEYS:
        assert_cross(a[key], b[key], data(name)[key])
