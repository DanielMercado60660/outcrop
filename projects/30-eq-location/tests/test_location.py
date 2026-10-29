"""Locked judge for both paths. Test inputs come exclusively from saved goldens."""
from functools import cache

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import (
    COVERAGE_MAX,
    COVERAGE_MIN,
    ELLIPSE_CHI2_95,
    FD_STEPS,
    GEIGER_MAX_POSITION_M,
    GEIGER_MAX_TIME_S,
    GRADIENT_RTOL,
    GRID_MAX_CELLS,
    GRID_POSITION_ATOL_M,
    LOSS_ATOL,
    NOISY_TRIALS,
    PARITY_RTOL,
    TIME_ATOL_S,
)

FUNCTIONS = {"grid": "grid_search", "clean": "geiger", "noisy": "geiger",
             "gradient": "misfit_and_grad"}


def arguments(g, name):
    out = {k: np.ascontiguousarray(g[k], dtype=np.float32)
           for k in ("stations", "arrivals", "sigma")}
    out["velocity"] = float(g["velocity"])
    extra = {"grid": ("x", "y", "z"), "clean": ("initial",),
             "noisy": ("initial",), "gradient": ("models",)}[name]
    out.update({k: np.ascontiguousarray(g[k], dtype=np.float32) for k in extra})
    return out


def validate(out, name, batch, stations):
    shapes = {"models": (batch, 4), "objective": (batch,)}
    if name in ("clean", "noisy"):
        shapes.update(covariance=(batch, 4, 4), ellipse=(batch, 2, 2))
    elif name == "gradient":
        shapes = {"predicted": (batch, stations), "objective": (batch,),
                  "gradient": (batch, 4)}
    assert set(out) == set(shapes)
    for key, shape in shapes.items():
        assert isinstance(out[key], np.ndarray)
        assert out[key].shape == shape, key
        assert out[key].dtype == np.float32, key
        assert np.all(np.isfinite(out[key])), key


@cache
def result(path, name):
    g = golden(name)
    out = call(path, FUNCTIONS[name], **arguments(g, name))
    validate(out, name, *g["arrivals"].shape)
    return out


def close_norm(actual, expected):
    assert rel_err(actual, expected) <= PARITY_RTOL


def close_covariance(actual, expected):
    # Normalize each entry by its parameter standard deviations so m^2 cannot
    # hide errors in s^2 or m*s. Zero correlations use the same 1e-4 floor.
    scale = np.sqrt(np.diag(expected))
    assert np.all(scale > 0)
    normalized = (actual.astype(float) - expected) / np.outer(scale, scale)
    assert np.max(np.abs(normalized)) <= PARITY_RTOL


def close_models(actual, expected):
    # Keep seconds separate from metres so location scale cannot hide bad t0.
    close_norm(actual[:, :3], expected[:, :3])
    np.testing.assert_allclose(actual[:, 3], expected[:, 3],
                               rtol=PARITY_RTOL, atol=TIME_ATOL_S)


def close_loss(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=PARITY_RTOL, atol=LOSS_ATOL)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_grid_reference(path):
    g, out = golden("grid"), result(path, "grid")
    np.testing.assert_allclose(out["models"][:, :3], g["models"][:, :3],
                               rtol=0, atol=GRID_POSITION_ATOL_M)
    close_models(out["models"], g["models"])
    close_loss(out["objective"], g["objective"])


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_grid_within_one_cell(path):
    g, out = golden("grid"), result(path, "grid")
    spacing = np.array([g[k][1] - g[k][0] for k in ("x", "y", "z")])
    # Euclidean error in cell units: stricter and unambiguous vs per-axis bounds.
    error = (out["models"][:, :3].astype(float) - g["truth"][:, :3]) / spacing
    assert np.all(np.linalg.norm(error, axis=1) <= GRID_MAX_CELLS)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", ["clean", "noisy"])
def test_geiger_reference(path, name):
    g, out = golden(name), result(path, name)
    close_models(out["models"], g["models"])
    close_loss(out["objective"], g["objective"])
    # Per-event comparisons prevent a large ellipse hiding a bad smaller one.
    for k in ("covariance", "ellipse"):
        for actual, expected in zip(out[k], g[k]):
            close_covariance(actual, expected)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_geiger_noise_free_recovery(path):
    g, out = golden("clean"), result(path, "clean")
    assert np.all(np.linalg.norm(out["models"][:, :3].astype(float)
                                 - g["truth"][:, :3], axis=1) <= GEIGER_MAX_POSITION_M)
    assert np.all(np.abs(out["models"][:, 3].astype(float)
                         - g["truth"][:, 3]) <= GEIGER_MAX_TIME_S)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_200_trial_95_percent_ellipse_coverage(path):
    g, out = golden("noisy"), result(path, "noisy")
    assert len(g["truth"]) == NOISY_TRIALS
    delta = out["models"][:, :2].astype(float) - g["truth"][:, :2]
    ellipse = out["ellipse"].astype(float)
    assert np.all(np.linalg.eigvalsh(ellipse) > 0)
    mahal = np.einsum("bi,bij,bj->b", delta, np.linalg.inv(ellipse), delta)
    coverage = np.mean(mahal <= 1)
    assert COVERAGE_MIN <= coverage <= COVERAGE_MAX, (coverage, mahal)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", ["clean", "noisy"])
def test_uncertainty_definition(path, name):
    out = result(path, name)
    cov, ellipse = out["covariance"].astype(float), out["ellipse"].astype(float)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    for c, e in zip(cov, ellipse):
        close_covariance(c, c.T)
        close_covariance(e, e.T)
        close_covariance(e, ELLIPSE_CHI2_95 * c[:2, :2])


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_objective_and_gradient_reference(path):
    g, out = golden("gradient"), result(path, "gradient")
    close_norm(out["predicted"], g["expected_predicted"])
    close_loss(out["objective"], g["expected_objective"])
    # Each parameter uses its own scale (derivatives have different units).
    for k in range(4):
        close_norm(out["gradient"][:, k], g["expected_gradient"][:, k])


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("component", range(4), ids=["east", "north", "depth", "origin"])
def test_centered_finite_difference_gradient(path, component):
    g = golden("gradient")
    kwargs = arguments(g, "gradient")
    plus, minus = kwargs["models"].copy(), kwargs["models"].copy()
    plus[:, component] += FD_STEPS[component]
    minus[:, component] -= FD_STEPS[component]
    a = call(path, "misfit_and_grad", **dict(kwargs, models=plus))
    b = call(path, "misfit_and_grad", **dict(kwargs, models=minus))
    validate(a, "gradient", *g["arrivals"].shape)
    validate(b, "gradient", *g["arrivals"].shape)
    # Actual representable span, rather than ideal 2h, includes float32 rounding.
    fd = ((a["objective"].astype(float) - b["objective"].astype(float))
          / (plus[:, component].astype(float) - minus[:, component].astype(float)))
    analytic = result(path, "gradient")["gradient"][:, component]
    assert rel_err(analytic, fd) <= GRADIENT_RTOL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_grid_to_geiger_pipeline(path):
    g = golden("clean")
    kwargs = arguments(g, "clean")
    kwargs["initial"] = result(path, "grid")["models"].copy()
    out = call(path, "geiger", **kwargs)
    validate(out, "clean", *g["arrivals"].shape)
    close_models(out["models"], g["models"])
    assert np.all(np.linalg.norm(out["models"][:, :3].astype(float)
                                 - g["truth"][:, :3], axis=1) <= GEIGER_MAX_POSITION_M)
    assert np.all(np.abs(out["models"][:, 3].astype(float)
                         - g["truth"][:, 3]) <= GEIGER_MAX_TIME_S)


@pytest.mark.parametrize("name", ["grid", "clean", "noisy", "gradient"])
def test_cross_mlx_metal(name):
    a, b = result("mlx", name), result("metal", name)
    for key in a:
        if key == "models":
            close_models(a[key], b[key])
        elif key == "objective":
            close_loss(a[key], b[key])
        elif key == "gradient":
            for k in range(4):
                close_norm(a[key][:, k], b[key][:, k])
        elif key in ("covariance", "ellipse"):
            for aa, bb in zip(a[key], b[key]):
                close_covariance(aa, bb)
        else:
            close_norm(a[key], b[key])
