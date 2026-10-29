"""Analytic and cross-path checks; all input arrays come from goldens."""

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import (
    CROSS_RELATIVE_ERROR,
    CROSS_ZERO_SCALE_ERROR,
    GRADIENT_RELATIVE_ERROR,
    RELATIVE_ERROR,
    ZERO_SCALE_ERROR,
)


def run(path, case):
    inputs = {key: case[key] for key in ("coordinates", "centers", "masses", "radii")}
    result = call(path, "gravity", **inputs)
    assert set(result) == {"potential", "acceleration"}
    for key in result:
        assert result[key].shape == case[key].shape
        assert result[key].dtype == np.float32
        assert np.isfinite(result[key]).all()
    return result


def assert_field(actual, reference, relative, zero_scale):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = np.abs(actual - reference)
    nonzero = reference != 0
    assert np.all(error[nonzero] <= relative * np.abs(reference[nonzero])), (
        f"maximum elementwise relative error: {np.max(error[nonzero] / np.abs(reference[nonzero]))}"
    )
    assert np.all(error[~nonzero] <= zero_scale * np.max(np.abs(reference)))


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", ["point_grid", "sphere", "mixed"])
def test_analytic(path, name):
    case = golden(name)
    result = run(path, case)
    for key in ("potential", "acceleration"):
        assert_field(result[key], case[key], RELATIVE_ERROR, ZERO_SCALE_ERROR)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_centered_potential_gradient(path):
    case = golden("gradient")
    # The potential result has the same expected grid shape, but this golden
    # contains only the independently differenced acceleration reference.
    case["potential"] = np.empty(case["coordinates"].shape[:-1])
    result = run(path, case)
    assert_field(result["acceleration"], case["acceleration"], GRADIENT_RELATIVE_ERROR, ZERO_SCALE_ERROR)


@pytest.mark.parametrize("name", ["point_grid", "sphere", "mixed"])
def test_mlx_metal_cross_check(name):
    case = golden(name)
    mlx = run("mlx", case)
    metal = run("metal", case)
    for key in ("potential", "acceleration"):
        # Use the analytic field as denominator, so near-zero cancellation is
        # judged consistently with the individual implementation checks.
        difference = np.asarray(mlx[key], np.float64) - np.asarray(metal[key], np.float64)
        assert_field(case[key] + difference, case[key], CROSS_RELATIVE_ERROR, CROSS_ZERO_SCALE_ERROR)
