"""Locked parity checks against saved Harmonica results."""

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import (
    CROSS_FAR_REL,
    CROSS_NEAR_REL,
    FAR_REL,
    NEAR_REL,
    PROJECTION_REL,
    ZERO_ABS_NT,
)

KEYS = ("b_e", "b_n", "b_u", "total_field")
INPUTS = ("coordinates", "prisms", "magnetization", "inducing_field")
NEAR_CASES = ("single", "translated", "multiple", "axis", "zero")


def run(path, case):
    data = golden(case)
    result = call(path, "prism_magnetic", **{k: data[k] for k in INPUTS})
    assert set(result) == set(KEYS)
    for key in KEYS:
        assert result[key].shape == (data["coordinates"].shape[0],)
        assert result[key].dtype == np.float32
        assert np.isfinite(result[key]).all()
    return data, result


def compare(actual, expected, rel):
    # Normalize by the largest value of each field, with an absolute nT floor
    # so sign-changing or exactly zero components are meaningfully checked.
    expected = np.asarray(expected, dtype=np.float64)
    error = np.max(np.abs(np.asarray(actual, dtype=np.float64) - expected))
    assert error <= rel * np.max(np.abs(expected)) + ZERO_ABS_NT, (error, expected)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", NEAR_CASES)
def test_reference_near(path, case):
    data, result = run(path, case)
    for key in KEYS:
        compare(result[key], data[key], NEAR_REL)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("index", [0, 1, 2], ids=["8-widths", "10-widths", "12-widths"])
def test_reference_far_by_distance(path, index):
    data, result = run(path, "far")
    for key in KEYS:
        compare(result[key][[index, index + 3]], data[key][[index, index + 3]], FAR_REL)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", ("single", "multiple", "translated", "far"))
def test_total_field_projection(path, case):
    data, result = run(path, case)
    components = np.column_stack([result[k] for k in KEYS[:3]]).astype(np.float64)
    compare(result["total_field"], components @ data["inducing_field"].astype(np.float64),
            PROJECTION_REL)


@pytest.mark.parametrize("case", NEAR_CASES + ("far",))
def test_cross_mlx_metal(case):
    _, mlx = run("mlx", case)
    _, metal = run("metal", case)
    if case == "far":
        for i in range(3):
            for key in KEYS:
                compare(metal[key][[i, i + 3]], mlx[key][[i, i + 3]], CROSS_FAR_REL)
    else:
        for key in KEYS:
            compare(metal[key], mlx[key], CROSS_NEAR_REL)
