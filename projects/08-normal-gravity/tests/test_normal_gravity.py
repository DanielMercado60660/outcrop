"""Locked parity tests against float64 Boule WGS84 values."""
import numpy as np
import pytest
from outcrop_harness import call, golden
from tolerances import GRAVITY_RELATIVE

CASES = ("surface", "near_earth", "orbital")
PATHS = ("mlx",)  # MLX-only project.


def evaluate(path, case):
    data = golden(case)
    result = call(path, "normal_gravity", latitude=data["latitude"], height=data["height"])
    assert set(result) == {"gravity"}
    actual = result["gravity"]
    assert actual.shape == data["gravity"].shape
    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    return actual, data["gravity"]


def assert_pointwise_relative(actual, expected, tolerance):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert np.all(expected != 0), "relative criterion requires nonzero references"
    errors = np.abs(actual - expected) / np.abs(expected)
    assert np.max(errors) <= tolerance, f"max pointwise relative error {np.max(errors):.9g} > {tolerance}"


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_boule_parity(path, case):
    actual, reference = evaluate(path, case)
    assert_pointwise_relative(actual, reference, GRAVITY_RELATIVE)
