"""Locked float32 parity against independent float64 gravity references."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import (
    CROSS_PATH_RELATIVE_ERROR,
    RELATIVE_ERROR,
    ZERO_ABSOLUTE_ERROR_MGAL,
)

CASES = ("rectangle", "polygon", "polygon_reversed", "polygon_translated", "concave", "cylinder", "slab")


def evaluate(path, data):
    result = call(path, "polygon_gravity", vertices=data["vertices"],
                  observations=data["observations"], density=float(data["density"]))
    assert set(result) == {"gz"}
    value = result["gz"]
    assert value.shape == data["gz"].shape
    assert value.dtype == np.float32
    assert np.all(np.isfinite(value))
    return value


def assert_scaled_error(actual, reference, limit):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = np.max(np.abs(actual - reference)) / np.max(np.abs(reference))
    assert error <= limit, f"maximum scale-normalized error {error:.9g} exceeds {limit}"


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_reference(path, case):
    data = golden(case)
    assert_scaled_error(evaluate(path, data), data["gz"], RELATIVE_ERROR)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_zero_density(path):
    data = golden("zero_density")
    assert np.max(np.abs(evaluate(path, data))) <= ZERO_ABSOLUTE_ERROR_MGAL


@pytest.mark.parametrize("case", CASES)
def test_mlx_metal_crosscheck(case):
    data = golden(case)
    mlx = evaluate("mlx", data)
    metal = evaluate("metal", data)
    # Normalize by the independent float64 reference, not either implementation.
    error = np.max(np.abs(mlx.astype(np.float64) - metal.astype(np.float64)))
    assert error / np.max(np.abs(data["gz"])) <= CROSS_PATH_RELATIVE_ERROR
