"""Matched quadrature parity, independent refinement, and path agreement."""

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import CROSS_REL, FINE_REL, MATCHED_REL

CASES = ("near", "mixed", "distance")
FIELDS = ("potential", "g_z")
RADIAL = (False, True)


def evaluate(path, data, field, radial):
    result = call(path, "tesseroid_gravity", coordinates=data["coordinates"],
                  tesseroids=data["tesseroids"], density=data["density"],
                  field=field, radial_adaptive=radial)
    assert set(result) == {"field"}
    value = result["field"]
    assert value.shape == (len(data["coordinates"]),)
    assert value.dtype == np.float32
    assert np.all(np.isfinite(value))
    return value.astype(np.float64)


def assert_relative_each(actual, expected, tolerance):
    # Each point is measured against its own magnitude, including distant points.
    assert np.all(expected != 0)
    error = np.abs(actual - expected) / np.abs(expected)
    assert np.all(error <= tolerance), f"per-point relative errors: {error}"


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("radial", RADIAL, ids=("horizontal", "radial"))
def test_matched(path, case, field, radial):
    data = golden(case)
    actual = evaluate(path, data, field, radial)
    expected = data[f"{field}_{int(radial)}"]
    if case == "distance":
        print(f"{path} {field} radial={radial} radii_m="
              f"{data['coordinates'][:, 2].tolist()} per_point_relative_error="
              f"{(np.abs(actual - expected) / np.abs(expected)).tolist()}")
    assert_relative_each(actual, expected, MATCHED_REL)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("radial", RADIAL, ids=("horizontal", "radial"))
def test_refined_reference(path, case, field, radial):
    data = golden(case)
    assert_relative_each(evaluate(path, data, field, radial),
                         data[f"{field}_fine"], FINE_REL)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("radial", RADIAL, ids=("horizontal", "radial"))
def test_mlx_metal_crosscheck(case, field, radial):
    data = golden(case)
    mlx = evaluate("mlx", data, field, radial)
    metal = evaluate("metal", data, field, radial)
    assert_relative_each(mlx, metal, CROSS_REL)
