"""MLX parity against frozen Harmonica grids."""
import numpy as np
import pytest
from outcrop_harness import call, golden, rel_err
from tolerances import EDGE_BAND, REFERENCE_RELATIVE

PATHS = ["mlx"]  # MLX-only project.
CASES = ("periodic", "localized", "rough")
OPERATIONS = ("upward", "rtp", "vertical", "tilt")


def run(path, data, operation):
    names = ("dx", "dy", "height", "inclination", "declination",
             "magnetization_inclination", "magnetization_declination")
    args = {key: float(data[key]) for key in names}
    result = call(path, "filter_grid", grid=data["grid"], operation=operation, **args)
    assert set(result) == {"result"}
    value = result["result"]
    assert value.shape == data["grid"].shape
    assert value.dtype == np.float32
    assert np.isfinite(value).all()
    return value


def interior(a):
    return a[EDGE_BAND:-EDGE_BAND, EDGE_BAND:-EDGE_BAND]


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_harmonica(path, case, operation):
    data = golden(case)
    actual = interior(run(path, data, operation))
    expected = interior(data[operation])
    assert rel_err(actual, expected) <= REFERENCE_RELATIVE
