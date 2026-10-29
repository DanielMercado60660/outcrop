import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import ANALYTIC_REL, CROSS_PATH_REL, REFERENCE_REL

CASES = ("homogeneous", "block", "layered")
INPUTS = ("hx", "hz", "origin", "conductivity", "ab", "mn", "currents", "ky", "weights")


def run(path, data):
    kwargs = {key: data[key].astype(np.float32) for key in INPUTS}
    out = call(path, "dc_forward", **kwargs)
    assert set(out) == {"voltage", "apparent_resistivity"}
    for value in out.values():
        assert value.shape == data["voltage"].shape
        assert value.dtype == np.float32
        assert np.all(np.isfinite(value))
    return out


def check_relative(actual, expected, tolerance):
    # Per datum, so a large response cannot hide an error in a smaller one.
    assert np.all(np.abs(np.asarray(actual, dtype=np.float64) - expected)
                  <= tolerance * np.abs(expected))


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_reference(path, case):
    data = golden(case)
    out = run(path, data)
    for key in ("voltage", "apparent_resistivity"):
        check_relative(out[key], data[key], REFERENCE_REL)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_analytic_halfspace(path):
    data = golden("homogeneous")
    out = run(path, data)
    check_relative(out["apparent_resistivity"], data["analytic"], ANALYTIC_REL)


@pytest.mark.parametrize("case", CASES)
def test_mlx_metal_crosscheck(case):
    data = golden(case)
    mlx = run("mlx", data)
    metal = run("metal", data)
    for key in ("voltage", "apparent_resistivity"):
        check_relative(mlx[key], metal[key], CROSS_PATH_REL)
