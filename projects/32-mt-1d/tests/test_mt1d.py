"""Parity with analytic half-spaces and SimPEG layered responses."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import CROSS_RTOL, RESPONSE_RTOL

CASES = ["half_low", "half_high", "conductive_cover", "resistive_cover",
         "three_layers", "four_layers", "uniform_layers"]


def run(path, data):
    return call(path, "mt1d", resistivity=data["resistivity"],
                thickness=data["thickness"], frequency=data["frequency"])


def check_output(out, reference, rtol):
    assert set(out) == {"apparent_resistivity", "phase"}
    for key in ("apparent_resistivity", "phase"):
        actual = out[key]
        assert actual.shape == reference[key].shape
        assert actual.dtype == np.float32
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, reference[key], rtol=rtol, atol=0)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_reference(path, case):
    data = golden(case)
    check_output(run(path, data), data, RESPONSE_RTOL)


@pytest.mark.parametrize("case", CASES)
def test_cross_mlx_metal(case):
    data = golden(case)
    check_output(run("metal", data), run("mlx", data), CROSS_RTOL)
