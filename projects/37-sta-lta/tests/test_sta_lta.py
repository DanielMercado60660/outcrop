import numpy as np
import pytest
from outcrop_harness import call, golden
from tolerances import CHAR_ATOL, CHAR_RTOL

CASES = ["long", "constant", "impulse", "equal_windows"]
PATHS = ["metal"]  # Metal-only project.


def characteristic(path, data):
    out = call(path, "classic_sta_lta", trace=data["trace"],
               nsta=int(data["nsta"]), nlta=int(data["nlta"]))
    assert set(out) == {"characteristic"}
    value = out["characteristic"]
    assert value.shape == data["trace"].shape
    assert value.dtype == np.float32
    assert np.isfinite(value).all()
    assert np.all(value[:int(data["nlta"]) - 1] == 0)
    return value


def triggers(path, cf, data):
    out = call(path, "trigger_onset", characteristic=cf,
               on=float(data["on"]), off=float(data["off"]))
    assert set(out) == {"triggers"}
    value = out["triggers"]
    assert value.dtype == np.int64
    assert value.ndim == 2 and value.shape[1] == 2
    return value


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_characteristic_and_trigger(path, case):
    data = golden(case)
    actual = characteristic(path, data)
    np.testing.assert_allclose(actual, data["characteristic"], rtol=CHAR_RTOL, atol=CHAR_ATOL)
    np.testing.assert_array_equal(triggers(path, actual, data), data["triggers"])


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_trigger_boundaries(path):
    data = golden("trigger_edges")
    np.testing.assert_array_equal(triggers(path, data["characteristic"], data), data["triggers"])


