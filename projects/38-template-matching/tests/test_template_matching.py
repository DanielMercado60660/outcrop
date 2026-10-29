"""Metal correlation checks against the ObsPy reference."""
import numpy as np
import pytest
from outcrop_harness import call, golden
from tolerances import CORRELATION_ATOL

CASES = ("multi", "degenerate", "edges")
PATHS = ("metal",)


def run(path, case):
    g = golden(case)
    result = call(path, "template_match", data=g["data"],
                  templates=g["templates"], threshold=float(g["threshold"]))
    assert set(result) == {"correlation", "detections"}
    for key in ("correlation", "detections"):
        assert result[key].shape == g[key].shape
        assert result[key].dtype == np.float32
        assert np.isfinite(result[key]).all()
    return result, g


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_obspy_parity(path, case):
    result, g = run(path, case)
    np.testing.assert_allclose(result["correlation"], g["correlation"],
                               rtol=0, atol=CORRELATION_ATOL)
    np.testing.assert_array_equal(result["detections"], g["detections"])
    np.testing.assert_array_equal(result["detections"],
                                  (result["correlation"] >= float(g["threshold"])).astype(np.float32))
