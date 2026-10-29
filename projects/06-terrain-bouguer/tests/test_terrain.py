import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import ABS_MGAL, ALGEBRA_ATOL, FAR_RTOL, NEAR_RTOL, SLAB_RTOL

INPUTS = ("easting", "northing", "surface", "reference", "density", "coordinates")
OUTPUTS = ("layer_gz", "slab_gz", "terrain_gz", "correction_gz")


def run(path, data):
    args = {key: data[key].astype(np.float32) for key in INPUTS}
    out = call(path, "terrain_bouguer", **args)
    assert set(out) == set(OUTPUTS)
    for value in out.values():
        assert value.shape == (len(data["coordinates"]),)
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
    return out


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", ["terrain", "depression", "flat"])
def test_reference(path, case):
    data = golden(case)
    out = run(path, data)
    for key in ("layer_gz", "terrain_gz", "correction_gz"):
        np.testing.assert_allclose(out[key][:4], data[key][:4], rtol=NEAR_RTOL, atol=ABS_MGAL)
        np.testing.assert_allclose(out[key][4:], data[key][4:], rtol=FAR_RTOL, atol=ABS_MGAL)
    np.testing.assert_allclose(out["slab_gz"], data["slab_gz"], rtol=SLAB_RTOL, atol=ALGEBRA_ATOL)
    np.testing.assert_allclose(out["terrain_gz"], out["layer_gz"] - out["slab_gz"], rtol=0, atol=ALGEBRA_ATOL)
    np.testing.assert_allclose(out["correction_gz"], -out["layer_gz"], rtol=0, atol=ALGEBRA_ATOL)


@pytest.mark.parametrize("case", ["terrain", "depression", "flat"])
def test_mlx_metal_crosscheck(case):
    data = golden(case)
    a, b = run("mlx", data), run("metal", data)
    for key in OUTPUTS:
        np.testing.assert_allclose(a[key][:4], b[key][:4], rtol=NEAR_RTOL, atol=ABS_MGAL)
        np.testing.assert_allclose(a[key][4:], b[key][4:], rtol=FAR_RTOL, atol=ABS_MGAL)
