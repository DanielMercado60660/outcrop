"""Analytic plate response and cross-implementation parity."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import CROSS_PATH_RELATIVE_MAX_NORM, RELATIVE_MAX_NORM, ZERO_ABSOLUTE_M

CASES = [(0, "continuous"), (1, "broken"), (2, "continuous"),
         (3, "broken"), (4, "continuous"), (5, "broken")]


def run(path, index, plate):
    data = golden(f"case_{index}")
    out = call(path, "plate_flexure", x=data["x"],
               rigidity=float(data["rigidity"]),
               foundation=float(data["foundation"]),
               load=float(data["load"]), plate=plate)
    assert set(out) == {"deflection"}
    result = out["deflection"]
    assert result.shape == data["x"].shape
    assert result.dtype == np.dtype("float32")
    assert np.all(np.isfinite(result))
    return result, data["deflection"]


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("index,plate", CASES, ids=[f"case_{i}_{p}" for i, p in CASES])
def test_analytic(path, index, plate):
    result, reference = run(path, index, plate)
    if index >= 4:
        assert np.max(np.abs(result)) <= ZERO_ABSOLUTE_M
    else:
        assert rel_err(result, reference) <= RELATIVE_MAX_NORM


@pytest.mark.parametrize("index,plate", CASES, ids=[f"case_{i}_{p}" for i, p in CASES])
def test_mlx_metal_cross_check(index, plate):
    mlx, reference = run("mlx", index, plate)
    metal, _ = run("metal", index, plate)
    if index >= 4:
        assert np.max(np.abs(mlx.astype(np.float64) - metal)) <= ZERO_ABSOLUTE_M
    else:
        assert (np.max(np.abs(mlx.astype(np.float64) - metal)) /
                np.max(np.abs(reference))) <= CROSS_PATH_RELATIVE_MAX_NORM
