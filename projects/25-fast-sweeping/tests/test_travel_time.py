"""Metal parity with scikit-fmm and first-order analytic convergence."""
from itertools import pairwise

import numpy as np
import pytest
from outcrop_harness import call, golden
from tolerances import MAX_OBSERVED_ORDER, MIN_OBSERVED_ORDER, REFERENCE_REL_MAX

PATHS = ["metal"]
CASES = [f"{medium}_{dim}d_{n}" for dim in (2, 3)
         for medium in ("homogeneous", "gradient", "heterogeneous")
         for n in ((21, 41, 81) if medium != "heterogeneous" else (31,))]
CASES += ["heterogeneous_2d_rect", "heterogeneous_3d_rect"]


def solve(path, data):
    result = call(path, "travel_time", phi=data["phi"].astype(np.float32),
                  speed=data["speed"].astype(np.float32), spacing=float(data["spacing"]))
    assert set(result) == {"travel"}
    travel = result["travel"]
    assert travel.shape == data["speed"].shape
    assert travel.dtype == np.float32
    assert np.all(np.isfinite(travel))
    assert np.all(travel >= 0)
    return travel.astype(np.float64)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_scikit_fmm_parity(path, case):
    data = golden(case)
    actual = solve(path, data)
    reference = data["travel"]
    error = np.max(np.abs(actual - reference)) / np.max(np.abs(reference))
    assert error <= REFERENCE_REL_MAX, (case, error)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("dim", [2, 3], ids=["2d", "3d"])
@pytest.mark.parametrize("medium", ["homogeneous", "gradient"])
def test_first_order_analytic_convergence(path, dim, medium):
    errors = []
    for n in (21, 41, 81):
        data = golden(f"{medium}_{dim}d_{n}")
        actual = solve(path, data)
        mask = data["radius"] >= 0.25
        errors.append(float(np.mean(np.abs(actual[mask] - data["analytic"][mask]))))
    assert errors[0] > errors[1] > errors[2], errors
    for coarse, fine in pairwise(errors):
        order = np.log2(coarse / fine)
        assert MIN_OBSERVED_ORDER <= order <= MAX_OBSERVED_ORDER, (errors, order)
