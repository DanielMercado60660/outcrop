import numpy as np
import pytest
from outcrop_harness import call, golden, rel_err
from tolerances import MAX_SPREAD_RATIO, RESIDUAL_ABS_SECONDS, SOLUTION_REL


@pytest.fixture(params=["clean", "noisy"], ids=["clean", "noisy"])
def case(request):
    return golden(request.param)


def run(path, case):
    args = {name: case[name].astype(np.float32) for name in
            ("stations", "initial_locations", "initial_origins", "observed")}
    args["pairs"] = case["pairs"].astype(np.int32)
    args["velocity"] = float(case["velocity"])
    args["damping"] = float(case["damping"])
    return call(path, "relocate", **args)


@pytest.mark.parametrize("path", ["mlx"], ids=["mlx"])
def test_reference_solution(path, case):
    out = run(path, case)
    assert set(out) == {"locations", "origins"}
    for name in ("locations", "origins"):
        assert out[name].shape == case[name].shape
        assert out[name].dtype == np.float32
        assert np.isfinite(out[name]).all()
        assert rel_err(out[name], case[name]) <= SOLUTION_REL


@pytest.mark.parametrize("path", ["mlx"], ids=["mlx"])
def test_differential_arrivals(path, case):
    out = run(path, case)
    arrival = out["origins"].astype(np.float64)[:, None] + np.linalg.norm(
        out["locations"].astype(np.float64)[:, None, :] - case["stations"][None, :, :],
        axis=2) / float(case["velocity"])
    pairs = case["pairs"]
    residual = arrival[pairs[:, 0]] - arrival[pairs[:, 1]] - case["observed"]
    assert np.max(np.abs(residual - case["residuals"])) <= RESIDUAL_ABS_SECONDS


@pytest.mark.parametrize("path", ["mlx"], ids=["mlx"])
def test_fault_spread(path, case):
    out = run(path, case)
    before = np.std(case["initial_locations"][:, 1])
    after = np.std(out["locations"][:, 1].astype(np.float64))
    assert after <= MAX_SPREAD_RATIO * before
