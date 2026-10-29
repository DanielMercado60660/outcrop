"""Locked parity and analytic checks for both execution paths."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import (
    CROSS_REL,
    MIN_COARSE_ERROR,
    ORDER2_SKFMM_SANITY_REL,
    PARITY_REL,
    REFINEMENT_RATIO,
)

CASES = ([f"heterogeneous_{d}d" for d in (2, 3)]
         + [f"homogeneous_{d}d" for d in (2, 3)]
         + [f"{kind}_{d}d_{n}" for kind in ("gradient", "radial")
            for d in (2, 3) for n in (20, 40, 80)])


def solve(path, data, order):
    result = call(path, "travel_time", phi=data["phi"], speed=data["speed"],
                  spacing=data["spacing"].tolist(), order=order)
    assert set(result) == {"time"}
    time = result["time"]
    assert time.shape == data["phi"].shape
    assert time.dtype == np.float32
    assert np.isfinite(time).all()
    assert (time >= 0).all()
    assert (time[data["phi"] == 0] == 0).all()
    return time


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("order", [1, 2], ids=lambda order: f"order{order}")
def test_reference_parity(path, name, order):
    data = golden(name)
    time = solve(path, data, order)
    assert rel_err(time, data[f"time_order{order}"]) <= PARITY_REL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", CASES)
def test_order2_skfmm_sanity(path, name):
    data = golden(name)
    assert rel_err(solve(path, data, 2), data["time_skfmm_order2"]) <= ORDER2_SKFMM_SANITY_REL


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("order", [1, 2], ids=lambda order: f"order{order}")
def test_mlx_metal_crosscheck(name, order):
    data = golden(name)
    mlx = solve("mlx", data, order)
    metal = solve("metal", data, order)
    assert rel_err(mlx, metal) <= CROSS_REL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("dim", [2, 3], ids=lambda dim: f"{dim}d")
@pytest.mark.parametrize("order", [1, 2], ids=lambda order: f"order{order}")
def test_homogeneous_planar_analytic(path, dim, order):
    data = golden(f"homogeneous_{dim}d")
    assert rel_err(solve(path, data, order), data["analytic"]) <= PARITY_REL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("dim", [2, 3], ids=lambda dim: f"{dim}d")
@pytest.mark.parametrize("kind", ["gradient", "radial"])
def test_first_order_analytic_convergence(path, dim, kind):
    errors = []
    for n in (20, 40, 80):
        data = golden(f"{kind}_{dim}d_{n}")
        time = solve(path, data, 1).astype(np.float64)
        mask = data.get("mask", np.ones(time.shape, dtype=bool))
        errors.append(float(np.max(np.abs(time[mask] - data["analytic"][mask]))))
    assert errors[0] >= MIN_COARSE_ERROR
    assert errors[1] / errors[0] <= REFINEMENT_RATIO, errors
    assert errors[2] / errors[1] <= REFINEMENT_RATIO, errors
