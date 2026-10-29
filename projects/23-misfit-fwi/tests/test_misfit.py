"""Locked MLX objective, adjoint, and local cycle-skipping acceptance tests."""

import numpy as np
import pytest
from outcrop_harness import call, golden, rel_err

from ._local_search import recover
from .tolerances import (
    GRAD_RTOL,
    MIN_BASIN_RATIO,
    MIN_L2_BASIN_SECONDS,
    PARITY_RTOL,
    RECOVERY_SECONDS,
    ZERO_ATOL,
)

# This is an MLX-only project. The shared harness PATHS includes both
# globally; using it unfiltered would add Metal tests that do not apply.
PATHS = ["mlx"]
KINDS = ["l2", "envelope"]
CASES = ["odd", "even", "small"]


def checked(path, predicted, observed, dt, kind, epsilon):
    out = call(path, "misfit_and_grad", predicted=predicted, observed=observed,
               dt=float(dt), kind=kind, epsilon=float(epsilon))
    assert set(out) == {"objective", "gradient"}
    assert out["objective"].shape == ()
    assert out["gradient"].shape == predicted.shape
    for arr in out.values():
        assert arr.dtype == np.dtype("float32")
        assert np.all(np.isfinite(arr))
    assert float(out["objective"]) >= -ZERO_ATOL
    return out


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", CASES)
def test_float64_reference_parity(path, kind, case):
    g = golden(case)
    out = checked(path, g["predicted"], g["observed"], g["dt"], kind, g["epsilon"])
    for key in ("objective", "gradient"):
        assert rel_err(out[key], g[f"{kind}_{key}"]) <= PARITY_RTOL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", CASES)
def test_centered_gradient(path, kind, case):
    g = golden(case)
    out = checked(path, g["predicted"], g["observed"], g["dt"], kind, g["epsilon"])
    adjoint, finite_difference = [], []
    # Per-objective/per-regime steps are saved by the judge. The small envelope
    # residual cancels near epsilon; a 0.002 step was not float32-feasible there.
    h = float(g[f"{kind}_fd_step"])
    for direction in g["directions"]:
        plus = checked(path, (g["predicted"] + h*direction).astype(np.float32),
                       g["observed"], g["dt"], kind, g["epsilon"])
        minus = checked(path, (g["predicted"] - h*direction).astype(np.float32),
                        g["observed"], g["dt"], kind, g["epsilon"])
        adjoint.append(np.sum(out["gradient"].astype(np.float64)*direction))
        finite_difference.append((float(plus["objective"])-float(minus["objective"]))/ (2*h))
    assert rel_err(adjoint, g[f"{kind}_directional"]) <= PARITY_RTOL
    assert rel_err(finite_difference, adjoint) <= GRAD_RTOL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("kind", KINDS)
def test_exact_fit_and_polarity(path, kind):
    g = golden("even")
    x = g["observed"]
    exact = checked(path, x, x, g["dt"], kind, g["epsilon"])
    assert np.max(np.abs(exact["gradient"])) <= ZERO_ATOL
    assert abs(float(exact["objective"])) <= ZERO_ATOL
    flipped = checked(path, -x, x, g["dt"], kind, g["epsilon"])
    if kind == "envelope":
        # Envelope information alone cannot determine source polarity.
        assert abs(float(flipped["objective"])) <= ZERO_ATOL
        assert np.max(np.abs(flipped["gradient"])) <= ZERO_ATOL
    else:
        expected = 2*float(g["dt"])*np.sum(x.astype(float)**2)/len(x)
        assert rel_err(flipped["objective"], expected) <= PARITY_RTOL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", ["zero_prediction", "zero_pair"])
def test_silence_and_smoothing(path, kind, case):
    g = golden("boundary")
    observed = g["nonzero"] if case == "zero_prediction" else g["zero"]
    out = checked(path, g["zero"], observed, g["dt"], kind, g["epsilon"])
    for key in out:
        expected = g[f"{case}_{kind}_{key}"]
        if np.all(expected == 0):
            assert np.max(np.abs(out[key])) <= ZERO_ATOL
        else:
            assert rel_err(out[key], expected) <= PARITY_RTOL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("kind", KINDS)
def test_dt_and_receiver_normalization(path, kind):
    g = golden("odd")
    x, y = g["predicted"], g["observed"]
    original = checked(path, x, y, g["dt"], kind, g["epsilon"])
    doubled_dt = checked(path, x, y, 2*g["dt"], kind, g["epsilon"])
    for key in original:
        assert rel_err(doubled_dt[key], 2*original[key]) <= PARITY_RTOL
    duplicate = checked(path, np.concatenate([x, x]), np.concatenate([y, y]),
                        g["dt"], kind, g["epsilon"])
    assert rel_err(duplicate["objective"], original["objective"]) <= PARITY_RTOL
    assert rel_err(duplicate["gradient"],
                   np.concatenate([original["gradient"], original["gradient"]])/2) <= PARITY_RTOL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("kind", KINDS)
def test_physical_waveform_parity(path, kind):
    g = golden("cycle")
    for row, index in enumerate(g["probe_indices"]):
        out = checked(path, g["traces"][index:index+1], g["observed"],
                      g["dt"], kind, g["epsilon"])
        assert rel_err(out["objective"], g[f"{kind}_probe_objective"][row]) <= PARITY_RTOL
        assert rel_err(out["gradient"], g[f"{kind}_probe_gradient"][row:row+1]) <= PARITY_RTOL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("sign_index", [0, 1], ids=["negative-delay", "positive-delay"])
def test_cycle_skipping_basin_ratio(path, sign_index):
    g = golden("cycle")
    sign = [-1, 1][sign_index]
    basins = {}
    for kind in KINDS:
        def evaluate(trace, kind=kind):
            return checked(path, trace, g["observed"], g["dt"], kind, g["epsilon"])

        final = np.array([recover(g, sign*start, evaluate) for start in g["initial"]])
        success = np.abs(final) <= RECOVERY_SECONDS
        # Explicit successful and cycle-skipped controls, matched to float64.
        np.testing.assert_array_equal(success, g[f"{kind}_success"][sign_index])
        failures = np.flatnonzero(~success)
        count = int(failures[0]) if len(failures) else len(success)
        basins[kind] = float(g["initial"][count-1]) if count else 0.
    assert basins["l2"] >= MIN_L2_BASIN_SECONDS
    assert basins["envelope"] >= MIN_BASIN_RATIO * basins["l2"]
