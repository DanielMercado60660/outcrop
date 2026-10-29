"""Finite-fault joint InSAR/GPS inversion. MLX-only project.

Goldens use a published-checklist-validated float64 Okada surface kernel.
All data originate in judge/golden; no implementation-generated synthetic data.
"""
import numpy as np
import pytest
from outcrop_harness import call, golden

from .tolerances import (
    FD_STEP_M,
    GRADIENT_RTOL,
    MAX_MOMENT_REL_ERROR,
    MIN_SLIP_CORRELATION,
    OBJECTIVE_FLOOR,
    OBJECTIVE_RTOL,
    PARITY_RTOL,
)

PATHS = ["mlx"]  # MLX-only project; no Metal contract or cross-check applies.
CASES = ["joint_noiseless", "joint_noisy", "joint_alternate"]


def checked(out, expected):
    assert set(out) == set(expected)
    for key, shape in expected.items():
        assert isinstance(out[key], np.ndarray), key
        assert out[key].shape == shape, key
        assert out[key].dtype == np.float32, key
        assert np.all(np.isfinite(out[key])), key


def parity(actual, expected, tolerance=PARITY_RTOL):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    error = float(np.max(abs(actual - expected)))
    scale = float(np.max(abs(expected)))
    assert error <= tolerance * scale, (error, scale, tolerance)


def args(g, prefix=""):
    # Prefix fixtures explicitly store all perturbed inputs in the goldens.
    return {"green": g[prefix + "green"].astype(np.float32),
            "data": g[prefix + "data"].astype(np.float32),
            "sigma": g.get(prefix + "sigma", g["sigma"]).astype(np.float32),
            "regularizer": g.get(prefix + "regularizer", g["regularizer"]).astype(np.float32),
            "damping": float(g["damping"])}


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_recovery_and_parity(path, case):
    g = golden(case)
    kw = args(g)
    out = call(path, "invert_slip", **kw, areas=g["areas"].astype(np.float32),
               shear_modulus=float(g["shear_modulus"]))
    checked(out, {"slip": (6,), "predicted": (207,), "moment": ()})
    for key in out:
        parity(out[key], g[key])
    # Scientific thresholds compare with truth, independently of the minimizer.
    correlation = np.corrcoef(out["slip"].astype(np.float64), g["true_slip"])[0, 1]
    assert correlation >= MIN_SLIP_CORRELATION
    assert abs(float(out["moment"]) / float(g["true_moment"]) - 1) <= MAX_MOMENT_REL_ERROR
    slip = out["slip"].astype(np.float64)
    parity(out["predicted"], g["green"] @ slip)
    parity(out["moment"], g["shear_modulus"] * (g["areas"] @ abs(slip)))
    # Evaluate the returned solution independently in double precision; fitting
    # or reporting a good slip alone cannot mask an incorrect prediction/J.
    residual = (g["green"] @ slip - g["data"]) / g["sigma"]
    objective = 0.5 * (residual @ residual + float(g["damping"])**2 * np.sum((g["regularizer"] @ slip)**2))
    optimum = float(g["optimum_objective"])
    assert abs(objective-optimum) <= OBJECTIVE_RTOL * max(abs(optimum), OBJECTIVE_FLOOR)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("variant", ["zero", "negative", "permuted", "strong"])
def test_inverse_variants(path, case, variant):
    g = golden(case)
    prefix = variant + "_"
    kw = args(g, prefix if variant == "permuted" else "")
    if variant == "strong":
        kw["damping"] = float(g["strong_damping"])
    else:
        kw["data"] = g[prefix + "data"].astype(np.float32)
    areas = g["permuted_areas"] if variant == "permuted" else g["areas"]
    out = call(path, "invert_slip", **kw, areas=areas.astype(np.float32),
               shear_modulus=float(g["shear_modulus"]))
    checked(out, {"slip": (6,), "predicted": (207,), "moment": ()})
    for key in out:
        parity(out[key], g[prefix + key])


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", ["data_and_penalty", "penalty"])
def test_objective_gradient_parity(path, case, mode):
    g = golden(case)
    prefix = "penalty_" if mode == "penalty" else ""
    out = call(path, "misfit_and_grad", **args(g, prefix), slip=g["trial"].astype(np.float32))
    checked(out, {"objective": (), "gradient": (6,)})
    for key in out:
        parity(out[key], g[prefix + key])


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", ["data_and_penalty", "penalty"])
def test_centered_finite_difference(path, case, mode):
    g = golden(case)
    prefix = "penalty_" if mode == "penalty" else ""
    kw = args(g, prefix)
    slip = g["trial"].astype(np.float32)
    out = call(path, "misfit_and_grad", **kw, slip=slip)
    checked(out, {"objective": (), "gradient": (6,)})
    fd = np.empty(6, dtype=np.float64)
    for j in range(6):
        plus, minus = slip.copy(), slip.copy()
        plus[j] += FD_STEP_M
        minus[j] -= FD_STEP_M
        high = call(path, "misfit_and_grad", **kw, slip=plus)
        low = call(path, "misfit_and_grad", **kw, slip=minus)
        checked(high, {"objective": (), "gradient": (6,)})
        checked(low, {"objective": (), "gradient": (6,)})
        # Actual representable float32 separation; reference step is 0.01 m.
        fd[j] = (float(high["objective"]) - float(low["objective"])) / (float(plus[j])-float(minus[j]))
    parity(out["gradient"], fd, GRADIENT_RTOL)
