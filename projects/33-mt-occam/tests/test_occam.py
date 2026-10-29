"""Locked tests for the MLX-only project; no test imports or evaluates a reference solver."""
import numpy as np
import pytest
from outcrop_harness import call, golden
from tolerances import (
    FORWARD_ATOL,
    FORWARD_RTOL,
    GRADIENT_ATOL,
    GRADIENT_RTOL,
    MAX_ITERATIONS,
    MISFIT_SLACK,
    MODEL_LOWER,
    MODEL_MAX_ERROR,
    MODEL_UPPER,
    OBJECTIVE_ATOL,
    OBJECTIVE_RTOL,
    ROUGHNESS_ATOL,
    ROUGHNESS_RTOL,
    TARGET_RMS,
)

PATHS = ["mlx"]  # MLX-only project. No Metal or cross-path tests.
FORWARD_CASES = ["halfspace", "two_layer", "thin_conductor", "thick_resistor"]
INVERSION_CASES = ["conductor", "resistor", "constant"]


def inputs(g, names):
    return {name: np.ascontiguousarray(g[name], dtype=np.float32) for name in names}


def checked(out, shapes):
    for name, shape in shapes.items():
        assert name in out, f"missing output {name}"
        a = out[name]
        assert isinstance(a, np.ndarray)
        assert a.shape == shape, (name, a.shape, shape)
        assert a.dtype == np.float32, (name, a.dtype)
        assert np.all(np.isfinite(a)), name


def objective_inputs(g):
    kwargs = inputs(g, ["model", "thickness", "frequencies", "observed", "sigma"])
    kwargs["alpha"] = float(g["alpha"])
    return kwargs


def gradient_close(actual, expected):
    error = np.linalg.norm(np.asarray(actual, np.float64)-expected)
    assert error <= GRADIENT_ATOL + GRADIENT_RTOL * np.linalg.norm(expected), (
        actual, expected, error)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", FORWARD_CASES)
def test_forward_float64_parity(path, case):
    g = golden(f"forward_{case}")
    out = call(path, "forward_mt", **inputs(g, ["model", "thickness", "frequencies"]))
    checked(out, {"response": g["response"].shape})
    np.testing.assert_allclose(out["response"], g["response"],
                               rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", ["0_0", "0_7", "stationary"])
def test_objective_gradient_float64_parity(path, case):
    g = golden(case if case == "stationary" else f"gradient_{case}")
    kw = objective_inputs(g)
    out = call(path, "objective_and_grad", **kw)
    checked(out, {"objective": (), "gradient": g["model"].shape})
    np.testing.assert_allclose(out["objective"], g["objective"],
                               rtol=OBJECTIVE_RTOL, atol=OBJECTIVE_ATOL)
    gradient_close(out["gradient"], g["gradient"])
    # Test consistency between public forward and objective APIs as well.
    prediction = call(path, "forward_mt", **{k: kw[k] for k in
                      ("model", "thickness", "frequencies")})
    checked(prediction, {"response": g["observed"].shape})
    r = (prediction["response"].astype(np.float64)-g["observed"])/g["sigma"]
    phi = .5*np.mean(r*r) + .5*kw["alpha"]*np.sum(np.diff(g["model"])**2)
    np.testing.assert_allclose(out["objective"], phi,
                               rtol=OBJECTIVE_RTOL, atol=OBJECTIVE_ATOL)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", ["0_0", "0_7"])
def test_centered_finite_difference_gradient(path, case):
    g = golden(f"gradient_{case}")
    kw = objective_inputs(g)
    out = call(path, "objective_and_grad", **kw)
    checked(out, {"objective": (), "gradient": g["model"].shape})
    plus, minus = [], []
    for index in range(len(g["model"])):
        for models, expected, values in (
            (g["plus_models"], g["plus_objectives"], plus),
            (g["minus_models"], g["minus_objectives"], minus),
        ):
            perturbed = call(path, "objective_and_grad", **dict(
                kw, model=np.ascontiguousarray(models[index], dtype=np.float32)))
            checked(perturbed, {"objective": (), "gradient": g["model"].shape})
            np.testing.assert_allclose(perturbed["objective"], expected[index],
                                       rtol=OBJECTIVE_RTOL, atol=OBJECTIVE_ATOL)
            values.append(float(perturbed["objective"]))
    # Actual representable step sizes are saved; no randomness or regenerated data.
    fd = (np.array(plus)-minus)/np.diag(g["plus_models"]-g["minus_models"])
    gradient_close(out["gradient"], fd)
    # Also compare to float64 centered differences so correlated errors cannot
    # make a wrong implementation objective and gradient agree with each other.
    oracle_fd = (g["plus_objectives"]-g["minus_objectives"])/np.diag(
        g["plus_models"]-g["minus_models"])
    gradient_close(out["gradient"], oracle_fd)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", INVERSION_CASES)
def test_occam_target_recovery_and_smoothness(path, case):
    g = golden(f"inversion_{case}")
    kw = inputs(g, ["initial_model", "thickness", "frequencies", "observed", "sigma"])
    result = call(path, "invert_occam", **kw, target_rms=TARGET_RMS,
                  max_iterations=MAX_ITERATIONS)
    checked(result, {"model": g["truth"].shape, "response": g["observed"].shape,
                     "rms": (), "roughness": ()})
    m = result["model"].astype(np.float64)
    assert np.all((m >= MODEL_LOWER) & (m <= MODEL_UPPER))
    prediction = call(path, "forward_mt", model=result["model"],
                      thickness=kw["thickness"], frequencies=kw["frequencies"])
    checked(prediction, {"response": g["observed"].shape})
    np.testing.assert_allclose(result["response"], prediction["response"],
                               rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    # Float64 arithmetic here only measures errors in the returned float32 arrays.
    r = (prediction["response"].astype(np.float64)-g["observed"])/g["sigma"]
    rms = np.sqrt(np.mean(r*r))
    assert rms <= TARGET_RMS + MISFIT_SLACK, rms
    np.testing.assert_allclose(result["rms"], rms,
                               rtol=OBJECTIVE_RTOL, atol=OBJECTIVE_ATOL)
    error = np.max(abs(m[g["resolved"]]-g["truth"][g["resolved"]]))
    assert error <= MODEL_MAX_ERROR, error
    rough = .5*np.sum(np.diff(m)**2)
    np.testing.assert_allclose(result["roughness"], rough,
                               rtol=OBJECTIVE_RTOL, atol=OBJECTIVE_ATOL)
    assert rough <= (1+ROUGHNESS_RTOL)*float(g["roughness"])+ROUGHNESS_ATOL, rough
