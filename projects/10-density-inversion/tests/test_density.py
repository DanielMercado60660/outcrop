"""Locked parity, adjoint, gradient and inversion criteria for both paths."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, golden_meta

from .tolerances import tol

CASES = ["small", "blocks_a", "blocks_b"]
BLOCKS = ["blocks_a", "blocks_b"]


def arr(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def arguments(g):
    return {"sensitivity": arr(g["sensitivity"]), "data": arr(g["data"]),
            "uncertainty": arr(g["uncertainty"]), "reference": arr(g["reference"]),
            "shape": g["shape"].tolist(), "cell_size": float(g["cell_size"]),
            "depth_offset": float(g["depth_offset"]), "alpha_xyz": arr(g["alpha_xyz"]),
            "beta": float(g["beta"])}


def output(out, fields):
    assert set(out) == set(fields)
    for key, shape in fields.items():
        assert isinstance(out[key], np.ndarray)
        assert out[key].shape == shape, key
        assert out[key].dtype == np.float32, key
        assert np.all(np.isfinite(out[key])), key


def close(actual, expected, tolerance=None):
    """Relative L2 parity, with max-scale parity to catch localized errors too."""
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    t = tol("parity") if tolerance is None else tolerance
    assert a.shape == b.shape
    if not np.any(b):
        assert np.array_equal(a, b)
    else:
        assert np.linalg.norm(a-b) <= t*np.linalg.norm(b)
        assert np.max(np.abs(a-b)) <= t*np.max(np.abs(b))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("transpose", [False, True], ids=["forward", "adjoint"])
def test_operator_parity(path, case, transpose):
    g = golden(case)
    expected = g["adjoint" if transpose else "forward"]
    out = call(path, "apply_operator", sensitivity=arr(g["sensitivity"]),
               vector=arr(g["dual" if transpose else "probe"]), transpose=transpose)
    output(out, {"result": expected.shape})
    close(out["result"], expected)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("case", CASES)
def test_adjoint_dot_product(path, case):
    g = golden(case)
    u, v = arr(g["probe"]), arr(g["dual"])
    gu = call(path, "apply_operator", sensitivity=arr(g["sensitivity"]), vector=u,
              transpose=False)["result"].astype(np.float64)
    gtv = call(path, "apply_operator", sensitivity=arr(g["sensitivity"]), vector=v,
               transpose=True)["result"].astype(np.float64)
    scale = np.linalg.norm(gu)*np.linalg.norm(v) + np.linalg.norm(u)*np.linalg.norm(gtv)
    assert abs(gu@v - u@gtv)/scale <= tol("adjoint")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("beta_index", [0, 1, 2, 3], ids=["data_only", "balanced", "strong_reg", "reg_only"])
def test_objective_gradient_parity(path, case, beta_index):
    g = golden(case)
    kw = arguments(g)
    kw.update(model=arr(g["probe"]), reference=arr(g["probe_reference"]),
              beta=float(g["probe_betas"][min(beta_index, 2)]))
    if beta_index == 3:
        kw.update(sensitivity=arr(g["zero_sensitivity"]), data=arr(g["zero_data"]), beta=1.0)
    out = call(path, "misfit_and_grad", **kw)
    output(out, {"objective": (), "phi_d": (), "phi_m": (),
                 "gradient": g["probe"].shape, "weights": g["probe"].shape})
    for key, expected in [("objective", g["phi_m"] if beta_index == 3 else g["objectives"][beta_index]),
                          ("phi_d", np.array(0.0) if beta_index == 3 else g["phi_d"]), ("phi_m", g["phi_m"]),
                          ("gradient", g["reg_gradient"] if beta_index == 3 else g["gradients"][beta_index]), ("weights", g["weights"])]:
        close(out[key], expected)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("beta_index", [0, 1, 2, 3], ids=["data_only", "balanced", "strong_reg", "reg_only"])
def test_centered_gradient(path, beta_index):
    g = golden("small")
    kw = arguments(g)
    kw.update(reference=arr(g["probe_reference"]), beta=float(g["probe_betas"][min(beta_index, 2)]))
    if beta_index == 3:
        kw.update(sensitivity=arr(g["zero_sensitivity"]), data=arr(g["zero_data"]), beta=1.0)
    model, v = arr(g["probe"]), arr(g["direction"])
    h = tol("fd_step")
    base = call(path, "misfit_and_grad", model=model, **kw)
    plus = call(path, "misfit_and_grad", model=arr(model+h*v), **kw)
    minus = call(path, "misfit_and_grad", model=arr(model-h*v), **kw)
    fd = (float(plus["objective"])-float(minus["objective"]))/(2*h)
    dot = base["gradient"].astype(np.float64)@v.astype(np.float64)
    assert np.isfinite(fd) and np.isfinite(dot)
    assert abs(fd-dot)/max(abs(fd), abs(dot)) <= tol("gradient")


def validate_solution(g, out, kw):
    output(out, {"model": g["true_model"].shape, "predicted": g["data"].shape})
    model = out["model"].astype(np.float64)
    matrix = kw["sensitivity"].astype(np.float64)
    predicted = matrix@model
    close(out["predicted"], predicted)
    a = matrix/kw["uncertainty"].astype(np.float64)[:, None]
    b = kw["data"].astype(np.float64)/kw["uncertainty"].astype(np.float64)
    ref = kw["reference"].astype(np.float64)
    residual = a.T@(a@model-b) + kw["beta"]*g["precision"]@(model-ref)
    rhs = a.T@b + kw["beta"]*g["precision"]@ref
    assert np.linalg.norm(residual)/np.linalg.norm(rhs) <= tol("stationarity")
    return model, predicted


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("case", BLOCKS)
def test_inversion_pass_criteria(path, case):
    g = golden(case)
    kw = arguments(g)
    model, predicted = validate_solution(g, call(path, "invert", **kw), kw)
    chi = np.sum(((predicted-g["data"])/g["uncertainty"])**2)
    assert abs(chi/float(g["target"])-1) <= tol("chi_fraction")
    assert np.corrcoef(model, g["true_model"])[0, 1] >= tol("correlation")
    assert np.linalg.norm(model-g["true_model"]) <= (1+tol("model_error_fraction"))*float(g["error"])


@pytest.mark.parametrize("path", PATHS)
def test_invert_nonzero_reference(path):
    g = golden("small")
    kw = arguments(g)
    kw["reference"] = arr(g["probe_reference"])
    validate_solution(g, call(path, "invert", **kw), kw)


@pytest.mark.parametrize("case", CASES)
def test_mlx_metal_cross_parity(case):
    g = golden(case)
    kw = arguments(g)
    # Unique regularized minimizer plus objective, gradient and both operators.
    calls = [("invert", kw), ("misfit_and_grad", dict(kw, model=arr(g["probe"])))]
    for transpose in [False, True]:
        calls.append(("apply_operator", {"sensitivity": kw["sensitivity"], "transpose": transpose,
                                        "vector": arr(g["dual" if transpose else "probe"])}))
    for fn, args in calls:
        left, right = call("mlx", fn, **args), call("metal", fn, **args)
        assert set(left) == set(right)
        for key in left:
            close(left[key], right[key])


@pytest.mark.parametrize("case", CASES)
def test_golden_integrity(case):
    g = golden(case)
    assert golden_meta()["tolerances"]["parity"]["value"] == tol("parity")
    close(g["predicted"], g["sensitivity"]@g["model"], tol("oracle_check"))
    close(g["clean"], g["sensitivity"]@g["true_model"], tol("oracle_check"))
    close(np.sum(((g["predicted"]-g["data"])/g["uncertainty"])**2),
          g["target"], tol("oracle_check"))
    assert g["sensitivity"].dtype == np.float64
    assert np.all(g["sensitivity"] < 0)  # upward gz over positive buried density
    assert np.all(g["uncertainty"] > 0)
