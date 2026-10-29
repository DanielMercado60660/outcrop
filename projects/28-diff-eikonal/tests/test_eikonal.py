"""Locked Tag-B parity and centered-FD tests; no reference code is imported."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err

from .tolerances import FD_REL, FD_STEPS, GRADIENT_COSINE_MIN, PARITY_REL, ZERO_ATOL

CASES = ["homogeneous", "smooth_multishot", "layered"]
FIELDS = ("traveltime", "predicted", "objective", "gradient")


def arguments(g):
    return {k: g[k].copy() for k in ("slowness", "sources", "receivers", "observed", "weights")} | {"spacing": g["spacing"].tolist()}


def checked(path, args):
    before = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    out = call(path, "misfit_and_grad", **args)
    ns, nr = args["observed"].shape
    shape = args["slowness"].shape
    shapes = {"traveltime": (ns, *shape), "predicted": (ns, nr), "objective": (), "gradient": shape}
    assert set(out) == set(FIELDS)
    for key in FIELDS:
        assert out[key].shape == shapes[key], (key, out[key].shape)
        assert out[key].dtype == np.float32, (key, out[key].dtype)
        assert np.isfinite(out[key]).all(), key
    for k, v in before.items():
        np.testing.assert_array_equal(args[k], v, err_msg=f"mutated input {k}")
    assert np.all(out["traveltime"] >= ZERO_ATOL)
    assert out["objective"] >= ZERO_ATOL
    for shot, (z, x) in enumerate(args["sources"]):
        assert out["traveltime"][shot, z, x] == ZERO_ATOL
    np.testing.assert_array_equal(out["predicted"], out["traveltime"][:, args["receivers"][:, 0], args["receivers"][:, 1]])
    return out


def parity(actual, expected):
    for key in FIELDS:
        assert rel_err(actual[key], expected[key]) <= PARITY_REL, key


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_adjoint_reference(path, case):
    g = golden(case)
    out = checked(path, arguments(g))
    parity(out, g)
    a, b = out["gradient"].astype(np.float64).ravel(), g["gradient"].ravel()
    cosine = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
    assert cosine >= GRADIENT_COSINE_MIN
    # Independent scalar definition catches objective normalization/weight errors.
    residual = out["predicted"].astype(np.float64) - g["observed"]
    expected_loss = 0.5*np.sum(g["weights"]*residual**2)
    assert rel_err(out["objective"], expected_loss) <= PARITY_REL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("probe", range(5), ids=["dense0", "dense1", "node0", "node1", "node2"])
def test_centered_finite_difference(path, case, probe):
    g = golden(case)
    args = arguments(g)
    base = checked(path, args)
    args["slowness"] = g["fd_plus"][probe].copy()
    plus = checked(path, args)
    args["slowness"] = g["fd_minus"][probe].copy()
    minus = checked(path, args)
    fd = (float(plus["objective"]) - float(minus["objective"])) / (2*FD_STEPS[probe])
    dot = np.sum(base["gradient"].astype(np.float64)*g["directions"][probe])
    assert abs(fd-dot)/max(abs(fd), abs(dot)) <= FD_REL
    reference = g["fd_reference"][probe]
    assert abs(fd-reference)/abs(reference) <= FD_REL


@pytest.mark.parametrize("case", CASES)
def test_cross_mlx_metal(case):
    g = golden(case)
    parity(checked("mlx", arguments(g)), checked("metal", arguments(g)))


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_zero_residual(path, case):
    g = golden(case)
    args = arguments(g)
    out = checked(path, args)
    args["observed"] = out["predicted"].copy()
    zero = checked(path, args)
    # The exact-zero contract requires repeatable forward solves even when
    # observations change; explicitly expose nondeterministic sweep updates.
    np.testing.assert_array_equal(zero["traveltime"], out["traveltime"])
    np.testing.assert_array_equal(zero["predicted"], out["predicted"])
    assert zero["objective"] == ZERO_ATOL
    assert np.max(np.abs(zero["gradient"])) == ZERO_ATOL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_zero_weights(path, case):
    g = golden(case)
    args = arguments(g)
    args["weights"] = np.zeros_like(args["weights"])
    out = checked(path, args)
    assert out["objective"] == ZERO_ATOL
    assert np.max(np.abs(out["gradient"])) == ZERO_ATOL
    assert rel_err(out["traveltime"], g["traveltime"]) <= PARITY_REL


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", CASES)
def test_weight_scaling(path, case):
    g = golden(case)
    args = arguments(g)
    args["weights"] *= 2
    out = checked(path, args)
    assert rel_err(out["objective"], 2*g["objective"]) <= PARITY_REL
    assert rel_err(out["gradient"], 2*g["gradient"]) <= PARITY_REL
    assert rel_err(out["traveltime"], g["traveltime"]) <= PARITY_REL
