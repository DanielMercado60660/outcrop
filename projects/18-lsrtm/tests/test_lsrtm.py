"""Parity, inverse-problem quality, and derivative checks on project 17 data.

No reference imports or generated random inputs. Width and residual criteria
are evaluated from returned images and independently re-modeled predictions,
never accepted solely from an implementation's reported metrics.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, golden_meta

from . import tolerances as tol
from .metrics import WIDTH_DEFINITION, reflector_metrics


@pytest.fixture(scope="module", params=PATHS, ids=PATHS)
def path(request):
    return request.param


@pytest.fixture(scope="module")
def layered():
    return golden("layered")


@pytest.fixture(scope="module")
def small():
    return golden("small")


def kwargs(g):
    """Contract input conversion only; all survey data come from saved goldens."""
    return {"v": g["v"].astype(np.float32), "mask": g["mask"].astype(np.float32),
            "source_amplitudes": g["source_amplitudes"].astype(np.float32),
            "source_locations": g["source_locations"], "receiver_locations": g["receiver_locations"],
            "spacing": g["spacing"].tolist(), "dt": float(g["dt"]),
            "pml_width": int(g["pml_width"]), "pml_freq": float(g["pml_freq"]),
            "accuracy": int(g["accuracy"])}


def check_output(out, shapes):
    assert set(out) == set(shapes), f"output keys {set(out)} != {set(shapes)}"
    for key, shape in shapes.items():
        a = out[key]
        assert isinstance(a, np.ndarray)
        assert a.shape == tuple(shape), (key, a.shape, shape)
        assert a.dtype == np.float32, (key, a.dtype)
        assert np.all(np.isfinite(a)), f"nonfinite {key}"


def close(actual, expected, rtol=tol.PARITY_RTOL, scale_reference=None):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert a.shape == b.shape
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    reference = b if scale_reference is None else np.asarray(scale_reference, np.float64)
    norm = float(np.max(np.abs(reference)))
    err = float(np.max(np.abs(a - b)))
    if norm == 0.0:
        assert err == 0.0
    else:
        assert err / norm <= rtol, f"relative max error {err / norm:.8g} > {rtol}"


def objective_call(path, g, model=None, observed=None):
    return call(path, "misfit_and_grad", **kwargs(g),
                model=g["model"].astype(np.float32) if model is None else model,
                observed=g["observed"].astype(np.float32) if observed is None else observed)


def solve_call(path, g, iterations=None, observed=None):
    return call(path, "lsrtm", **kwargs(g),
                iterations=int(g["iterations"]) if iterations is None else iterations,
                observed=g["observed"].astype(np.float32) if observed is None else observed)


@pytest.fixture(scope="module")
def solution(path, layered):
    out = solve_call(path, layered)
    check_output(out, {"image": layered["v"].shape, "rtm": layered["v"].shape,
                       "predicted": layered["observed"].shape,
                       "residual_history": (int(layered["iterations"]) + 1,)})
    return out


@pytest.fixture(scope="module")
def small_objective(path, small):
    out = objective_call(path, small)
    check_output(out, {"objective": (), "gradient": small["v"].shape,
                       "predicted": small["observed"].shape})
    return out


@pytest.mark.parametrize("case", ["layered", "small"])
def test_born_reference(path, case):
    g = golden(case)
    out = call(path, "born_model", **kwargs(g), model=g["truth"].astype(np.float32))
    check_output(out, {"predicted": g["observed"].shape})
    close(out["predicted"], g["observed"])


def test_objective_gradient_reference(small, small_objective):
    for key in ("objective", "gradient", "predicted"):
        close(small_objective[key], small[key])
    assert np.all(small_objective["gradient"][small["mask"] == 0.0] == 0.0)
    residual = small_objective["predicted"].astype(np.float64) - small["observed"]
    close(small_objective["objective"], 0.5 * np.sum(residual**2))


@pytest.mark.parametrize("case", ["layered", "small"])
def test_zero_model_is_negative_rtm_gradient(path, case):
    g = golden(case)
    out = objective_call(path, g, model=g["zero_model"])
    check_output(out, {"objective": (), "gradient": g["v"].shape,
                       "predicted": g["observed"].shape})
    close(out["gradient"], -g["rtm"])
    close(out["objective"], g["zero_objective"])
    assert np.all(out["predicted"] == 0.0)


def test_mask_is_part_of_operator_and_gradient(path, small, small_objective):
    out = objective_call(path, small, model=small["model_outside"])
    check_output(out, {"objective": (), "gradient": small["v"].shape,
                       "predicted": small["observed"].shape})
    for key in ("objective", "gradient", "predicted"):
        close(out[key], small_objective[key])
    assert np.all(out["gradient"][small["mask"] == 0.0] == 0.0)


def test_adjoint_dot_product(path, small):
    out = objective_call(path, small, model=small["zero_model"], observed=small["probe_data"])
    check_output(out, {"objective": (), "gradient": small["v"].shape,
                       "predicted": small["observed"].shape})
    at_probe = -out["gradient"].astype(np.float64)
    close(at_probe, small["adjoint"])
    am = call(path, "born_model", **kwargs(small), model=small["model"])
    check_output(am, {"predicted": small["observed"].shape})
    lhs = np.sum(am["predicted"].astype(np.float64) * small["probe_data"].astype(np.float64))
    rhs = np.sum(small["model"].astype(np.float64) * at_probe)
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs)) <= tol.PARITY_RTOL


@pytest.mark.parametrize("direction", range(3), ids=["dense", "coordinate-a", "coordinate-b"])
def test_centered_finite_difference_gradient(path, small, small_objective, direction):
    # Perturbed models are saved by the float64 judge, rounded to public float32.
    plus = objective_call(path, small, model=small["fd_models_plus"][direction])
    minus = objective_call(path, small, model=small["fd_models_minus"][direction])
    for out in (plus, minus):
        check_output(out, {"objective": (), "gradient": small["v"].shape,
                           "predicted": small["observed"].shape})
    fd = (float(plus["objective"]) - float(minus["objective"])) / (2.0 * tol.FD_STEP)
    d = small["directions"][direction].astype(np.float64)
    dot = float(np.sum(small_objective["gradient"].astype(np.float64) * d))
    # Check against both the implementation gradient and the float64 FD oracle.
    assert abs(fd - dot) / max(abs(fd), abs(dot)) <= tol.GRADIENT_RTOL
    ref = float(small["fd_values"][direction])
    assert abs(fd - ref) / abs(ref) <= tol.GRADIENT_RTOL


def test_cgls_reference(layered, solution):
    for key in ("image", "rtm", "predicted", "residual_history"):
        close(solution[key], layered[key])
    assert np.all(solution["image"][layered["mask"] == 0.0] == 0.0)
    assert np.all(solution["rtm"][layered["mask"] == 0.0] == 0.0)
    history = solution["residual_history"].astype(np.float64)
    assert history[0] == 1.0
    assert np.all(history >= 0.0)
    assert np.all(np.diff(history) <= tol.HISTORY_SLACK)


def test_residual_reduced_fifty_percent(path, layered, solution):
    # Re-model the returned image, so fabricated reported history cannot pass.
    pred = call(path, "born_model", **kwargs(layered), model=solution["image"])
    check_output(pred, {"predicted": layered["observed"].shape})
    close(pred["predicted"], solution["predicted"])
    observed = layered["observed"].astype(np.float64)
    residual = np.linalg.norm(pred["predicted"].astype(np.float64) - observed)
    ratio = residual / np.linalg.norm(observed)
    close(np.asarray(ratio), solution["residual_history"][-1].astype(np.float64))
    assert ratio <= tol.MAX_RESIDUAL_RATIO
    # RTM amplitude is arbitrary: give it its optimal scalar least-squares fit.
    # Re-model the implementation RTM rather than accepting a reported baseline.
    base = call(path, "born_model", **kwargs(layered), model=solution["rtm"])
    check_output(base, {"predicted": layered["observed"].shape})
    u = base["predicted"].astype(np.float64)
    alpha = np.sum(u * observed) / np.sum(u * u)
    baseline = np.linalg.norm(alpha * u - observed)
    assert residual <= tol.MAX_RESIDUAL_RATIO * baseline


@pytest.mark.parametrize("reflector", range(3), ids=["shallow", "middle", "deep"])
@pytest.mark.parametrize("column", range(3), ids=["x240m", "x360m", "x480m"])
def test_each_reflector_narrower_twenty_percent(layered, solution, reflector, column):
    rtm_depth, rtm_width = reflector_metrics(solution["rtm"], layered["columns"], layered["windows"])
    ls_depth, ls_width = reflector_metrics(solution["image"], layered["columns"], layered["windows"])
    truth = layered["reflector_depths"][reflector]
    assert abs(rtm_depth[reflector, column] - truth) <= tol.MAX_DEPTH_ERROR_CELLS
    assert abs(ls_depth[reflector, column] - truth) <= tol.MAX_DEPTH_ERROR_CELLS
    assert ls_width[reflector, column] <= tol.MAX_WIDTH_RATIO * rtm_width[reflector, column]
    # Also anchor the comparator to the independent reference RTM width.
    assert ls_width[reflector, column] <= tol.MAX_WIDTH_RATIO * layered["rtm_widths"][reflector, column]


@pytest.mark.parametrize("iterations", [0, 1], ids=["zero-steps", "one-step"])
def test_solver_start_and_best_scaled_rtm(path, layered, iterations):
    out = solve_call(path, layered, iterations=iterations)
    check_output(out, {"image": layered["v"].shape, "rtm": layered["v"].shape,
                       "predicted": layered["observed"].shape,
                       "residual_history": (iterations + 1,)})
    close(out["rtm"], layered["rtm"])
    if iterations == 0:
        assert np.all(out["image"] == 0.0)
        assert np.all(out["predicted"] == 0.0)
    else:
        close(out["image"], layered["rtm_scaled"])
        close(out["predicted"], layered["rtm_scaled_predicted"])
    close(out["residual_history"], layered["residual_history"][:iterations + 1])


def test_zero_data(path, small):
    out = solve_call(path, small, iterations=3, observed=small["zero_observed"])
    check_output(out, {"image": small["v"].shape, "rtm": small["v"].shape,
                       "predicted": small["observed"].shape, "residual_history": (4,)})
    for value in out.values():
        assert np.all(value == 0.0)


def test_changed_data_polarity_and_scale(path, layered):
    out = solve_call(path, layered, observed=layered["scaled_observed"])
    check_output(out, {"image": layered["v"].shape, "rtm": layered["v"].shape,
                       "predicted": layered["observed"].shape,
                       "residual_history": (int(layered["iterations"]) + 1,)})
    for key in ("image", "rtm", "predicted"):
        close(out[key], layered[f"scaled_{key}"])
    close(out["residual_history"], layered["scaled_residual_history"])


@pytest.mark.parametrize("fn", ["born_model", "misfit_and_grad"],
                         ids=["mlx-metal-born", "mlx-metal-objective-gradient"])
def test_cross_path_operator(small, fn):
    args = dict(kwargs(small), model=small["model"])
    if fn == "misfit_and_grad":
        args["observed"] = small["observed"].astype(np.float32)
    mlx = call("mlx", fn, **args)
    metal = call("metal", fn, **args)
    shapes = {"predicted": small["observed"].shape}
    if fn == "misfit_and_grad":
        shapes.update(objective=(), gradient=small["v"].shape)
    for out in (mlx, metal):
        check_output(out, shapes)
    for key in shapes:
        close(mlx[key], metal[key], tol.CROSS_RTOL, scale_reference=small[key])


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_golden_integrity_and_tolerance_manifest(path):
    # No modeling here: detect stale or accidentally edited golden bundles.
    meta = golden_meta()
    assert meta["width_metric"] == WIDTH_DEFINITION
    scheme = meta["scheme"]
    assert scheme["specification"] == "tests/SCHEME.md"
    assert scheme["reference"] == "judge/scheme_reference.py"
    assert set(scheme["sha256"]) == {"tests/SCHEME.md", "judge/scheme_reference.py"}
    root = Path(__file__).resolve().parents[1]
    for name, digest in scheme["sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert set(scheme["validation"]) == {"layered", "small"}
    for diagnostic in scheme["validation"].values():
        assert diagnostic["step_ratio"] == 1
        assert diagnostic["max_relative_error"] <= tol.ORACLE_RTOL
        assert all(error <= tol.ORACLE_RTOL for error in diagnostic["relative_max_errors"].values())
    for key in tol.REASONS:
        assert meta["tolerances"][key]["value"] == getattr(tol, key)
        assert meta["tolerances"][key]["reason"] == tol.REASONS[key]
    for case in ("layered", "small"):
        g = golden(case)
        assert set(g) == set(meta["inputs"][case])
        for key, value in g.items():
            record = meta["inputs"][case][key]
            assert list(value.shape) == record["shape"]
            assert str(value.dtype) == record["dtype"]
            assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == record["sha256"]
    # The oracle itself must pass every resolution check before backends are
    # judged. Recompute measurements to catch a stale width convention/bundle.
    g = golden("layered")
    assert meta["width_transform_length_layered"] == g["v"].shape[0]
    for image, depth_key, width_key in (("rtm", "rtm_depths", "rtm_widths"),
                                       ("image", "ls_depths", "ls_widths")):
        depths, widths = reflector_metrics(g[image], g["columns"], g["windows"])
        assert np.array_equal(depths, g[depth_key])
        close(widths, g[width_key], tol.ORACLE_RTOL)
        assert np.all(np.abs(depths - g["reflector_depths"][:, None]) <= tol.MAX_DEPTH_ERROR_CELLS)
    assert np.all(g["ls_widths"] <= tol.MAX_WIDTH_RATIO * g["rtm_widths"])
    residual = np.linalg.norm(g["predicted"] - g["observed"])
    assert residual <= tol.MAX_RESIDUAL_RATIO * np.linalg.norm(g["observed"])
    assert residual <= tol.MAX_RESIDUAL_RATIO * np.linalg.norm(g["rtm_scaled_predicted"] - g["observed"])


@pytest.mark.parametrize("pair", ["mlx-metal"], ids=["mlx-metal"])
def test_cross_path_solver(layered, pair):
    mlx, metal = [solve_call(path, layered) for path in pair.split("-")]
    shapes = {"image": layered["v"].shape, "rtm": layered["v"].shape,
              "predicted": layered["observed"].shape,
              "residual_history": (int(layered["iterations"]) + 1,)}
    for out in (mlx, metal):
        check_output(out, shapes)
    for key in shapes:
        close(mlx[key], metal[key], tol.CROSS_RTOL, scale_reference=layered[key])
