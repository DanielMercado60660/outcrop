"""Locked MLX-only EikoNet tests. All problem inputs originate in judge/golden."""
from __future__ import annotations

import numpy as np
import pytest
from outcrop_harness import call, golden

from . import tolerances as tol

PATHS = ["mlx"]  # MLX-only project: no Metal implementation or cross-check required.
CASES = ["homogeneous_2d", "smooth_2d", "smooth_3d"]


def f32(value):
    return np.asarray(value, dtype=np.float32)


def network_args(g, parameters=None, sources=None, receivers=None):
    return {
        "parameters": f32(g["parameters"] if parameters is None else parameters),
        "sources": f32(g["sources"] if sources is None else sources),
        "receivers": f32(g["receivers"] if receivers is None else receivers),
        "origin": f32(g["origin"]),
        "length_scale": float(g["length_scale"]),
        "reference_velocity": float(g["reference_velocity"]),
        "width": int(g["width"]),
    }


def arrays(out, shapes):
    assert set(out) == set(shapes), f"output keys {set(out)} != {set(shapes)}"
    for name, shape in shapes.items():
        assert isinstance(out[name], np.ndarray), name
        assert out[name].shape == shape, (name, out[name].shape, shape)
        assert out[name].dtype == np.float32, (name, out[name].dtype)
        assert np.all(np.isfinite(out[name])), name


def relative_l2(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    denominator = np.linalg.norm(expected)
    assert denominator > 0, "test fixture must have a nonzero reference norm"
    return float(np.linalg.norm(actual - expected) / denominator)


def assert_parity(actual, expected):
    # Max error normalized by maximum reference magnitude: no absolute slack
    # can hide a missing small travel-time derivative or parameter gradient.
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    scale = np.max(np.abs(b))
    assert scale > 0
    error = float(np.max(np.abs(a - b)) / scale)
    assert error <= tol.NETWORK_REL, error


@pytest.fixture(scope="module", params=PATHS, ids=PATHS)
def path(request):
    return request.param


@pytest.fixture(scope="module", params=[2, 3], ids=["2d", "3d"])
def differential(request):
    return golden(f"derivatives_{request.param}d")


def test_fixed_network_forward_and_spatial_gradient(path, differential):
    g = differential
    out = call(path, "evaluate", **network_args(g))
    arrays(out, {"time": g["time"].shape, "receiver_gradient": g["receiver_gradient"].shape})
    assert_parity(out["time"], g["time"])
    assert_parity(out["receiver_gradient"], g["receiver_gradient"])


def test_fixed_physics_objective_and_parameter_gradient(path, differential):
    g = differential
    out = call(path, "physics_loss_and_grad", **network_args(g), velocities=f32(g["velocities"]))
    arrays(out, {"loss": (), "gradient": g["gradient"].shape})
    assert_parity(out["loss"], g["loss"])
    assert_parity(out["gradient"], g["gradient"])


def test_all_parameter_centered_differences(path, differential):
    g = differential
    base = call(path, "physics_loss_and_grad", **network_args(g), velocities=f32(g["velocities"]))
    arrays(base, {"loss": (), "gradient": g["gradient"].shape})
    fd = np.empty_like(g["gradient"])
    # EVERY coordinate, at the judge's saved, float32-representable +/- inputs.
    # Nominal h=0.002 dimensionless; divide by the actual represented separation.
    for j, (plus, minus) in enumerate(zip(g["parameter_plus"], g["parameter_minus"])):
        hi = call(path, "physics_loss_and_grad", **network_args(g, parameters=plus),
                  velocities=f32(g["velocities"]))
        lo = call(path, "physics_loss_and_grad", **network_args(g, parameters=minus),
                  velocities=f32(g["velocities"]))
        arrays(hi, {"loss": (), "gradient": g["gradient"].shape})
        arrays(lo, {"loss": (), "gradient": g["gradient"].shape})
        fd[j] = (float(hi["loss"]) - float(lo["loss"])) / (plus[j] - minus[j])
    assert relative_l2(base["gradient"], fd) <= tol.GRADIENT_FD_REL
    # Also anchors the differenced objective to the float64 oracle.
    assert relative_l2(fd, g["parameter_fd"]) <= tol.GRADIENT_FD_REL


def test_receiver_centered_differences(path, differential):
    g = differential
    base = call(path, "evaluate", **network_args(g))
    arrays(base, {"time": g["time"].shape, "receiver_gradient": g["receiver_gradient"].shape})
    fd = np.empty_like(g["receiver_gradient"])
    # h=1 metre; the judge saved both perturbations, so no test data is generated.
    for j, (plus, minus) in enumerate(zip(g["receiver_plus"], g["receiver_minus"])):
        hi = call(path, "evaluate", **network_args(g, receivers=plus))
        lo = call(path, "evaluate", **network_args(g, receivers=minus))
        arrays(hi, {"time": g["time"].shape, "receiver_gradient": g["receiver_gradient"].shape})
        arrays(lo, {"time": g["time"].shape, "receiver_gradient": g["receiver_gradient"].shape})
        fd[:, j] = (hi["time"].astype(np.float64) - lo["time"].astype(np.float64)) / (
            plus[:, j] - minus[:, j])
    assert relative_l2(base["receiver_gradient"], fd) <= tol.GRADIENT_FD_REL
    assert relative_l2(fd, g["receiver_fd"]) <= tol.GRADIENT_FD_REL


@pytest.fixture(scope="module", params=CASES, ids=CASES)
def medium(request):
    return golden(request.param)


@pytest.fixture(scope="module")
def trained(path, medium):
    g = medium
    kwargs = {
        "initial_parameters": f32(g["initial_parameters"]),
        "sources": f32(g["collocation_sources"]),
        "receivers": f32(g["collocation_receivers"]),
        "velocities": f32(g["collocation_velocities"]),
        "origin": f32(g["origin"]), "length_scale": float(g["length_scale"]),
        "reference_velocity": float(g["reference_velocity"]), "width": int(g["width"]),
        "steps": int(g["steps"]), "learning_rate": float(g["learning_rate"]),
    }
    copies = {key: value.copy() for key, value in kwargs.items() if isinstance(value, np.ndarray)}
    out = call(path, "train", **kwargs)
    arrays(out, {"parameters": g["initial_parameters"].shape})
    for key, before in copies.items():
        assert np.array_equal(kwargs[key], before), f"train mutated {key}"
    return out["parameters"]


@pytest.fixture(scope="module")
def predictions(path, medium, trained):
    out = call(path, "evaluate", **network_args(medium, parameters=trained))
    arrays(out, {"time": medium["time"].shape,
                 "receiver_gradient": medium["receivers"].shape})
    return out


@pytest.mark.parametrize("source_id", [0, 1, 2], ids=["source0", "source1", "source2"])
def test_held_out_travel_time_accuracy(path, medium, predictions, source_id):
    mask = medium["source_id"] == source_id
    truth, actual = medium["time"][mask], predictions["time"][mask].astype(np.float64)
    assert np.all(truth > 0)  # No zero-denominator masking or floors.
    assert np.all(actual > 0)
    errors = np.abs(actual - truth) / truth
    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    assert median <= tol.MEDIAN_TIME_REL, f"source {source_id}: median rel err {median:.6g}"
    assert p95 <= tol.P95_TIME_REL, f"source {source_id}: p95 rel err {p95:.6g}"


def test_trained_source_boundary(path, medium, trained):
    sources = medium["source_locations"]
    out = call(path, "evaluate", **network_args(medium, parameters=trained,
                                               sources=sources, receivers=sources))
    arrays(out, {"time": (len(sources),), "receiver_gradient": sources.shape})
    assert np.max(np.abs(out["time"])) <= tol.SOURCE_TIME_ATOL
    assert np.array_equal(out["receiver_gradient"], np.zeros_like(out["receiver_gradient"]))


def test_trained_objective_uses_returned_spatial_gradient(path, medium, trained):
    # The fixed-weight goldens anchor the network equations. This identity checks
    # that the trained objective and returned field still describe the same loss.
    g = medium
    kwargs = network_args(g, parameters=trained, sources=g["collocation_sources"],
                          receivers=g["collocation_receivers"])
    field = call(path, "evaluate", **kwargs)
    arrays(field, {"time": g["collocation_velocities"].shape,
                   "receiver_gradient": g["collocation_receivers"].shape})
    objective = call(path, "physics_loss_and_grad", **kwargs,
                     velocities=f32(g["collocation_velocities"]))
    arrays(objective, {"loss": (), "gradient": trained.shape})
    residual = g["collocation_velocities"]**2 * np.sum(
        field["receiver_gradient"].astype(np.float64)**2, axis=1) - 1
    expected = np.mean(residual**2)
    assert float(objective["loss"]) >= 0
    assert abs(np.sqrt(float(objective["loss"])) - np.sqrt(expected)) <= (
        tol.NETWORK_REL * np.sqrt(expected) + tol.OBJECTIVE_SQRT_ATOL)


def test_prediction_batch_order_invariance(path, medium, trained, predictions):
    # Reordered small batches exercise arbitrary query pairs, rather than one
    # special full-grid evaluation. This selects existing data, never generates it.
    for selection in (slice(None, None, -1), slice(None, None, 137)):
        sources = medium["sources"][selection]
        receivers = medium["receivers"][selection]
        out = call(path, "evaluate", **network_args(medium, parameters=trained,
                                                   sources=sources, receivers=receivers))
        arrays(out, {"time": (len(sources),), "receiver_gradient": receivers.shape})
        assert_parity(out["time"], predictions["time"][selection])
        assert_parity(out["receiver_gradient"], predictions["receiver_gradient"][selection])
