"""Parity against analytic Snell and constant-gradient ray references."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import ANGLE_REL, ANGLE_SCALE, CROSS_FACTOR, ENDPOINT_REL, TIME_REL

CASES = [
    "layered_multi", "layered_homogeneous", "layered_partial",
    "gradient_positive", "gradient_negative", "gradient_zero", "gradient_shallow",
]


def arguments(data, case, operation):
    args = {"depth": float(data["depth"])}
    if case.startswith("layered"):
        args.update(interfaces=data["interfaces"], velocities=data["velocities"])
        kind = "layered"
    else:
        args.update(v0=float(data["v0"]), gradient=float(data["gradient"]))
        kind = "gradient"
    args["angles" if operation == "shoot" else "offsets"] = data[
        "angles" if operation == "shoot" else "offsets"]
    return operation + "_" + kind, args


def compare(actual, expected, operation, factor):
    keys = ("endpoints", "times", "angles") if operation == "bend" else ("endpoints", "times")
    assert set(actual) == set(keys)
    for key in keys:
        assert actual[key].shape == expected[key].shape, key
        assert actual[key].dtype == np.float32, key
        assert np.all(np.isfinite(actual[key])), key
    distance = np.linalg.norm(expected["endpoints"].astype(np.float64), axis=1)
    error = np.linalg.norm(actual["endpoints"].astype(np.float64) - expected["endpoints"], axis=1)
    assert np.all(error <= factor * ENDPOINT_REL * distance), error / distance
    error = np.abs(actual["times"].astype(np.float64) - expected["times"])
    assert np.all(error <= factor * TIME_REL * np.abs(expected["times"])), error / expected["times"]
    if operation == "bend":
        error = np.abs(actual["angles"].astype(np.float64) - expected["angles"])
        scale = np.maximum(np.abs(expected["angles"]), ANGLE_SCALE)
        assert np.all(error <= factor * ANGLE_REL * scale), error / scale


def reference(data, operation):
    prefix = "shot" if operation == "shoot" else "bent"
    result = {key: data[prefix + "_" + key] for key in ("endpoints", "times")}
    if operation == "bend":
        result["angles"] = data["bent_angles"]
    return result


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("operation", ["shoot", "bend"])
@pytest.mark.parametrize("case", CASES)
def test_analytic(path, operation, case):
    data = golden(case)
    fn, args = arguments(data, case, operation)
    compare(call(path, fn, **args), reference(data, operation), operation, 1.0)


@pytest.mark.parametrize("operation", ["shoot", "bend"])
@pytest.mark.parametrize("case", CASES)
def test_mlx_metal_crosscheck(operation, case):
    data = golden(case)
    fn, args = arguments(data, case, operation)
    mlx = call("mlx", fn, **args)
    metal = call("metal", fn, **args)
    # Validate both independently, then use the float64 reference for scaling.
    ref = reference(data, operation)
    compare(mlx, ref, operation, 1.0)
    compare(metal, ref, operation, 1.0)
    # Compare differences against the reference scale (not a rounded output).
    for key in ref:
        assert mlx[key].shape == metal[key].shape
        if key == "endpoints":
            err = np.linalg.norm(mlx[key].astype(np.float64) - metal[key], axis=1)
            scale = np.linalg.norm(ref[key], axis=1)
            limit = ENDPOINT_REL
        elif key == "times":
            err = np.abs(mlx[key].astype(np.float64) - metal[key])
            scale = np.abs(ref[key])
            limit = TIME_REL
        else:
            err = np.abs(mlx[key].astype(np.float64) - metal[key])
            scale = np.maximum(np.abs(ref[key]), ANGLE_SCALE)
            limit = ANGLE_REL
        assert np.all(err <= CROSS_FACTOR * limit * scale), err / scale
