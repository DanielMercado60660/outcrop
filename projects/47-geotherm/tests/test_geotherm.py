"""Analytic parity, independent age recovery, and float64 gradient checks."""

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, golden_meta

from .tolerances import (
    AGE_ATOL_MYR,
    AGE_EXACT_RTOL,
    AGE_RECOVERY_RTOL,
    FLUX_ATOL_W_M2,
    FORWARD_RTOL,
    GRADIENT_ATOL_PER_MYR,
    GRADIENT_RTOL,
    OBJECTIVE_ATOL,
    TEMPERATURE_ATOL_C,
)


def params(alternate=False):
    meta = golden_meta()
    key = "parameters_alternate_and_inversion" if alternate else "parameters_forward_and_gradient"
    return meta[key]


def array32(value):
    return np.asarray(value, dtype=np.float32)


def check_array(out, name, shape):
    assert name in out
    actual = out[name]
    assert isinstance(actual, np.ndarray)
    assert actual.shape == shape
    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    return actual


def forward_args(case):
    data = golden(case)
    return data, dict(depth_m=array32(data["depth_m"]), age_myr=array32(data["age_myr"]),
                      **params(case == "forward_alt"))


def inversion_args():
    data = golden("inversion")
    return data, dict(surface_heat_flow_w_m2=array32(data["surface_heat_flow_w_m2"]), **params(True))


def gradient_args():
    data = golden("gradient")
    return data, dict(age_myr=array32(data["age_myr"]),
                      observed_heat_flow_w_m2=array32(data["observed_heat_flow_w_m2"]),
                      flux_scale_w_m2=golden_meta()["flux_scale_w_m2"], **params())


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("case", ["forward", "forward_alt"])
def test_forward(path, case):
    data, kwargs = forward_args(case)
    out = call(path, "geotherm", **kwargs)
    shape = (data["age_myr"].size, data["depth_m"].size)
    np.testing.assert_allclose(check_array(out, "temperature_c", shape), data["temperature_c"],
                               rtol=FORWARD_RTOL, atol=TEMPERATURE_ATOL_C)
    np.testing.assert_allclose(check_array(out, "heat_flow_w_m2", shape), data["heat_flow_w_m2"],
                               rtol=FORWARD_RTOL, atol=FLUX_ATOL_W_M2)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_recover_age(path):
    data, kwargs = inversion_args()
    out = call(path, "invert_age", **kwargs)
    ages = check_array(out, "age_myr", data["age_myr"].shape)
    np.testing.assert_allclose(ages, data["age_myr"], rtol=AGE_RECOVERY_RTOL, atol=0)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_recover_age_exact(path):
    data, kwargs = inversion_args()
    out = call(path, "invert_age", **kwargs)
    ages = check_array(out, "age_myr", data["age_myr"].shape)
    np.testing.assert_allclose(ages, data["age_myr"], rtol=AGE_EXACT_RTOL, atol=0)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_misfit_gradient_centered_difference(path):
    data, kwargs = gradient_args()
    out = call(path, "heat_flow_misfit_and_grad", **kwargs)
    objective = check_array(out, "objective", ())
    gradient = check_array(out, "gradient", data["age_myr"].shape)
    np.testing.assert_allclose(objective, data["objective"], rtol=FORWARD_RTOL, atol=OBJECTIVE_ATOL)
    # Oracle evaluates both sides in float64, with h = 0.01 Myr per coordinate.
    h = golden_meta()["finite_difference_step_myr"]
    finite_difference = (data["objective_plus"] - data["objective_minus"]) / (2 * h)
    np.testing.assert_allclose(gradient, finite_difference, rtol=GRADIENT_RTOL,
                               atol=GRADIENT_ATOL_PER_MYR)


@pytest.mark.parametrize("case", ["forward", "forward_alt"])
def test_cross_mlx_metal_forward(case):
    data, kwargs = forward_args(case)
    a, b = (call(path, "geotherm", **kwargs) for path in PATHS)
    shape = (data["age_myr"].size, data["depth_m"].size)
    for key, atol in [("temperature_c", TEMPERATURE_ATOL_C), ("heat_flow_w_m2", FLUX_ATOL_W_M2)]:
        np.testing.assert_allclose(check_array(a, key, shape), check_array(b, key, shape),
                                   rtol=FORWARD_RTOL, atol=atol)


def test_cross_mlx_metal_inversion():
    data, kwargs = inversion_args()
    a, b = (call(path, "invert_age", **kwargs) for path in PATHS)
    shape = data["age_myr"].shape
    np.testing.assert_allclose(check_array(a, "age_myr", shape), check_array(b, "age_myr", shape),
                               rtol=FORWARD_RTOL, atol=AGE_ATOL_MYR)


def test_cross_mlx_metal_gradient():
    data, kwargs = gradient_args()
    a, b = (call(path, "heat_flow_misfit_and_grad", **kwargs) for path in PATHS)
    for key, shape, atol in [("objective", (), OBJECTIVE_ATOL),
                             ("gradient", data["age_myr"].shape, GRADIENT_ATOL_PER_MYR)]:
        np.testing.assert_allclose(check_array(a, key, shape), check_array(b, key, shape),
                                   rtol=GRADIENT_RTOL, atol=atol)
