"""Focused solver regressions; run separately from the locked tests."""

import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture(scope="module")
def impl():
    spec = importlib.util.spec_from_file_location(
        "slip_inversion_impl", Path(__file__).with_name("impl.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("zero_data", [False, True])
def test_poorly_conditioned_full_rank(impl, zero_data):
    # A.T @ A loses the 1e-8 contribution in float32 and becomes singular.
    green = np.array([[1, 1], [0, 1e-4]], dtype=np.float32)
    data = np.array([1, 2e-4], dtype=np.float32)
    expected = np.array([-1, 2], dtype=np.float32)
    if zero_data:
        data[:] = 0
        expected[:] = 0
    out = impl.invert_slip(
        green=green,
        data=data,
        sigma=np.ones(2, dtype=np.float32),
        regularizer=np.empty((0, 2), dtype=np.float32),
        damping=0.0,
        areas=np.array([3, 5], dtype=np.float32),
        shear_modulus=7.0,
    )
    for key, shape in {"slip": (2,), "predicted": (2,), "moment": ()}.items():
        value = np.asarray(out[key])
        assert value.shape == shape
        assert value.dtype == np.float32
        assert np.all(np.isfinite(value))
        if zero_data:
            np.testing.assert_array_equal(value, np.zeros(shape, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(out["slip"]), expected, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(out["predicted"]), data, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(out["moment"]), 0 if zero_data else 91, rtol=1e-4)


def test_least_squares_vjp_with_nonzero_residual(impl):
    a = np.array([[1, 1], [0, 1e-4], [0, 2e-4]], dtype=np.float32)
    b = np.array([[1], [3e-4], [3e-4]], dtype=np.float32)
    cotangent = np.array([[0.7], [-0.2]], dtype=np.float32)
    # Independent double-precision QR checks both input derivatives, including
    # the matrix derivative's residual term for an inconsistent system.
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    q, r = np.linalg.qr(a64)
    solution = np.linalg.solve(r, q.T @ b64)
    y = np.linalg.solve(r.T, cotangent.astype(np.float64))
    z = np.linalg.solve(r, y)
    expected_b = q @ y
    expected_a = (b64 - a64 @ solution) @ z.T - expected_b @ solution.T
    with mx.stream(mx.cpu):
        c = mx.array(cotangent)

        def loss(matrix, rhs):
            return mx.sum(impl._least_squares(matrix, rhs) * c)

        grad_a, grad_b = mx.grad(loss, argnums=(0, 1))(mx.array(a), mx.array(b))
        np.testing.assert_allclose(np.asarray(grad_a), expected_a, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(grad_b), expected_b, rtol=1e-4, atol=1e-5)
