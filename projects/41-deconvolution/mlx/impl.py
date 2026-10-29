"""Causal least-squares spiking and predictive deconvolution in MLX."""

from math import isfinite
from operator import index

import mlx.core as mx


def _positive_integer(value, name):
    """Validate discrete filter parameters before constructing MLX arrays."""
    try:
        value = index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_stabilization(stabilization):
    if not isfinite(stabilization) or stabilization < 0:
        raise ValueError("stabilization must be finite and nonnegative")


@mx.custom_function
def _solve(a, b):
    """Solve a square system with a column-vector right-hand side."""
    return mx.linalg.solve(a, b, stream=mx.cpu)


@_solve.vjp
def _solve_vjp(primals, cotangent, output):
    a, _ = primals
    adjoint = mx.linalg.solve(mx.swapaxes(a, -1, -2), cotangent, stream=mx.cpu)
    return -(adjoint @ mx.swapaxes(output, -1, -2)), adjoint


def spiking_deconvolution(trace, wavelet, inverse_length, stabilization):
    length = _positive_integer(inverse_length, "inverse_length")
    _validate_stabilization(stabilization)
    x = mx.asarray(trace, dtype=mx.float32)
    w = mx.asarray(wavelet, dtype=mx.float32)
    # Each column contains the wavelet shifted by its causal filter delay.
    rows = mx.arange(w.shape[0] + length - 1)[:, None]
    cols = mx.arange(length)[None, :]
    indices = rows - cols
    padded = mx.concatenate([mx.zeros((1,), dtype=mx.float32), w])
    indices = mx.where((indices >= 0) & (indices < w.shape[0]), indices + 1, 0)
    a = padded[indices]
    normal = a.T @ a + stabilization * mx.eye(length, dtype=mx.float32)
    rhs = a[0, :].reshape(length, 1)
    inverse = _solve(normal, rhs).reshape(length)
    recovered = mx.convolve(x, inverse, mode="full")[:x.shape[0]]
    return {"inverse": inverse, "recovered": recovered}


def predictive_deconvolution(trace, order, lag, stabilization):
    order = _positive_integer(order, "order")
    lag = _positive_integer(lag, "lag")
    _validate_stabilization(stabilization)
    x = mx.asarray(trace, dtype=mx.float32)
    n = x.shape[0]
    max_lag = max(order - 1, lag + order - 1)
    correlations = mx.stack([
        mx.sum(x[:n - k] * x[k:]) if k < n else mx.zeros((), dtype=mx.float32)
        for k in range(max_lag + 1)
    ])
    indices = mx.arange(order)
    normal = correlations[mx.abs(indices[:, None] - indices[None, :])]
    normal = normal + stabilization * correlations[0] * mx.eye(order, dtype=mx.float32)
    rhs = correlations[lag:lag + order].reshape(order, 1)
    coefficients = _solve(normal, rhs).reshape(order)
    error_filter = mx.concatenate([
        mx.ones((1,), dtype=mx.float32),
        mx.zeros((lag - 1,), dtype=mx.float32),
        -coefficients,
    ])
    residual = mx.convolve(x, error_filter, mode="full")[:n]
    return {"coefficients": coefficients, "residual": residual}
