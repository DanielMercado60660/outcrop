"""Additional direct-call regressions, separate from the locked tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "project23_mlx_impl", Path(__file__).with_name("impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)


@pytest.mark.parametrize("kind", ["l2", "envelope"])
def test_direct_numpy_boundary(kind):
    predicted = np.array([[0.2, -0.5, 0.7, 0.1]], dtype=np.float32)
    out = _impl.misfit_and_grad(
        predicted, np.zeros_like(predicted), 0.002, kind, 1e-3
    )
    assert set(out) == {"objective", "gradient"}
    assert out["objective"].shape == ()
    assert out["gradient"].shape == predicted.shape
    for value in out.values():
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.dtype("float32")
        assert np.all(np.isfinite(value))


@pytest.mark.parametrize("ntime", [31, 32])
@pytest.mark.parametrize("amplitude", [0.0, 1e-30, 1e20])
@pytest.mark.parametrize("epsilon", [1e-30, 1e-3])
@pytest.mark.parametrize("polarity", [-1.0, 1.0])
def test_extreme_envelope_zero(ntime, amplitude, epsilon, polarity):
    # Include DC, multiple traces, and odd/even transforms. The strided view
    # also checks the direct boundary with differently laid out host arrays.
    samples = np.arange(3 * ntime, dtype=np.float32).reshape(3, ntime)
    observed = (amplitude * np.cos(samples)).astype(np.float32)
    storage = np.empty((3, 2 * ntime), dtype=np.float32)
    predicted = storage[:, ::2]
    predicted[:] = polarity * observed
    out = _impl.misfit_and_grad(predicted, observed, 0.002, "envelope", epsilon)
    for value in out.values():
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.dtype("float32")
        assert np.all(np.isfinite(value))
        np.testing.assert_array_equal(value, np.zeros_like(value))
