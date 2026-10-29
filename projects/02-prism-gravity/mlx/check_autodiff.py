"""Standalone autodiff checks; independent of the locked tests and goldens.

Run with the repository's pinned Python: ``python mlx/check_autodiff.py``. The float64
centered differences below use a scalar closed-form potential written from
the corner integral, never the MLX acceleration formulas. NumPy float64 is
used only for this validation oracle, not the production implementation.
"""

import math

import mlx.core as mx
import numpy as np
from impl import prism_gravity, prism_gravity_mlx


def _potential64(coordinates, prisms, density):
    """Scalar float64 corner integral, with zero-weight terms omitted."""
    values = []
    for point in np.asarray(coordinates, dtype=np.float64):
        sources = []
        for bounds, rho in zip(prisms, density):
            bounds = np.asarray(bounds, dtype=np.float64)
            corners = []
            for i in range(2):
                x = float(bounds[i] - point[0])
                for j in range(2):
                    y = float(bounds[2 + j] - point[1])
                    for k in range(2):
                        z = float(bounds[4 + k] - point[2])
                        r = math.sqrt(x * x + y * y + z * z)
                        terms = []
                        for a, b, c in ((x, y, z), (y, z, x), (z, x, y)):
                            # A nonzero coefficient guarantees a finite log.
                            if a * b != 0:
                                arg = ((a * a + b * b) / (r - c)
                                       if c < 0 else r + c)
                                terms.append(a * b * math.log(arg))
                            if a != 0:
                                terms.append(-0.5 * a * a
                                             * math.atan((b * c) / (a * r)))
                        # Here i/j/k select lower, then upper bounds.
                        sign = -1 if (i + j + k) % 2 == 0 else 1
                        corners.append(sign * math.fsum(terms))
            sources.append(float(rho) * math.fsum(corners))
        values.append(6.67430e-11 * math.fsum(sources))
    return np.array(values, dtype=np.float64)


def _differences(function, values, step):
    values = np.asarray(values, dtype=np.float64)
    gradient = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2 * step)
    return gradient


def _relative_check(actual, expected, label, tolerance=1e-3):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, label
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
    scale = np.max(np.abs(expected))
    assert scale > 0, label
    error = np.max(np.abs(actual - expected)) / scale
    print(f'{label}: max relative error = {error:.6g}')
    assert error <= tolerance, (label, error, tolerance)


def _potential_sum(xyz, bounds, rho):
    return mx.sum(prism_gravity_mlx(xyz, bounds, rho, 'potential')['potential'])


def _check_coordinates(xyz, bounds, rho, label):
    # Potential is continuously differentiable even at prism boundaries.
    # At a face, centered differences can have O(h) error because the second
    # derivative jumps. Check two steps instead of assuming O(h**2) there.
    def objective64(points):
        return np.sum(_potential64(points, bounds, rho))

    coarse = _differences(objective64, xyz, 1 / 32)
    fine = _differences(objective64, xyz, 1 / 64)
    autodiff = np.asarray(mx.grad(_potential_sum)(
        mx.array(xyz), mx.array(bounds), mx.array(rho)))
    for axis, field in enumerate(('g_e', 'g_n', 'g_z')):
        _relative_check(coarse[:, axis], fine[:, axis],
                        f'{label} {field} float64 step refinement')
        _relative_check(autodiff[:, axis], fine[:, axis],
                        f'{label} {field} mx.grad vs float64 differences')
        # Also pin the coordinate convention and mGal conversion.
        acceleration = autodiff[:, axis] * (-1e5 if field == 'g_z' else 1e5)
        _relative_check(acceleration, prism_gravity(xyz, bounds, rho, field)[field],
                        f'{label} {field} mx.grad vs field')


def main():
    bounds = np.array([[-100, 100, -80, 80, -120, 0]], dtype=np.float32)
    rho = np.array([2670], dtype=np.float32)
    # Exterior, interior, center, both face orientations, edges, vertices,
    # and extensions of corner axes in both directions exercise all limits.
    xyz = np.array([
        [173, -129, 211], [13, -17, -51], [0, 0, -60],
        [100, 13, -51], [-100, -11, -37], [13, 80, -51],
        [13, -17, 0], [13, -17, -120],
        [100, 80, -51], [-100, -80, -37],
        [100, 80, 0], [-100, -80, -120],
        [100, 80, 64], [100, 80, -184],
        [164, 80, 0], [-164, -80, -120],
        [100, 144, 0], [-100, -144, -120],
    ], dtype=np.float32)
    _check_coordinates(xyz, bounds, rho, 'single/boundary')

    mixed_bounds = np.concatenate((bounds, np.array([
        [140, 220, -140, -100, -230, -150],
        [-220, -160, 110, 170, -90, -30],
    ], dtype=np.float32)))
    mixed_rho = np.array([2670, -900, 0], dtype=np.float32)
    mixed_xyz = np.concatenate((xyz, np.array([
        [140, -140, -150], [-220, 110, -30],
    ], dtype=np.float32)))
    _check_coordinates(mixed_xyz, mixed_bounds, mixed_rho, 'mixed/boundary')
    shift = np.array([16384, -8192, 4096], dtype=np.float32)
    _check_coordinates(xyz + shift, bounds + np.repeat(shift, 2), rho,
                       'translated/boundary')

    # Bounds and density must remain connected to autodiff too. Exterior
    # observations avoid second-derivative singularities of the field.
    exterior = xyz[:1]
    args = (mx.array(exterior), mx.array(mixed_bounds), mx.array(mixed_rho))
    bounds_ad = mx.grad(_potential_sum, argnums=1)(*args)
    bounds_fd = _differences(
        lambda b: np.sum(_potential64(exterior, b, mixed_rho)), mixed_bounds, 1 / 64)
    _relative_check(bounds_ad, bounds_fd, 'potential bounds mx.grad')
    rho_ad = mx.grad(_potential_sum, argnums=2)(*args)
    rho_fd = _differences(
        lambda r: np.sum(_potential64(exterior, mixed_bounds, r)), mixed_rho, 1.0)
    _relative_check(rho_ad, rho_fd, 'potential density mx.grad')

    for field in ('potential', 'g_e', 'g_n', 'g_z'):
        out = prism_gravity(mixed_xyz, mixed_bounds, mixed_rho, field)
        assert set(out) == {field}
        assert isinstance(out[field], np.ndarray)
        assert out[field].dtype == np.float32
        assert out[field].shape == (len(mixed_xyz),)
        assert np.isfinite(out[field]).all()
        core = prism_gravity_mlx(mixed_xyz, mixed_bounds, mixed_rho, field)[field]
        np.testing.assert_array_equal(out[field], np.asarray(core))
    print('Autodiff and direct NumPy API checks passed.')


if __name__ == '__main__':
    main()
