"""Gravity of homogeneous rectangular prisms in Cartesian coordinates.

The corner antiderivatives are those of Nagy et al. (2000), with the
axis and vertex limits of Fukushima (2020). Field arithmetic is in MLX;
the public adapter materializes the required float32 NumPy result.
"""

import mlx.core as mx
import numpy as np

_G = 6.67430e-11


def _log_limit(x, y, z, radius):
    """Stable log(x + radius), including its axis and vertex limits."""
    transverse = y * y + z * z
    axis = transverse == 0
    # Keep discarded branches finite to avoid NaNs in derivatives.
    positive_arg = mx.where((x >= 0) & (radius > 0), x + radius, 1.0)
    denominator = mx.where(x < 0, radius - x, 1.0)
    ratio = mx.where((x < 0) & ~axis, transverse / denominator, 1.0)
    axis_arg = mx.where(x < 0, -2.0 * x, 1.0)
    negative = mx.where(axis, -mx.log(axis_arg), mx.log(ratio))
    return mx.where(radius == 0, 0.0, mx.where(x < 0, negative, mx.log(positive_arg)))


def _atan_limit(numerator, denominator):
    """Principal arctangent of a ratio, with finite values at zero."""
    safe_denominator = mx.where(denominator == 0, 1.0, denominator)
    ordinary = mx.arctan(numerator / safe_denominator)
    limit = mx.where(numerator > 0, 1.5707963267948966,
                     mx.where(numerator < 0, -1.5707963267948966, 0.0))
    return mx.where(denominator == 0, limit, ordinary)


def _corner(x, y, z, field):
    squared_radius = x * x + y * y + z * z
    radius = mx.where(squared_radius == 0, 0.0,
                      mx.sqrt(mx.where(squared_radius == 0, 1.0, squared_radius)))
    if field == 'potential':
        return (x * y * _log_limit(z, x, y, radius)
                + y * z * _log_limit(x, y, z, radius)
                + x * z * _log_limit(y, z, x, radius)
                - 0.5 * x * x * _atan_limit(y * z, x * radius)
                - 0.5 * y * y * _atan_limit(x * z, y * radius)
                - 0.5 * z * z * _atan_limit(x * y, z * radius))
    if field == 'g_e':
        return -(y * _log_limit(z, x, y, radius)
                 + z * _log_limit(y, z, x, radius)
                 - x * _atan_limit(y * z, x * radius))
    if field == 'g_n':
        return -(z * _log_limit(x, y, z, radius)
                 + x * _log_limit(z, x, y, radius)
                 - y * _atan_limit(x * z, y * radius))
    # The upward derivative has the opposite sign to downward gravity.
    return (x * _log_limit(y, z, x, radius)
            + y * _log_limit(x, y, z, radius)
            - z * _atan_limit(x * y, z * radius))


def prism_gravity_mlx(coordinates, prisms, density, field):
    """Return MLX fields, retaining autodiff through all three array inputs.

    Arguments, units and signs match :func:`prism_gravity`. To differentiate
    the potential with respect to observation coordinates, for example, use
    ``mx.grad(lambda xyz: mx.sum(prism_gravity_mlx(
    xyz, prisms, density, 'potential')['potential']))``. Its first two columns
    are eastward/northward acceleration in SI units; negate the third for
    downward acceleration and multiply all columns by 1e5 for mGal.
    """
    if field not in ('potential', 'g_e', 'g_n', 'g_z'):
        raise ValueError(f'Unknown gravity field: {field}')
    xyz = mx.asarray(coordinates, dtype=mx.float32)
    bounds = mx.asarray(prisms, dtype=mx.float32)
    rho = mx.asarray(density, dtype=mx.float32)
    # Subtract observations from corners before evaluating the antiderivative.
    total = mx.zeros((xyz.shape[0], bounds.shape[0]), dtype=mx.float32)
    for i in range(2):
        x = bounds[None, :, 1 - i] - xyz[:, None, 0]
        for j in range(2):
            y = bounds[None, :, 3 - j] - xyz[:, None, 1]
            for k in range(2):
                z = bounds[None, :, 5 - k] - xyz[:, None, 2]
                term = _corner(x, y, z, field)
                total = total + term if (i + j + k) % 2 == 0 else total - term
    scale = _G if field == 'potential' else _G * 1e5
    return {field: mx.sum(total * rho[None, :], axis=1) * scale}


def prism_gravity(coordinates, prisms, density, field):
    """Sum prism potential (J/kg) or acceleration (mGal) at observations.

    Return exactly ``{field: values}``, with a float32 NumPy array of shape
    ``(N,)``. Use :func:`prism_gravity_mlx` inside MLX autodiff transforms.
    """
    values = prism_gravity_mlx(coordinates, prisms, density, field)[field]
    return {field: np.asarray(values, dtype=np.float32)}
