"""Periodic Fourier-domain transformations for regularly sampled grids."""

import mlx.core as mx


def _direction(inclination, declination):
    inc = mx.array(inclination) * (mx.pi / 180.0)
    dec = mx.array(declination) * (mx.pi / 180.0)
    return mx.sin(dec) * mx.cos(inc), mx.cos(dec) * mx.cos(inc), mx.sin(inc)


def _difference(values, spacing, axis):
    """Centered interior and first-order one-sided boundary differences."""
    if values.shape[axis] == 1:
        return mx.zeros_like(values)
    if axis == 1:
        first = (values[:, 1:2] - values[:, :1]) / spacing
        middle = (values[:, 2:] - values[:, :-2]) / (2 * spacing)
        last = (values[:, -1:] - values[:, -2:-1]) / spacing
    else:
        first = (values[1:2, :] - values[:1, :]) / spacing
        middle = (values[2:, :] - values[:-2, :]) / (2 * spacing)
        last = (values[-1:, :] - values[-2:-1, :]) / spacing
    return mx.concatenate([first, middle, last], axis=axis)


def filter_grid(grid, dx, dy, operation, height, inclination, declination,
                magnetization_inclination, magnetization_declination):
    values = mx.array(grid, dtype=mx.float32)
    ny, nx = values.shape
    east_index = mx.arange(nx, dtype=mx.float32)
    north_index = mx.arange(ny, dtype=mx.float32)
    east_index = mx.where(east_index < (nx + 1) // 2, east_index, east_index - nx)
    north_index = mx.where(north_index < (ny + 1) // 2, north_index, north_index - ny)
    ke = (east_index * (2 * mx.pi / nx) / dx)[None, :]
    kn = (north_index * (2 * mx.pi / ny) / dy)[:, None]
    squared = ke * ke + kn * kn
    nonzero = squared != 0
    # Avoid the undefined derivative of sqrt at zero even in a masked branch.
    magnitude = mx.where(nonzero, mx.sqrt(mx.where(nonzero, squared, 1.0)), 0.0)

    if operation == "upward":
        factor = mx.exp(-height * magnitude)
    elif operation in ("vertical", "tilt"):
        factor = -magnitude
    elif operation == "rtp":
        fe, fn, fz = _direction(inclination, declination)
        me, mn, mz = _direction(magnetization_inclination,
                                magnetization_declination)
        imaginary = mx.array(1j, dtype=mx.complex64)
        field = fz * magnitude + imaginary * (fe * ke + fn * kn)
        magnetization = mz * magnitude + imaginary * (me * ke + mn * kn)
        denominator = field * magnetization
        factor = mx.where(nonzero, squared / mx.where(nonzero, denominator, 1.0), 0.0)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    spectrum = mx.fft.fft2(values)
    transformed = mx.real(mx.fft.ifft2(spectrum * factor))
    if operation == "tilt":
        east = _difference(values, dx, axis=1)
        north = _difference(values, dy, axis=0)
        transformed = mx.arctan2(transformed, mx.sqrt(east * east + north * north))
    return {"result": transformed.astype(mx.float32)}
