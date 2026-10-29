"""Signed gravity of a DEM prism layer, in upward coordinates."""

import mlx.core as mx

_G_MGAL = 6.67430e-6
_NODES = (-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
          0.2386191860831969, 0.6612093864662645, 0.9324695142031521)
_WEIGHTS = (0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
            0.4679139345726910, 0.3607615730481386, 0.1713244923791704)


def _kernel(x, y, z):
    """Prism corner primitive of the downward vertical attraction."""
    r = mx.sqrt(x * x + y * y + z * z)
    # Alternate forms avoid cancellation in r+x and r+y for negative x/y.
    arg_x = mx.where(x < 0, (y * y + z * z) / mx.maximum(r - x, 1e-30), r + x)
    arg_y = mx.where(y < 0, (x * x + z * z) / mx.maximum(r - y, 1e-30), r + y)
    log_x = mx.log(arg_x)
    log_y = mx.log(arg_y)
    return x * log_y + y * log_x - z * mx.arctan((x * y) / (z * r))


def terrain_bouguer(easting, northing, surface, reference, density, coordinates):
    """Return signed downward layer/slab gravity and corrections in mGal.

    Near cells use the prism primitive. For clearance of at least two cell
    widths, a 6 by 6 horizontal Gauss rule avoids corner cancellation; the
    vertical integral is analytic. All physical arithmetic stays in MLX so
    surface, reference, density, and observation derivatives remain available.
    """
    easting = mx.array(easting, dtype=mx.float32)
    northing = mx.array(northing, dtype=mx.float32)
    surface = mx.array(surface, dtype=mx.float32)
    reference = mx.array(reference, dtype=mx.float32)
    density = mx.array(density, dtype=mx.float32)
    coordinates = mx.array(coordinates, dtype=mx.float32)

    nx, ny = easting.shape[0], northing.shape[0]
    dx, dy = easting[1] - easting[0], northing[1] - northing[0]
    centers_x = mx.broadcast_to(easting[None, :], (ny, nx)).reshape(-1)
    centers_y = mx.broadcast_to(northing[:, None], (ny, nx)).reshape(-1)
    x = centers_x[None, :] - coordinates[:, 0:1]
    y = centers_y[None, :] - coordinates[:, 1:2]
    relief = (surface - reference).reshape(-1)
    z_surface = surface.reshape(1, -1) - coordinates[:, 2:3]
    z_reference = reference.reshape(1, -1) - coordinates[:, 2:3]

    # Integrating from reference to surface directly makes depressions
    # negative. Pair the vertical endpoints before accumulating corners:
    # equal endpoints then give exactly zero, with the correct derivative.
    corner_sum = mx.zeros_like(x)
    for i in range(2):
        xx = x + (i - 0.5) * dx
        for j in range(2):
            yy = y + (j - 0.5) * dy
            sign = 1.0 if (i + j) % 2 == 0 else -1.0
            vertical_difference = (_kernel(xx, yy, z_surface)
                                   - _kernel(xx, yy, z_reference))
            corner_sum = corner_sum + sign * vertical_difference

    # The vertical integral is 1/r_surface - 1/r_reference. Rationalize
    # that difference, using relief calculated BEFORE subtracting observation
    # height so even thin cells retain their signed thickness at altitude.
    vertical_numerator = -relief[None, :] * (z_surface + z_reference)
    quadrature = mx.zeros_like(x)
    for node_x, weight_x in zip(_NODES, _WEIGHTS):
        xx = x + (0.5 * node_x) * dx
        for node_y, weight_y in zip(_NODES, _WEIGHTS):
            yy = y + (0.5 * node_y) * dy
            horizontal_sq = xx * xx + yy * yy
            rs = mx.sqrt(horizontal_sq + z_surface * z_surface)
            rr = mx.sqrt(horizontal_sq + z_reference * z_reference)
            integral = vertical_numerator / (rs * rr * (rs + rr))
            quadrature = quadrature + (weight_x * weight_y) * integral
    quadrature = quadrature * (0.25 * dx * dy)
    clearance = -mx.maximum(z_surface, z_reference)
    prism = mx.where(clearance >= 2.0 * mx.maximum(dx, dy), quadrature, corner_sum)
    layer = mx.sum(prism * density.reshape(1, -1), axis=1) * _G_MGAL

    ix = mx.clip(mx.floor((coordinates[:, 0] - easting[0]) / dx + 0.5), 0, nx - 1).astype(mx.int32)
    iy = mx.clip(mx.floor((coordinates[:, 1] - northing[0]) / dy + 0.5), 0, ny - 1).astype(mx.int32)
    slab = mx.take((density * (surface - reference)).reshape(-1), iy * nx + ix)
    slab = slab * (2.0 * 3.141592653589793 * _G_MGAL)
    return {"layer_gz": layer, "slab_gz": slab,
            "terrain_gz": layer - slab, "correction_gz": -layer}
