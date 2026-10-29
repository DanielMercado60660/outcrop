"""Magnetic induction of uniformly magnetized rectangular prisms in ENU."""

import mlx.core as mx


def _atan_ratio(numerator, denominator):
    """Principal arctangent of a ratio, including its zero-denominator limit."""
    nonzero = denominator != 0
    ratio = numerator / mx.where(nonzero, denominator, 1.0)
    limit = mx.where(numerator > 0, 1.5707963267948966,
                     mx.where(numerator < 0, -1.5707963267948966, 0.0))
    # At denominator == 0 and numerator != 0, the local derivative of
    # atan(numerator / denominator) with respect to the denominator is
    # -1 / numerator. The reciprocal form retains it at the chosen limit.
    safe_numerator = mx.where(numerator != 0, numerator, 1.0)
    correction = mx.where(numerator != 0,
                          mx.arctan(denominator / safe_numerator), 0.0)
    limit = limit - correction
    return mx.where(nonzero, mx.arctan(ratio), limit)


def _safe_log(x, y, z, radius):
    """Stable log(radius + x), including the collinear limiting value."""
    negative = x < 0
    transverse = y * y + z * z
    collinear = transverse == 0
    # Keep every branch finite, including those not selected by mx.where.
    positive_arg = mx.where(negative, 1.0, radius + x)
    negative_den = mx.where(negative, radius - x, 1.0)
    negative_num = mx.where(collinear, 1.0, transverse)
    limit_arg = mx.where(negative, -2.0 * x, 1.0)
    neg_value = mx.where(collinear, -mx.log(limit_arg),
                         mx.log(negative_num / negative_den))
    return mx.where(negative, neg_value, mx.log(positive_arg))


def prism_magnetic(coordinates, prisms, magnetization, inducing_field):
    """Return the induction in nT and its linear ambient-field projection."""
    coordinates = mx.array(coordinates, dtype=mx.float32)
    prisms = mx.array(prisms, dtype=mx.float32)
    magnetization = mx.array(magnetization, dtype=mx.float32)
    inducing_field = mx.array(inducing_field, dtype=mx.float32)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if prisms.ndim != 2 or prisms.shape[1] != 6:
        raise ValueError("prisms must have shape (P, 6)")
    if magnetization.shape != (prisms.shape[0], 3):
        raise ValueError("magnetization must have shape (P, 3)")
    if inducing_field.shape != (3,):
        raise ValueError("inducing_field must have shape (3,)")

    # Each corner has shape (observations, prisms). Subtract before squaring
    # so a large common coordinate translation does not degrade the geometry.
    corners = []
    for i in range(2):
        plane = []
        x = prisms[None, :, i] - coordinates[:, None, 0]
        for j in range(2):
            line = []
            y = prisms[None, :, 2 + j] - coordinates[:, None, 1]
            for k in range(2):
                z = prisms[None, :, 4 + k] - coordinates[:, None, 2]
                r = mx.sqrt(x * x + y * y + z * z)
                line.append((
                    -_atan_ratio(y * z, x * r),
                    -_atan_ratio(x * z, y * r),
                    -_atan_ratio(x * y, z * r),
                    _safe_log(z, x, y, r),
                    _safe_log(y, x, z, r),
                    _safe_log(x, y, z, r),
                ))
            plane.append(line)
        corners.append(plane)

    # Apply upper-minus-lower differences along each axis. Pairing adjacent
    # corners also preserves cancellation at symmetric observation points.
    tensors = []
    for t in range(6):
        d00 = corners[0][0][1][t] - corners[0][0][0][t]
        d01 = corners[0][1][1][t] - corners[0][1][0][t]
        d10 = corners[1][0][1][t] - corners[1][0][0][t]
        d11 = corners[1][1][1][t] - corners[1][1][0][t]
        tensors.append((d11 - d10) - (d01 - d00))
    ee, nn, uu, en, eu, nu = tensors
    me = magnetization[None, :, 0]
    mn = magnetization[None, :, 1]
    mu = magnetization[None, :, 2]
    # mu_0/(4*pi) in SI and tesla-to-nanotesla conversion.
    b_e = 100.0 * mx.sum(ee * me + en * mn + eu * mu, axis=1)
    b_n = 100.0 * mx.sum(en * me + nn * mn + nu * mu, axis=1)
    b_u = 100.0 * mx.sum(eu * me + nu * mn + uu * mu, axis=1)
    total = (b_e * inducing_field[0] + b_n * inducing_field[1]
             + b_u * inducing_field[2])
    return {"b_e": b_e, "b_n": b_n, "b_u": b_u, "total_field": total}
