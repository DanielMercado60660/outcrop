"""One-dimensional acoustic leapfrog propagation with Mur boundaries."""

import math
import operator
import sys

import mlx.core as mx


def propagate(u0, u_prev, c, dx, dt, steps):
    """Return the displacement at each time, including the initial field."""
    steps = operator.index(steps)
    if steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    old = mx.asarray(u0, dtype=mx.float32)
    prev = mx.asarray(u_prev, dtype=mx.float32)
    distance = c * dt
    if math.isfinite(distance) and distance >= sys.float_info.min:
        courant = distance / dx
    else:
        # Scale the product before division so intermediate overflow or
        # underflow does not obscure a representable Courant number.
        c_mantissa, c_exponent = math.frexp(c)
        dt_mantissa, dt_exponent = math.frexp(dt)
        dx_mantissa, dx_exponent = math.frexp(dx)
        courant = math.ldexp(
            c_mantissa * dt_mantissa / dx_mantissa,
            c_exponent + dt_exponent - dx_exponent,
        )
    courant_squared = courant * courant
    q = (courant - 1.0) / (courant + 1.0)
    fields = [old]

    for step in range(steps):
        center = old[1:-1]
        laplacian = old[:-2] - 2.0 * center + old[2:]
        interior = 2.0 * center - prev[1:-1] + courant_squared * laplacian
        left = old[1:2] + q * (interior[:1] - old[:1])
        right = old[-2:-1] + q * (interior[-1:] - old[-1:])
        new = mx.concatenate((left, interior, right))
        fields.append(new)
        prev, old = old, new

        # Bound the size of the pending lazy computation for long runs.
        if (step + 1) % 32 == 0:
            mx.eval(prev, old)

    return {"wavefield": mx.stack(fields, axis=0)}
