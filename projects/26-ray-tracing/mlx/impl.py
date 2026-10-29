"""Direct downgoing rays in horizontal layers and linear-gradient media."""

import mlx.core as mx


def _layers(interfaces, velocities, depth):
    interfaces = mx.array(interfaces, dtype=mx.float32)
    velocities = mx.array(velocities, dtype=mx.float32)
    bounds = mx.concatenate([mx.array([0.0]), interfaces, mx.array([depth])])
    bounds = mx.clip(bounds, 0.0, depth)
    return bounds[1:] - bounds[:-1], velocities


def _layer_forward(q, thickness, velocity, vmax):
    # q is the horizontal slowness scaled by the largest traversed velocity.
    u = q[:, None] * (velocity / vmax)[None, :]
    # Untraversed layers cannot contribute, even if their velocity is higher.
    u = mx.where(thickness[None, :] > 0, u, 0.0)
    cosine = mx.sqrt(mx.maximum(1.0 - u * u, 1.0e-30))
    distance = mx.sum(thickness[None, :] * u / cosine, axis=1)
    time = mx.sum(thickness[None, :] / (velocity[None, :] * cosine), axis=1)
    derivative = mx.sum(
        thickness[None, :] * (velocity / vmax)[None, :] / (cosine ** 3),
        axis=1,
    )
    return distance, time, derivative


def _endpoints(x, depth):
    return mx.stack([x, mx.ones_like(x) * depth], axis=1)


def shoot_layered(interfaces, velocities, angles, depth):
    thickness, velocity = _layers(interfaces, velocities, depth)
    angles = mx.array(angles, dtype=mx.float32)
    vmax = mx.max(mx.where(thickness > 0, velocity, 0.0))
    q = mx.sin(angles) * (vmax / velocity[0])
    x, time, _ = _layer_forward(q, thickness, velocity, vmax)
    return {"endpoints": _endpoints(x, depth), "times": time}


def bend_layered(interfaces, velocities, offsets, depth):
    thickness, velocity = _layers(interfaces, velocities, depth)
    offsets = mx.array(offsets, dtype=mx.float32)
    vmax = mx.max(mx.where(thickness > 0, velocity, 0.0))
    # t is the tangent of the angle in the fastest traversed layer. Unlike
    # its sine, it remains distinguishable from the critical value for
    # arbitrarily shallow rays in float32.
    ratio = mx.where(thickness > 0, velocity / vmax, 0.0)
    fastest_thickness = mx.sum(mx.where((thickness > 0) & (velocity == vmax), thickness, 0.0))
    high = mx.abs(offsets) / fastest_thickness
    low = -high

    def forward(t):
        inv_norm = 1.0 / mx.sqrt(1.0 + t * t)
        sine = ratio[None, :] * (t * inv_norm)[:, None]
        cosine = mx.sqrt(
            (1.0 - ratio[None, :]) * (1.0 + ratio[None, :])
            + (ratio[None, :] * inv_norm[:, None]) ** 2
        )
        x = mx.sum(thickness[None, :] * sine / cosine, axis=1)
        time = mx.sum(thickness[None, :] / (velocity[None, :] * cosine), axis=1)
        derivative = mx.sum(
            thickness[None, :] * ratio[None, :] * inv_norm[:, None] ** 3
            / cosine ** 3, axis=1
        )
        return x, time, derivative

    for _ in range(40):
        mid = (low + high) * 0.5
        x, _, _ = forward(mid)
        low = mx.where(x < offsets, mid, low)
        high = mx.where(x < offsets, high, mid)
    t = (low + high) * 0.5
    # Refinement propagates derivatives through the implicit solution.
    for _ in range(2):
        x, _, derivative = forward(t)
        t = t + (offsets - x) / derivative
    _, time, _ = forward(t)
    r0 = velocity[0] / vmax
    angles = mx.arctan2(
        r0 * t, mx.sqrt((1.0 - r0) * (1.0 + r0) * (1.0 + t * t) + r0 * r0)
    )
    return {"endpoints": _endpoints(offsets, depth), "times": time, "angles": angles}


def _gradient_forward(v0, gradient, p, depth):
    v1 = v0 + gradient * depth
    c0 = mx.sqrt(mx.maximum(1.0 - (p * v0) ** 2, 0.0))
    c1 = mx.sqrt(mx.maximum(1.0 - (p * v1) ** 2, 0.0))
    x = p * depth * (v0 + v1) / (c0 + c1)
    if gradient == 0.0:
        time = depth / (v0 * c0)
    else:
        # Rationalizing c0-c1 and using log1p avoids small-gradient and
        # near-vertical cancellation in the analytic depth integral.
        dv = gradient * depth
        correction = p * p * dv * (v0 + v1) / ((c0 + c1) * (1.0 + c1))
        time = (mx.log1p(dv / v0) + mx.log1p(correction)) / gradient
    return x, time


def shoot_gradient(v0, gradient, angles, depth):
    angles = mx.array(angles, dtype=mx.float32)
    p = mx.sin(angles) / v0
    x, time = _gradient_forward(v0, gradient, p, depth)
    return {"endpoints": _endpoints(x, depth), "times": time}


def bend_gradient(v0, gradient, offsets, depth):
    offsets = mx.array(offsets, dtype=mx.float32)
    v1 = v0 + gradient * depth
    distance = mx.sqrt(offsets * offsets + depth * depth)
    if gradient == 0.0:
        time = distance / v0
    else:
        # The circular-ray travel time is 2 asinh(y)/|gradient|.
        # log1p retains the homogeneous limit when y is small.
        y = abs(gradient) * distance / (2.0 * (v0 * v1) ** 0.5)
        time = 2.0 * mx.log1p(y + y * y / (mx.sqrt(1.0 + y * y) + 1.0)) / abs(gradient)
    # The circle centre is (v0*cot(angle)/gradient, -v0/gradient).
    # Subtracting its squared distances to the source and receiver gives
    # 2*v0*x*cot(angle) = 2*v0*depth + gradient*(x*x + depth*depth).
    # This form also has the straight-ray limit atan2(x, depth) at g=0.
    angles = mx.arctan2(2.0 * offsets * v0, depth * (v0 + v1) + gradient * offsets * offsets)
    return {"endpoints": _endpoints(offsets, depth), "times": time, "angles": angles}
