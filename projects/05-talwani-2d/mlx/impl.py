"""Gravity of an infinitely long, uniform-density polygon."""

import mlx.core as mx

_GRAVITATIONAL_CONSTANT = 6.67430e-11
_MAX_PAIRS_PER_BATCH = 65536
_MAX_EDGES_PER_BATCH = 4096


@mx.checkpoint
def _edge_sum(vertices, end, observations):
    """Sum a bounded tile, recomputing its intermediates during autodiff."""
    edge = end - vertices
    start = vertices[None, :, :] - observations[:, None, :]
    finish = end[None, :, :] - observations[:, None, :]
    dx = edge[None, :, 0]
    dz = edge[None, :, 1]
    cross = start[:, :, 0] * dz - start[:, :, 1] * dx
    dot = start[:, :, 0] * finish[:, :, 0] + start[:, :, 1] * finish[:, :, 1]
    length_squared = dx * dx + dz * dz
    radius_squared = start[:, :, 0] ** 2 + start[:, :, 1] ** 2
    end_radius_squared = finish[:, :, 0] ** 2 + finish[:, :, 1] ** 2
    at_vertex = (radius_squared == 0) | (end_radius_squared == 0)
    # An incident edge has cross * log(r) -> 0 and cross * angle -> 0.
    # Guard inputs as well as the result: masking an infinite logarithm or
    # atan2(0, 0) afterwards can still contaminate automatic derivatives.
    angle = mx.arctan2(
        mx.where(at_vertex, 0.0, cross), mx.where(at_vertex, 1.0, dot)
    )
    # Compute the squared-radius difference directly to retain small changes.
    radius_difference = dx * (start[:, :, 0] + finish[:, :, 0]) + dz * (
        start[:, :, 1] + finish[:, :, 1]
    )
    increasing = radius_difference >= 0
    smaller_radius_squared = mx.where(increasing, radius_squared, end_radius_squared)
    magnitude = mx.where(increasing, radius_difference, -radius_difference)
    log_ratio = mx.where(increasing, 1.0, -1.0) * mx.log1p(
        mx.where(at_vertex, 0.0, magnitude)
        / mx.where(at_vertex, 1.0, smaller_radius_squared)
    )

    # cross = x_start * dz - z_start * dx; angle is signed from start to end.
    # After endpoint cancellation, twice the boundary integral is
    # sum(cross / length_squared * (2 dx angle - dz log(r_end**2/r_start**2))).
    # The minus sign on the logarithm follows from integrating log(r) dx;
    # using a plus sign cancels the cylinder's gravity instead of adding it.
    edge_integral = (cross / length_squared) * (2.0 * dx * angle - dz * log_ratio)
    return mx.sum(mx.where(at_vertex, 0.0, edge_integral), axis=1)


def polygon_gravity(vertices, observations, density):
    """Calculate downward gravity in mGal for x and positive-down z coordinates.

    Integrating out strike gives ``gz = 2 G rho integral(z / r**2, dA)``
    in observation-relative coordinates. Green's theorem changes the area
    integral to ``-integral(log(r), dx)`` for positive signed polygon area.
    Integrating along each straight edge and cancelling endpoint terms gives
    the angle-minus-logarithm expression above. All geometry stays in MLX.
    """
    observations = mx.array(observations, dtype=mx.float32)
    # Avoid geometry evaluation for an ordinary zero-density call. An MLX
    # scalar must keep the linear density factor, including its slope at zero.
    if observations.shape[0] == 0 or (
        not isinstance(density, mx.array) and density == 0
    ):
        return {"gz": mx.zeros((observations.shape[0],), dtype=mx.float32)}

    vertices = mx.array(vertices, dtype=mx.float32)
    end = mx.concatenate([vertices[1:], vertices[:1]], axis=0)

    # Compute orientation after translation to reduce cancellation.
    shifted = vertices - vertices[:1]
    shifted_end = end - vertices[:1]
    twice_area = mx.sum(
        shifted[:, 0] * shifted_end[:, 1]
        - shifted[:, 1] * shifted_end[:, 0]
    )
    winding = mx.where(twice_area > 0, 1.0, -1.0)
    scale = -winding * (_GRAVITATIONAL_CONSTANT * 1e5) * density

    edge_batch_size = min(vertices.shape[0], _MAX_EDGES_PER_BATCH)
    observation_batch_size = max(1, _MAX_PAIRS_PER_BATCH // edge_batch_size)
    chunks = []
    for first_observation in range(0, observations.shape[0], observation_batch_size):
        batch = observations[
            first_observation:first_observation + observation_batch_size
        ]
        total = None
        for first_edge in range(0, vertices.shape[0], edge_batch_size):
            last_edge = first_edge + edge_batch_size
            partial = _edge_sum(
                vertices[first_edge:last_edge], end[first_edge:last_edge], batch
            )
            total = partial if total is None else total + partial
            if vertices.shape[0] > edge_batch_size:
                # MLX is lazy: evaluate between tiles so forward evaluation
                # releases their intermediates. Checkpointing retains autodiff.
                mx.eval(total)
        gz = (scale * total).astype(mx.float32)
        mx.eval(gz)
        chunks.append(gz)

    return {"gz": chunks[0] if len(chunks) == 1 else mx.concatenate(chunks)}
