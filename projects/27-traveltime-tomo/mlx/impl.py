"""Straight-ray, cellwise slowness tomography implemented with MLX."""

import mlx.core as mx


def _axis_intervals(start, direction, edges):
    """Parameter intervals where each segment lies inside each axis cell."""
    lower = edges[:-1][None, :]
    upper = edges[1:][None, :]
    p = start[:, None]
    d = direction[:, None]
    moving = d != 0
    safe_d = mx.where(moving, d, 1.0)
    a = (lower - p) / safe_d
    b = (upper - p) / safe_d
    lo = mx.minimum(a, b)
    hi = mx.maximum(a, b)
    # Internal boundaries belong to the cell on the positive side.
    last = mx.arange(edges.shape[0] - 1)[None, :] == edges.shape[0] - 2
    inside = (p >= lower) & ((p < upper) | (last & (p <= upper)))
    lo = mx.where(moving, lo, mx.where(inside, 0.0, 1.0))
    hi = mx.where(moving, hi, mx.where(inside, 1.0, 0.0))
    return lo, hi


def ray_matrix(x_edges, z_edges, sources, receivers):
    x = mx.array(x_edges)
    z = mx.array(z_edges)
    source = mx.array(sources)
    delta = mx.array(receivers) - source
    xlo, xhi = _axis_intervals(source[:, 0], delta[:, 0], x)
    zlo, zhi = _axis_intervals(source[:, 1], delta[:, 1], z)
    lo = mx.maximum(mx.maximum(xlo[:, None, :], zlo[:, :, None]), 0.0)
    hi = mx.minimum(mx.minimum(xhi[:, None, :], zhi[:, :, None]), 1.0)
    length = mx.sqrt(mx.sum(delta * delta, axis=1))
    segments = mx.maximum(hi - lo, 0.0) * length[:, None, None]
    return {"matrix": segments.reshape((source.shape[0], -1))}


def solve_tomography(matrix, travel_times, reference_slowness, damping):
    g = mx.array(matrix)
    t = mx.array(travel_times)
    prior = mx.array(reference_slowness)
    residual = t - g @ prior
    ncells = g.shape[1]
    augmented = mx.concatenate((g, damping * mx.eye(ncells)), axis=0)
    rhs = mx.concatenate((residual, mx.zeros((ncells,))))
    q, r = mx.linalg.qr(augmented, stream=mx.cpu)
    # Accommodate either reduced or full QR factors.
    projected = q[:, :ncells].T @ rhs
    update = mx.linalg.solve_triangular(
        r[:ncells, :], projected, upper=True, stream=mx.cpu
    )
    slowness = prior + update
    return {"slowness": slowness, "predicted_times": g @ slowness}


def misfit_and_grad(matrix, travel_times, slowness, reference_slowness, damping):
    g = mx.array(matrix)
    t = mx.array(travel_times)
    prior = mx.array(reference_slowness)
    s = mx.array(slowness)

    def objective(model):
        residual = g @ model - t
        update = model - prior
        return 0.5 * (mx.sum(residual * residual)
                      + (damping * damping) * mx.sum(update * update))

    value, gradient = mx.value_and_grad(objective)(s)
    return {"objective": value, "gradient": gradient}
