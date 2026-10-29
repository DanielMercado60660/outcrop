"""Homogeneous-medium double-difference relocation with damped Gauss--Newton."""

import math

import mlx.core as mx


def relocate(stations, initial_locations, initial_origins, pairs, observed, velocity, damping):
    stations = mx.array(stations, dtype=mx.float32)
    initial_locations = mx.array(initial_locations, dtype=mx.float32)
    initial_origins = mx.array(initial_origins, dtype=mx.float32)
    pairs = mx.array(pairs, dtype=mx.int32)
    observed = mx.array(observed, dtype=mx.float32)
    n = initial_locations.shape[0]
    i, j = pairs[:, 0], pairs[:, 1]
    root_damping = math.sqrt(damping)

    # Location changes are scaled to seconds, matching the origin changes.
    def residual(x):
        x = x.reshape(n, 4)
        locations = initial_locations + velocity * x[:, :3]
        squared_distance = mx.sum(
            (locations[:, None, :] - stations[None, :, :]) ** 2, axis=2
        )
        # Select a finite square-root argument even on the inactive branch:
        # autodiff otherwise encounters the undefined derivative of sqrt(0).
        nonzero = squared_distance > 0
        distance = mx.where(nonzero, mx.sqrt(mx.where(nonzero, squared_distance, 1.0)), 0.0)
        origin_diff = (initial_origins[i] - initial_origins[j]) + (x[i, 3] - x[j, 3])
        return origin_diff[:, None] + (distance[i] - distance[j]) / velocity - observed

    def objective(x):
        r = residual(x)
        return mx.sum(r * r) + damping * mx.sum(x * x)

    objective_grad = mx.grad(objective)
    x = mx.zeros((n, 4), dtype=mx.float32)
    identity = mx.eye(n * 4, dtype=mx.float32)
    event_indices = mx.arange(n)
    mask_i = (i[:, None] == event_indices[None, :]).astype(mx.float32)
    mask_j = (j[:, None] == event_indices[None, :]).astype(mx.float32)

    for _ in range(20):
        locations = initial_locations + velocity * x[:, :3]
        diff = locations[:, None, :] - stations[None, :, :]
        distance = mx.sqrt(mx.sum(diff * diff, axis=2, keepdims=True))
        direction = diff / mx.maximum(distance, 1e-20)
        rays_i = direction[i]
        rays_j = direction[j]
        ones = mx.ones_like(rays_i[:, :, :1])
        event_i = mx.concatenate([rays_i, ones], axis=2)
        event_j = mx.concatenate([rays_j, ones], axis=2)
        jacobian = (event_i[:, :, None, :] * mask_i[:, None, :, None]
                    - event_j[:, :, None, :] * mask_j[:, None, :, None])
        jacobian = jacobian.reshape(-1, n * 4)
        a = mx.concatenate([jacobian, root_damping * identity], axis=0)
        b = -mx.concatenate([residual(x).reshape(-1), root_damping * x.reshape(-1)])
        # The augmented system avoids the precision loss of normal equations.
        q, r = mx.linalg.qr(a, stream=mx.cpu)
        rhs = q[:, :n * 4].T @ b
        step = mx.linalg.solve_triangular(r[:n * 4, :], rhs, upper=True, stream=mx.cpu).reshape(n, 4)
        value = objective(x)
        slope = mx.sum(objective_grad(x) * step)
        candidate = x
        accepted = mx.array(False)
        for k in range(10):
            alpha = 0.5 ** k
            trial = x + alpha * step
            take = (~accepted) & (objective(trial) <= value + 1e-4 * alpha * slope)
            candidate = mx.where(take, trial, candidate)
            accepted = accepted | take
        x = candidate
        mx.eval(x)

    return {
        "locations": (initial_locations + velocity * x[:, :3]).astype(mx.float32),
        "origins": (initial_origins + x[:, 3]).astype(mx.float32),
    }
