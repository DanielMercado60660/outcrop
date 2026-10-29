"""Homogeneous arrival-time location using MLX float32 operations."""

import mlx.core as mx

_CHI2_95 = 5.991464547107979
# Bound each event/node/station tensor to about 4 MiB of float32 values.
_GRID_WORK_ITEMS = 1 << 20
_GRID_EVENT_CHUNK = 64
_GRID_NODE_CHUNK = 4096


def _predict(stations, models, velocity):
    offset = models[:, None, :3] - stations[None, :, :]
    distance = mx.sqrt(mx.sum(offset * offset, axis=-1))
    return models[:, 3, None] + distance / velocity


def _loss(stations, arrivals, sigma, velocity, models):
    residual = (_predict(stations, models, velocity) - arrivals) / sigma
    return 0.5 * mx.sum(residual * residual, axis=-1)


def _relative_precision(sigma):
    # A common factor cancels from weighted means and Gauss-Newton steps.
    # Form the ratio before squaring: at least one precision is one and all
    # are <= 1, even when sigma**2 would overflow or underflow in float32.
    scale = mx.min(sigma)
    return scale, scale / sigma


def _scaled_loss(stations, arrivals, precision, velocity, models):
    # Use the scale-independent loss for search/step comparisons; return the
    # actual half chi-square through _loss with the original sigma instead.
    residual = (_predict(stations, models, velocity) - arrivals) * precision
    return 0.5 * mx.sum(residual * residual, axis=-1)


def misfit_and_grad(stations, arrivals, sigma, velocity, models):
    stations = mx.array(stations)
    arrivals = mx.array(arrivals)
    sigma = mx.array(sigma)
    models = mx.array(models)

    def total_loss(m):
        return mx.sum(_loss(stations, arrivals, sigma, velocity, m))

    gradient = mx.grad(total_loss)(models)
    return {
        "predicted": _predict(stations, models, velocity),
        "objective": _loss(stations, arrivals, sigma, velocity, models),
        "gradient": gradient,
    }


def grid_search(stations, arrivals, sigma, velocity, x, y, z):
    stations = mx.array(stations)
    arrivals = mx.array(arrivals)
    sigma = mx.array(sigma)
    x, y, z = mx.array(x), mx.array(y), mx.array(z)
    _, precision = _relative_precision(sigma)
    weight = precision * precision
    weight_sum = mx.sum(weight)
    batch, station_count = arrivals.shape
    if batch == 0:
        return {"models": mx.zeros((0, 4), dtype=arrivals.dtype),
                "objective": mx.zeros((0,), dtype=arrivals.dtype)}
    node_count = x.size * y.size * z.size
    if node_count == 0:
        raise ValueError("grid axes must be nonempty")
    event_chunk = max(1, min(batch, _GRID_EVENT_CHUNK,
                             _GRID_WORK_ITEMS // station_count))
    node_chunk = max(1, min(_GRID_NODE_CHUNK,
                            _GRID_WORK_ITEMS // (event_chunk * station_count)))
    fitted = []
    for event_start in range(0, batch, event_chunk):
        picks = arrivals[event_start:event_start + event_chunk]
        best_score = None
        best_models = None
        for node_start in range(0, node_count, node_chunk):
            node_stop = min(node_start + node_chunk, node_count)
            flat = mx.arange(node_start, node_stop)
            # Decode only this chunk, in C order (z fastest).
            nodes = mx.stack([x[flat // (y.size * z.size)],
                              y[(flat // z.size) % y.size],
                              z[flat % z.size]], axis=-1)
            offset = nodes[:, None, :] - stations[None, :, :]
            travel = mx.sqrt(mx.sum(offset * offset, axis=-1)) / velocity
            t0 = mx.sum((picks[:, None, :] - travel[None, :, :]) * weight,
                        axis=-1) / weight_sum
            residual = (t0[:, :, None] + travel[None, :, :]
                        - picks[:, None, :]) * precision
            score = 0.5 * mx.sum(residual * residual, axis=-1)
            chosen_score = mx.min(score, axis=1)
            # Explicit first-index reduction also resolves ties within a chunk.
            local = mx.arange(node_stop - node_start)
            index = mx.min(mx.where(score == chosen_score[:, None], local,
                                    node_stop - node_start), axis=1)
            chosen_t0 = mx.take_along_axis(t0, index[:, None], axis=1)
            chosen_models = mx.concatenate([nodes[index], chosen_t0], axis=1)
            if best_score is None:
                best_score, best_models = chosen_score, chosen_models
            else:
                # Strict improvement preserves the earlier chunk on ties.
                improve = chosen_score < best_score
                best_score = mx.where(improve, chosen_score, best_score)
                best_models = mx.where(improve[:, None], chosen_models, best_models)
            # Complete this chunk so the lazy graph cannot retain the entire
            # search. Only each event's current best candidate survives.
            mx.eval(best_score, best_models)
        fitted.append(best_models)
    models = mx.concatenate(fitted, axis=0)
    return {"models": models,
            "objective": _loss(stations, arrivals, sigma, velocity, models)}


def _inverse(matrix):
    """Batched Gauss-Jordan inverse of a positive definite matrix."""
    batch = matrix.shape[0]
    size = matrix.shape[1]
    identity = mx.broadcast_to(mx.eye(size, dtype=matrix.dtype),
                               (batch, size, size))
    rows = [mx.concatenate([matrix[:, i, :], identity[:, i, :]], axis=-1)
            for i in range(size)]
    for i in range(size):
        pivot = rows[i] / rows[i][:, i:i + 1]
        rows = [pivot if j == i else rows[j] - rows[j][:, i:i + 1] * pivot
                for j in range(size)]
    inverse = mx.stack([row[:, size:] for row in rows], axis=1)
    return 0.5 * (inverse + mx.swapaxes(inverse, 1, 2))


def _centered_system(stations, precision, models):
    # Eliminate the intercept by centering the spatial derivatives before
    # forming their normal matrix. This avoids subtracting two nearly equal
    # large terms when depth and origin time are strongly correlated.
    offset = models[:, None, :3] - stations[None, :, :]
    distance = mx.sqrt(mx.sum(offset * offset, axis=-1))
    direction = offset / distance[:, :, None]
    weight = precision * precision
    weight_sum = mx.sum(weight)
    mean = mx.sum(direction * weight[None, :, None], axis=1) / weight_sum
    centered = direction - mean[:, None, :]
    # These tiny contractions need full float32 products, independently of
    # the precision used by the device's matrix-multiplication kernels.
    # Explicit outer products also give an exactly symmetric normal matrix.
    weighted = centered * precision[None, :, None]
    normal = mx.sum(weighted[:, :, :, None] * weighted[:, :, None, :], axis=1)
    return distance, weight, weight_sum, mean, weighted, normal


def _geiger_step(stations, arrivals, precision, velocity, models):
    distance, weight, weight_sum, mean, weighted, normal = _centered_system(
        stations, precision, models)
    residual = models[:, 3, None] + distance / velocity - arrivals
    mean_residual = mx.sum(residual * weight, axis=1) / weight_sum
    centered_residual = (residual - mean_residual[:, None]) * precision
    rhs = mx.sum(weighted * centered_residual[:, :, None], axis=1)
    # Solve the Schur complement in coordinates [x/v, y/v, z/v].
    # Recover the joint origin-time step after eliminating its column.
    spatial_step = mx.sum(_inverse(normal) * rhs[:, None, :], axis=-1)
    time_step = mean_residual - mx.sum(mean * spatial_step, axis=-1)
    return mx.concatenate([velocity * spatial_step, time_step[:, None]], axis=1)


def _covariance(stations, precision, scale, velocity, models):
    _, _, weight_sum, mean, _, normal = _centered_system(stations, precision, models)
    spatial = _inverse(normal)
    cross = -mx.sum(spatial * mean[:, None, :], axis=-1, keepdims=True)
    time = 1.0 / weight_sum - mx.sum(mean[:, :, None] * cross, axis=1)
    top = mx.concatenate([spatial * (velocity * velocity), cross * velocity], axis=2)
    bottom = mx.concatenate([mx.swapaxes(cross, 1, 2) * velocity,
                             time[:, None, :]], axis=2)
    covariance = mx.concatenate([top, bottom], axis=1)
    # Undo only the common weight normalization, using the known sigma.
    # Multiply sequentially instead of forming scale**2, which can overflow
    # or underflow even when a covariance entry is representable.
    return (covariance * scale) * scale


def geiger(stations, arrivals, sigma, velocity, initial):
    stations = mx.array(stations)
    arrivals = mx.array(arrivals)
    sigma = mx.array(sigma)
    models = mx.array(initial)
    scale, precision = _relative_precision(sigma)
    best_loss = _scaled_loss(stations, arrivals, precision, velocity, models)
    # Use decreasing loss for the initial approach to the local minimum.
    for _ in range(15):
        step = _geiger_step(stations, arrivals, precision, velocity, models)
        candidate = models - step
        candidate_loss = _scaled_loss(stations, arrivals, precision, velocity, candidate)
        improve = candidate_loss < best_loss
        models = mx.where(improve[:, None], candidate, models)
        best_loss = mx.where(improve, candidate_loss, best_loss)
    # Near a noisy solution the improvement is quadratic in the step and can
    # be smaller than loss roundoff while the model still affects covariance.
    # Continue solving the linearized stationarity equations without a loss
    # acceptance test. The supplied starts lie in the local convergence basin.
    for _ in range(10):
        models = models - _geiger_step(stations, arrivals, precision, velocity, models)
    covariance = _covariance(stations, precision, scale, velocity, models)
    return {"models": models,
            "objective": _loss(stations, arrivals, sigma, velocity, models),
            "covariance": covariance,
            "ellipse": _CHI2_95 * covariance[:, :2, :2]}
