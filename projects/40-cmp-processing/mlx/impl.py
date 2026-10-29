"""CMP normal moveout, velocity semblance, and stacking using MLX."""

import mlx.core as mx


def _correct(gather, offsets, dt, velocity):
    """Interpolate each trace at its hyperbolic moveout time."""
    _, n_time = gather.shape
    if n_time == 0:
        return gather
    # Work in sample coordinates: squaring physical times can underflow even
    # when dt is a valid positive float32 (for example, dt = 1e-25).
    time = mx.arange(n_time, dtype=mx.float32)[None, :]
    moveout = mx.abs(offsets[:, None]) / (velocity * dt)
    moveout = mx.where(offsets[:, None] == 0, 0.0, moveout)
    # A moveout of n_time samples is already outside every output window.
    # Bounding it before squaring also handles overflowing offset ratios.
    moveout = mx.minimum(moveout, float(n_time))
    sample = mx.sqrt(time * time + moveout * moveout)
    in_range = sample <= (n_time - 1)
    # Never convert an unbounded index to int32 or feed it into a gather.
    sample = mx.where(in_range, sample, 0.0)
    lower = mx.floor(sample).astype(mx.int32)
    upper = mx.minimum(lower + 1, n_time - 1)
    fraction = sample - lower.astype(mx.float32)
    left = mx.take_along_axis(gather, lower, axis=1)
    right = mx.take_along_axis(gather, upper, axis=1)
    return mx.where(in_range, left + fraction * (right - left), 0.0)


def _window_sum(values, half_window):
    """Inclusive clipped sums in linear work, without prefix subtraction."""
    n_time = values.shape[0]
    width = min(half_window, max(n_time - 1, 0))
    if width == 0:
        return values
    window = 2 * width + 1
    # Each padded window spans at most two blocks of length `window`.
    # Combine a suffix of the first block with a prefix of the second.
    # Unlike a global prefix difference, this preserves quiet-window energy
    # after a strong arrival. Padding stays O(n_time), even for large windows.
    extra = (-(n_time + 2 * width)) % window
    blocks = mx.pad(values, [(width, width + extra)]).reshape(-1, window)
    prefix = mx.cumsum(blocks, axis=1).reshape(-1)
    suffix = mx.cumsum(blocks[:, ::-1], axis=1)[:, ::-1].reshape(-1)
    starts = mx.arange(n_time, dtype=mx.int32)
    right = prefix[window - 1:window - 1 + n_time]
    # A block-aligned window is already complete in its suffix.
    return suffix[:n_time] + mx.where(starts % window == 0, 0.0, right)


def nmo(gather, offsets, dt, velocity):
    gather = mx.asarray(gather, dtype=mx.float32)
    offsets = mx.asarray(offsets, dtype=mx.float32)
    return {"corrected": _correct(gather, offsets, dt, velocity)}


def semblance(gather, offsets, dt, velocities, half_window):
    gather = mx.asarray(gather, dtype=mx.float32)
    offsets = mx.asarray(offsets, dtype=mx.float32)
    velocities = mx.asarray(velocities, dtype=mx.float32)
    n_offset, n_time = gather.shape
    if velocities.shape[0] == 0 or n_time == 0:
        return {"semblance": mx.zeros((velocities.shape[0], n_time), dtype=mx.float32)}
    results = []
    for i in range(velocities.shape[0]):
        corrected = _correct(gather, offsets, dt, velocities[i])
        numerator = mx.sum(corrected, axis=0) ** 2
        denominator = mx.sum(corrected * corrected, axis=0)
        num_window = _window_sum(numerator, half_window)
        den_window = n_offset * _window_sum(denominator, half_window)
        safe_den = mx.where(den_window > 0, den_window, 1.0)
        results.append(mx.where(den_window > 0, num_window / safe_den, 0.0))
    return {"semblance": mx.stack(results, axis=0)}


def stack(gather, offsets, dt, velocity):
    gather = mx.asarray(gather, dtype=mx.float32)
    offsets = mx.asarray(offsets, dtype=mx.float32)
    return {"stack": mx.mean(_correct(gather, offsets, dt, velocity), axis=0)}
