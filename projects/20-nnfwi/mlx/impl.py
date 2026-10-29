"""Second-order-in-time acoustic propagation with convolutional PML."""

import math

import mlx.core as mx
import numpy as np

_FIRST = {
    2: (1 / 2,),
    4: (2 / 3, -1 / 12),
    6: (3 / 4, -3 / 20, 1 / 60),
    8: (4 / 5, -1 / 5, 4 / 105, -1 / 280),
}
_SECOND = {
    2: (1,),
    4: (4 / 3, -1 / 12),
    6: (3 / 2, -3 / 20, 1 / 90),
    8: (8 / 5, -1 / 5, 8 / 315, -1 / 560),
}


def _profile(n, p, low, high, h, dt, freq, vmax, coeffs):
    indices = np.arange(n, dtype=np.float64)
    qlow = np.clip((p + low - indices) / low, 0, 1) if low else np.zeros(n)
    qhigh = (np.clip((indices - (n - 1 - p - high)) / high, 0, 1)
             if high else np.zeros(n))
    q = np.maximum(qlow, qhigh)
    scale = 3 * vmax * math.log(1000) / (2 * h)
    sigma = np.zeros(n, dtype=np.float64)
    if low:
        sigma += scale * qlow**2 / low
    if high:
        sigma += scale * qhigh**2 / high
    alpha = math.pi * freq * (1 - q)
    total = sigma + alpha
    a = np.where(q > 0, np.exp(-total * dt), 0)
    ratio = np.divide(sigma, total, out=np.zeros_like(sigma), where=total != 0)
    b = np.where(q > 0, ratio * np.expm1(-total * dt), 0)
    db = np.zeros(n, dtype=np.float64)
    for j, coefficient in enumerate(coeffs, 1):
        db[p:-p] += coefficient * (b[p + j:n - p + j] - b[p - j:n - p - j]) / h
    return tuple(mx.array(x.astype(np.float32)) for x in (a, b, db))


def _derivatives(field, axis, p, h, first, second):
    """Return full-size D1 and D2 with zero halo."""
    nz, nx = field.shape[-2:]
    center = field[:, p:nz - p, p:nx - p]
    d1 = None
    d2 = None
    for j, (c, d) in enumerate(zip(first, second), 1):
        if axis == 0:
            plus = field[:, p + j:nz - p + j, p:nx - p]
            minus = field[:, p - j:nz - p - j, p:nx - p]
        else:
            plus = field[:, p:nz - p, p + j:nx - p + j]
            minus = field[:, p:nz - p, p - j:nx - p - j]
        term1 = c * (plus - minus)
        term2 = d * ((plus - center) + (minus - center))
        d1 = term1 if d1 is None else d1 + term1
        d2 = term2 if d2 is None else d2 + term2
    padding = [(0, 0), (p, p), (p, p)]
    return mx.pad(d1 / h, padding), mx.pad(d2 / (h * h), padding)


def _first_derivative(field, axis, p, h, coeffs):
    nz, nx = field.shape[-2:]
    result = None
    for j, c in enumerate(coeffs, 1):
        if axis == 0:
            plus = field[:, p + j:nz - p + j, p:nx - p]
            minus = field[:, p - j:nz - p - j, p:nx - p]
        else:
            plus = field[:, p:nz - p, p + j:nx - p + j]
            minus = field[:, p:nz - p, p - j:nx - p - j]
        term = c * (plus - minus)
        result = term if result is None else result + term
    return mx.pad(result / h, [(0, 0), (p, p), (p, p)])


def _resize(image, shape):
    cz, cx = image.shape
    nz, nx = shape
    z = np.linspace(0, cz - 1, nz, dtype=np.float64)
    x = np.linspace(0, cx - 1, nx, dtype=np.float64)
    # Dense interpolation weights give the same bilinear map while making
    # its transpose a reduction, rather than a scatter-add into coarse nodes.
    wz = mx.array(np.maximum(1 - np.abs(z[:, None] - np.arange(cz)), 0)
                  .astype(np.float32))
    wx = mx.array(np.maximum(1 - np.abs(x[:, None] - np.arange(cx)), 0)
                  .astype(np.float32))
    rows = mx.sum(wz[:, :, None] * image[None, :, :], axis=1)
    return mx.sum(rows[:, None, :] * wx[None, :, :], axis=2)


def _decode(parameters, latent, initial, scale, coarse_shape, hidden_size, mode):
    if mode == "grid":
        q = parameters.reshape(initial.shape)
    elif mode == "neural":
        length = latent.shape[0]
        h = hidden_size
        count = coarse_shape[0] * coarse_shape[1]
        k = h * length
        w1 = parameters[:k].reshape(h, length)
        b1 = parameters[k:k + h]
        k += h
        w2 = parameters[k:k + count * h].reshape(count, h)
        k += count * h
        b2 = parameters[k:k + count]
        hidden = mx.tanh(mx.sum(w1 * latent[None, :], axis=1) / math.sqrt(length) + b1)
        coarse = (mx.sum(w2 * hidden[None, :], axis=1) / math.sqrt(h) + b2)
        coarse = coarse.reshape(tuple(coarse_shape))
        q = _resize(coarse, initial.shape)
    else:
        raise ValueError("unknown parameterization")
    return initial + scale * mx.tanh(q)


def _extend(v, width):
    v = mx.concatenate([mx.broadcast_to(v[:1, :], (width, v.shape[1])), v,
                        mx.broadcast_to(v[-1:, :], (width, v.shape[1]))], axis=0)
    return mx.concatenate([mx.broadcast_to(v[:, :1], (v.shape[0], width)), v,
                           mx.broadcast_to(v[:, -1:], (v.shape[0], width))], axis=1)


def _make_problem(latent, initial_velocity, observed, source_amplitudes,
                  source_locations, receiver_locations, spacing, dt, data_scale,
                  velocity_scale, coarse_shape, hidden_size, parameterization):
    latent, initial = mx.array(latent), mx.array(initial_velocity)
    observed, source = mx.array(observed), np.asarray(source_amplitudes)
    shots, _, nt = source.shape
    p, offset = 2, 14
    nz, nx = initial.shape[0] + 2 * offset, initial.shape[1] + 2 * offset
    sloc, rloc = np.asarray(source_locations), np.asarray(receiver_locations)
    # Merge coincident sources once, in source order, using only fixed inputs.
    # Include the shot in each flattened index so shots never mix. The unique
    # indices make the per-step scatter collision-free and deterministic.
    cells = ((np.arange(shots, dtype=np.int64)[:, None] * nz
              + sloc[..., 0] + offset) * nx + sloc[..., 1] + offset)
    cells, inverse = np.unique(cells.reshape(-1), return_inverse=True)
    amplitudes = np.zeros((cells.size, nt), dtype=np.float32)
    np.add.at(amplitudes, inverse, source.reshape(shots * source.shape[1], nt))
    source_indices = mx.array(cells.astype(np.int32))
    source_values = mx.array(np.ascontiguousarray(amplitudes.T))
    rs = mx.array(np.broadcast_to(np.arange(shots, dtype=np.int32)[:, None], rloc.shape[:2]))
    rz = mx.array((rloc[..., 0] + offset).astype(np.int32))
    rx = mx.array((rloc[..., 1] + offset).astype(np.int32))
    first, second = _FIRST[4], _SECOND[4]
    az, bz, dbz = _profile(nz, p, 12, 12, spacing, dt, 18.0, 2600.0, first)
    ax, bx, dbx = _profile(nx, p, 12, 12, spacing, dt, 18.0, 2600.0, first)
    profiles = [(az[None, :, None], bz[None, :, None], dbz[None, :, None]),
                (ax[None, None, :], bx[None, None, :], dbx[None, None, :])]

    def decode(parameters):
        return _decode(parameters, latent, initial, velocity_scale, coarse_shape,
                       hidden_size, parameterization)

    def forward(parameters):
        velocity = decode(parameters)
        vdt2 = (_extend(velocity, offset) * dt) ** 2
        u = mx.zeros((shots, nz, nx), dtype=mx.float32)
        previous = mx.zeros_like(u)
        psi = [mx.zeros_like(u), mx.zeros_like(u)]
        zeta = [mx.zeros_like(u), mx.zeros_like(u)]
        traces = []
        for n in range(nt):
            traces.append(u[rs, rz, rx])
            contributions, next_psi, next_zeta = [], [], []
            for axis, (a, b, db) in enumerate(profiles):
                grad, lap = _derivatives(u, axis, p, spacing, first, second)
                apsi = a * psi[axis]
                t = (1 + b) * lap + db * grad + _first_derivative(
                    apsi, axis, p, spacing, first)
                next_psi.append(b * grad + apsi)
                next_zeta.append(b * t + a * zeta[axis])
                contributions.append((1 + b) * t + a * zeta[axis])
            injection = mx.zeros((shots * nz * nx,), dtype=mx.float32)
            if source_indices.size:
                injection = injection.at[source_indices].add(source_values[n])
            injection = injection.reshape(shots, nz, nx)
            following = 2 * u - previous + vdt2 * (
                contributions[0] + contributions[1] - injection)
            previous, u = u, following
            psi, zeta = next_psi, next_zeta
        predicted = mx.stack(traces, axis=-1)
        residual = (predicted - observed) / data_scale
        return 0.5 * mx.mean(residual * residual), predicted, velocity

    return forward, decode


def misfit_and_grad(*, parameters, latent, initial_velocity, observed,
                    source_amplitudes, source_locations, receiver_locations,
                    spacing, dt, data_scale, velocity_scale, coarse_shape,
                    hidden_size, parameterization):
    forward, _ = _make_problem(latent, initial_velocity, observed, source_amplitudes,
                               source_locations, receiver_locations, spacing, dt,
                               data_scale, velocity_scale, coarse_shape, hidden_size,
                               parameterization)

    def loss(p):
        return forward(p)[0]

    p = mx.array(parameters)
    objective, gradient = mx.value_and_grad(loss)(p)
    _, predicted, velocity = forward(p)
    mx.eval(objective, gradient, predicted, velocity)
    return {"objective": np.asarray(objective), "gradient": np.asarray(gradient),
            "predicted": np.asarray(predicted), "velocity": np.asarray(velocity)}


def invert(*, parameters, latent, initial_velocity, observed, source_amplitudes,
           source_locations, receiver_locations, spacing, dt, data_scale,
           velocity_scale, coarse_shape, hidden_size, parameterization, steps,
           learning_rate):
    forward, decode = _make_problem(latent, initial_velocity, observed, source_amplitudes,
                                    source_locations, receiver_locations, spacing, dt,
                                    data_scale, velocity_scale, coarse_shape, hidden_size,
                                    parameterization)

    def loss(p):
        return forward(p)[0]

    value_grad = mx.value_and_grad(loss)
    p = mx.array(parameters)
    m, s = mx.zeros_like(p), mx.zeros_like(p)
    history = []
    for t in range(1, steps + 1):
        value, grad = value_grad(p)
        m = 0.9 * m + 0.1 * grad
        s = 0.999 * s + 0.001 * grad * grad
        p = p - learning_rate * (m / (1 - 0.9 ** t)) / (
            mx.sqrt(s / (1 - 0.999 ** t)) + 1e-8)
        mx.eval(value, p, m, s)
        history.append(value)
    history.append(loss(p))
    velocity = decode(p)
    history = mx.stack(history)
    mx.eval(p, velocity, history)
    return {"parameters": np.asarray(p), "velocity": np.asarray(velocity),
            "loss_history": np.asarray(history)}
