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


def scalar(velocity, source, source_locations, receiver_locations, spacing,
           dt, order, pml_width, pml_freq, max_vel):
    first, second = _FIRST[order], _SECOND[order]
    p = order // 2
    low_z, high_z, low_x, high_x = pml_width
    v = mx.pad(mx.array(velocity), [(p + low_z, p + high_z),
                                    (p + low_x, p + high_x)], mode="edge")
    nz, nx = v.shape
    source = mx.array(source)
    shots, _, nt = source.shape
    sloc = np.asarray(source_locations)
    rloc = np.asarray(receiver_locations)
    if nt == 0:
        return {"receiver_amplitudes": mx.zeros((shots, rloc.shape[1], 0),
                                               dtype=mx.float32)}
    src_shot = mx.array(np.broadcast_to(np.arange(shots, dtype=np.int32)[:, None],
                                        sloc.shape[:2]))
    rec_shot = mx.array(np.broadcast_to(np.arange(shots, dtype=np.int32)[:, None],
                                        rloc.shape[:2]))
    sz = mx.array((sloc[..., 0] + p + low_z).astype(np.int32))
    sx = mx.array((sloc[..., 1] + p + low_x).astype(np.int32))
    rz = mx.array((rloc[..., 0] + p + low_z).astype(np.int32))
    rx = mx.array((rloc[..., 1] + p + low_x).astype(np.int32))
    az, bz, dbz = _profile(nz, p, low_z, high_z, spacing, dt,
                            pml_freq, max_vel, first)
    ax, bx, dbx = _profile(nx, p, low_x, high_x, spacing, dt,
                            pml_freq, max_vel, first)
    profiles = [(az[None, :, None], bz[None, :, None], dbz[None, :, None]),
                (ax[None, None, :], bx[None, None, :], dbx[None, None, :])]
    vdt2 = (v * dt) ** 2
    source_scale = vdt2[sz, sx]
    u = mx.zeros((shots, nz, nx), dtype=mx.float32)
    previous = mx.zeros_like(u)
    psi = [mx.zeros_like(u), mx.zeros_like(u)]
    zeta = [mx.zeros_like(u), mx.zeros_like(u)]
    traces = []
    for n in range(nt):
        trace = u[rec_shot, rz, rx]
        traces.append(trace)
        contributions = []
        next_psi = []
        next_zeta = []
        for axis, (a, b, db) in enumerate(profiles):
            grad, lap = _derivatives(u, axis, p, spacing, first, second)
            apsi = a * psi[axis]
            t = (1 + b) * lap + db * grad + _first_derivative(
                apsi, axis, p, spacing, first)
            next_psi.append(b * grad + apsi)
            next_zeta.append(b * t + a * zeta[axis])
            contributions.append((1 + b) * t + a * zeta[axis])
        following = 2 * u - previous + vdt2 * (contributions[0] + contributions[1])
        following = following.at[src_shot, sz, sx].add(-source_scale * source[:, :, n])
        previous, u = u, following
        psi, zeta = next_psi, next_zeta
        # Bound the lazy graph and retain each receiver sample before advancing.
        mx.eval(u, previous, *psi, *zeta, trace)
    return {"receiver_amplitudes": mx.stack(traces, axis=-1)}
