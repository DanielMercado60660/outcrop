"""Two-dimensional acoustic Born propagation in MLX."""

import math

import mlx.core as mx
import numpy as np

_FIRST = {2: (0.5,), 4: (2 / 3, -1 / 12)}
_SECOND = {2: (1.0,), 4: (4 / 3, -1 / 12)}


def _locations(locations, name, shots, ny, nx, points=None):
    """Validate model coordinates before offsetting or narrowing indices."""
    locations = np.asarray(locations)
    if (locations.ndim != 3 or locations.shape[0] != shots
            or locations.shape[2] != 2
            or (points is not None and locations.shape[1] != points)):
        count = "points" if points is None else str(points)
        raise ValueError(f"{name} must have shape [{shots}, {count}, 2]")
    if not np.issubdtype(locations.dtype, np.integer):
        raise ValueError(f"{name} must contain integer grid indices")
    if (np.any(locations < 0) or np.any(locations[..., 0] >= ny)
            or np.any(locations[..., 1] >= nx)):
        raise ValueError(f"{name} indices must satisfy 0 <= y < {ny} and 0 <= x < {nx}")
    return locations


def _profile(n, p, width, dx, dt, freq, vmax, coeffs):
    if width == 0:
        zero = mx.zeros((n,), dtype=mx.float32)
        return zero, zero, zero
    i = np.arange(n, dtype=np.float64)
    fraction = np.maximum(np.clip((p + width - i) / width, 0, 1),
                          np.clip((i - (n - 1 - p - width)) / width, 0, 1))
    sigma = 3 * vmax * math.log(1000) / (2 * width * dx) * fraction**2
    alpha = math.pi * freq * (1 - fraction)
    total = sigma + alpha
    a = np.where(fraction > 0, np.exp(-total * dt), 0)
    ratio = np.divide(sigma, total, out=np.zeros_like(sigma), where=total != 0)
    b = np.where(fraction > 0, ratio * np.expm1(-total * dt), 0)
    db = np.zeros(n, dtype=np.float64)
    for j, c in enumerate(coeffs, 1):
        db[p:n-p] += c * (b[p+j:n-p+j] - b[p-j:n-p-j]) / dx
    return tuple(mx.array(z.astype(np.float32)) for z in (a, b, db))


def _derivatives(q, axis, p, dx, periodic, first, second):
    if periodic:
        d1 = d2 = None
        for j, (c, d) in enumerate(zip(first, second), 1):
            plus = mx.roll(q, -j, axis=axis + 1)
            minus = mx.roll(q, j, axis=axis + 1)
            t1 = c * (plus - minus)
            t2 = d * ((plus - q) + (minus - q))
            d1 = t1 if d1 is None else d1 + t1
            d2 = t2 if d2 is None else d2 + t2
        return d1 / dx, d2 / (dx * dx)
    ny, nx = q.shape[-2:]
    center = q[:, p:ny-p, p:nx-p]
    d1 = d2 = None
    for j, (c, d) in enumerate(zip(first, second), 1):
        if axis == 0:
            plus = q[:, p+j:ny-p+j, p:nx-p]
            minus = q[:, p-j:ny-p-j, p:nx-p]
        else:
            plus = q[:, p:ny-p, p+j:nx-p+j]
            minus = q[:, p:ny-p, p-j:nx-p-j]
        t1 = c * (plus - minus)
        t2 = d * ((plus - center) + (minus - center))
        d1 = t1 if d1 is None else d1 + t1
        d2 = t2 if d2 is None else d2 + t2
    padding = [(0, 0), (p, p), (p, p)]
    return mx.pad(d1 / dx, padding), mx.pad(d2 / (dx * dx), padding)


def _model(v, dv, source_amplitudes, source_locations, receiver_locations,
           wavefield_0, wavefield_m1, wavefield_sc_0, wavefield_sc_m1,
           dx, dt, accuracy, boundary, pml_width, pml_freq, max_vel,
           evaluate=False):
    first, second = _FIRST[accuracy], _SECOND[accuracy]
    p = accuracy // 2
    periodic = boundary == "periodic"
    offset = 0 if periodic else p + pml_width
    v = mx.array(v, dtype=mx.float32)
    ny, nx = v.shape
    source = mx.array(source_amplitudes, dtype=mx.float32)
    shots, nsrc, nt = source.shape
    sloc = _locations(source_locations, "source_locations", shots, ny, nx, nsrc)
    rloc = _locations(receiver_locations, "receiver_locations", shots, ny, nx)
    if nt == 0:
        shape = (shots, rloc.shape[1], 0)
        return mx.zeros(shape, dtype=mx.float32), mx.zeros(shape, dtype=mx.float32)
    if not periodic:
        # The transpose of take sums the clamped copies into model edges.
        iy = mx.array(np.clip(np.arange(-offset, ny + offset), 0, ny - 1).astype(np.int32))
        ix = mx.array(np.clip(np.arange(-offset, nx + offset), 0, nx - 1).astype(np.int32))
        v = mx.take(mx.take(v, iy, axis=0), ix, axis=1)
        dv = mx.take(mx.take(dv, iy, axis=0), ix, axis=1)
    ny, nx = v.shape
    si = mx.array(np.broadcast_to(np.arange(shots, dtype=np.int32)[:, None], sloc.shape[:2]))
    ri = mx.array(np.broadcast_to(np.arange(shots, dtype=np.int32)[:, None], rloc.shape[:2]))
    sy = mx.array((sloc[..., 0] + offset).astype(np.int32))
    sx = mx.array((sloc[..., 1] + offset).astype(np.int32))
    ry = mx.array((rloc[..., 0] + offset).astype(np.int32))
    rx = mx.array((rloc[..., 1] + offset).astype(np.int32))
    padding = [(0, 0), (offset, offset), (offset, offset)]

    def initial(q):
        q = mx.array(q, dtype=mx.float32)
        return q if periodic else mx.pad(q, padding)

    u, old_u = initial(wavefield_0), initial(wavefield_m1)
    s, old_s = initial(wavefield_sc_0), initial(wavefield_sc_m1)
    if not periodic:
        ay, by, dby = _profile(ny, p, pml_width, dx, dt, pml_freq, max_vel, first)
        ax, bx, dbx = _profile(nx, p, pml_width, dx, dt, pml_freq, max_vel, first)
        profiles = [(ay[None, :, None], by[None, :, None], dby[None, :, None]),
                    (ax[None, None, :], bx[None, None, :], dbx[None, None, :])]
        psi_u = [mx.zeros_like(u), mx.zeros_like(u)]
        zeta_u = [mx.zeros_like(u), mx.zeros_like(u)]
        psi_s = [mx.zeros_like(s), mx.zeros_like(s)]
        zeta_s = [mx.zeros_like(s), mx.zeros_like(s)]
    vdt2 = (v * dt) ** 2
    tangent = (2 * dt * dt) * v * dv
    traces_u, traces_s = [], []
    for n in range(nt):
        traces_u.append(u[ri, ry, rx])
        traces_s.append(s[ri, ry, rx])
        hu, hs = [], []
        if not periodic:
            new_pu, new_zu, new_ps, new_zs = [], [], [], []
        for axis in range(2):
            du, lu = _derivatives(u, axis, p, dx, periodic, first, second)
            ds, ls = _derivatives(s, axis, p, dx, periodic, first, second)
            if periodic:
                hu.append(lu)
                hs.append(ls)
            else:
                a, b, db = profiles[axis]
                apu, aps = a * psi_u[axis], a * psi_s[axis]
                dapu, _ = _derivatives(apu, axis, p, dx, False, first, second)
                daps, _ = _derivatives(aps, axis, p, dx, False, first, second)
                tu = (1 + b) * lu + db * du + dapu
                ts = (1 + b) * ls + db * ds + daps
                hu.append((1 + b) * tu + a * zeta_u[axis])
                hs.append((1 + b) * ts + a * zeta_s[axis])
                new_pu.append(b * du + apu)
                new_ps.append(b * ds + aps)
                new_zu.append(b * tu + a * zeta_u[axis])
                new_zs.append(b * ts + a * zeta_s[axis])
        next_u = 2 * u - old_u + vdt2 * (hu[0] + hu[1])
        next_s = 2 * s - old_s + vdt2 * (hs[0] + hs[1]) + tangent * (hu[0] + hu[1])
        if nsrc:
            next_u = next_u.at[si, sy, sx].add(-vdt2[sy, sx] * source[:, :, n])
            next_s = next_s.at[si, sy, sx].add(-tangent[sy, sx] * source[:, :, n])
        old_u, u = u, next_u
        old_s, s = s, next_s
        if not periodic:
            psi_u, zeta_u, psi_s, zeta_s = new_pu, new_zu, new_ps, new_zs
        if evaluate:
            state = [u, old_u, s, old_s, traces_u[-1], traces_s[-1]]
            if not periodic:
                state += psi_u + zeta_u + psi_s + zeta_s
            mx.eval(*state)
    return mx.stack(traces_u, axis=-1), mx.stack(traces_s, axis=-1)


def born(v, dv, source_amplitudes, source_locations, receiver_locations,
         wavefield_0, wavefield_m1, wavefield_sc_0, wavefield_sc_m1,
         dx, dt, accuracy, boundary, pml_width, pml_freq, max_vel):
    background, scattered = _model(
        v, mx.array(dv, dtype=mx.float32), source_amplitudes, source_locations, receiver_locations,
        wavefield_0, wavefield_m1, wavefield_sc_0, wavefield_sc_m1,
        dx, dt, accuracy, boundary, pml_width, pml_freq, max_vel, evaluate=True)
    return {"background": background, "scattered": scattered}


def misfit_and_grad(v, dv, source_amplitudes, source_locations,
                    receiver_locations, wavefield_0, wavefield_m1,
                    wavefield_sc_0, wavefield_sc_m1, dx, dt, accuracy,
                    boundary, pml_width, pml_freq, max_vel, observed):
    observed = mx.array(observed, dtype=mx.float32)

    def objective(direction):
        _, scattered = _model(
            v, direction, source_amplitudes, source_locations, receiver_locations,
            wavefield_0, wavefield_m1, wavefield_sc_0, wavefield_sc_m1,
            dx, dt, accuracy, boundary, pml_width, pml_freq, max_vel)
        residual = scattered - observed
        return 0.5 * mx.sum(residual * residual)

    value, gradient = mx.value_and_grad(objective)(mx.array(dv, dtype=mx.float32))
    return {"objective": value, "gradient": gradient}
