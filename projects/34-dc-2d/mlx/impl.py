"""Cell-centred 2.5D DC forward modelling using matrix-free MLX CG."""

import mlx.core as mx
import numpy as np
from scipy.special import k0e, k1e


def _apply(x, diagonal, gx, gz):
    xl = mx.concatenate((mx.zeros_like(x[:, :1]), x[:, :-1]), axis=1)
    xr = mx.concatenate((x[:, 1:], mx.zeros_like(x[:, :1])), axis=1)
    xd = mx.concatenate((mx.zeros_like(x[:, :, :1]), x[:, :, :-1]), axis=2)
    xu = mx.concatenate((x[:, :, 1:], mx.zeros_like(x[:, :, :1])), axis=2)
    zx = mx.zeros((1, gx.shape[1]), dtype=gx.dtype)
    zz = mx.zeros((gz.shape[0], 1), dtype=gz.dtype)
    gl = mx.concatenate((zx, gx), axis=0)[None, :, :, None]
    gr = mx.concatenate((gx, zx), axis=0)[None, :, :, None]
    gd = mx.concatenate((zz, gz), axis=1)[None, :, :, None]
    gu = mx.concatenate((gz, zz), axis=1)[None, :, :, None]
    return diagonal * x - gl * xl - gr * xr - gd * xd - gu * xu


@mx.compile
def _cg_step(x, r, p, rz, diagonal, gx, gz, threshold):
    ap = _apply(p, diagonal, gx, gz)
    pap = mx.sum(p * ap, axis=(1, 2), keepdims=True)
    active = rz > threshold
    alpha = mx.where(active, rz / mx.maximum(pap, 1e-30), 0.0)
    x = x + alpha * p
    r = r - alpha * ap
    z = r / diagonal
    rz_new = mx.sum(r * z, axis=(1, 2), keepdims=True)
    beta = mx.where(active, rz_new / mx.maximum(rz, 1e-30), 0.0)
    p = z + beta * p
    return x, r, p, rz_new


def _boundary_coefficients(hx, hz, origin, ky):
    """Robin conductance per unit conductivity on side and bottom faces."""
    xn = origin[0] + np.r_[0.0, np.cumsum(hx)]
    zn = origin[1] + np.r_[0.0, np.cumsum(hz)]
    xc = (xn[:-1] + xn[1:]) * 0.5
    zc = (zn[:-1] + zn[1:]) * 0.5
    midpoint = 0.5 * (xn[0] + xn[-1])
    top = zn[-1]

    def alpha(dx, dz, projection):
        radius = np.sqrt(dx * dx + dz * dz)
        kr = ky[:, None] * radius[None, :]
        return ky[:, None] * (k1e(kr) / k0e(kr)) * projection[None, :] / radius[None, :]

    left_dist = abs(xn[0] - midpoint)
    right_dist = abs(xn[-1] - midpoint)
    bottom_dist = top - zn[0]
    al = alpha(np.full_like(zc, left_dist), zc - top, np.full_like(zc, left_dist))
    ar = alpha(np.full_like(zc, right_dist), zc - top, np.full_like(zc, right_dist))
    ab = alpha(xc - midpoint, np.full_like(xc, bottom_dist), np.full_like(xc, bottom_dist))
    left = hz[None, :] * al / (1.0 + 0.5 * hx[0] * al)
    right = hz[None, :] * ar / (1.0 + 0.5 * hx[-1] * ar)
    bottom = hx[None, :] * ab / (1.0 + 0.5 * hz[0] * ab)
    out = np.zeros((len(ky), len(hx), len(hz)), dtype=np.float32)
    out[:, 0, :] += left
    out[:, -1, :] += right
    out[:, :, 0] += bottom
    return out


def _receiver_matrix(xc, positions):
    """Linear interpolation between cell centres, constant beyond endpoints."""
    matrix = np.zeros((len(positions), len(xc)), dtype=np.float32)
    for row, position in enumerate(positions):
        hi = np.searchsorted(xc, position, side="right")
        if hi == 0:
            matrix[row, 0] = 1.0
        elif hi == len(xc):
            matrix[row, -1] = 1.0
        else:
            t = (position - xc[hi - 1]) / (xc[hi] - xc[hi - 1])
            matrix[row, hi - 1] = 1.0 - t
            matrix[row, hi] = t
    return matrix


def dc_forward(hx, hz, origin, conductivity, ab, mn, currents, ky, weights):
    # Host setup in float32 (precision policy): a float64 copy changes the Robin coefficients by <= 3.3e-7 relative.
    # Geometry is host-side setup; conductivity and the solve stay in MLX.
    hx_np = np.asarray(hx, dtype=np.float32)
    hz_np = np.asarray(hz, dtype=np.float32)
    origin_np = np.asarray(origin, dtype=np.float32)
    ky_np = np.asarray(ky, dtype=np.float32)
    ab_np = np.asarray(ab, dtype=np.float32)
    mn_np = np.asarray(mn, dtype=np.float32)
    nx, nz = len(hx_np), len(hz_np)
    n = len(ab_np)
    xc = origin_np[0] + np.cumsum(hx_np) - 0.5 * hx_np

    sigma = mx.array(conductivity, dtype=mx.float32)
    hx_m = mx.array(hx_np.astype(np.float32))
    hz_m = mx.array(hz_np.astype(np.float32))
    k = mx.array(ky_np.astype(np.float32))
    # Integral face resistivity: half-cell widths divided by conductivity.
    gx = hz_m[None, :] / (0.5 * hx_m[:-1, None] / sigma[:-1, :]
                           + 0.5 * hx_m[1:, None] / sigma[1:, :])
    gz = hx_m[:, None] / (0.5 * hz_m[None, :-1] / sigma[:, :-1]
                           + 0.5 * hz_m[None, 1:] / sigma[:, 1:])
    zx = mx.zeros((1, nz), dtype=mx.float32)
    zz = mx.zeros((nx, 1), dtype=mx.float32)
    face_diag = (mx.concatenate((zx, gx), axis=0)
                 + mx.concatenate((gx, zx), axis=0)
                 + mx.concatenate((zz, gz), axis=1)
                 + mx.concatenate((gz, zz), axis=1))
    bc = mx.array(_boundary_coefficients(hx_np, hz_np, origin_np, ky_np))
    mass = sigma * hx_m[:, None] * hz_m[None, :]
    diagonal = (face_diag[None, :, :] + bc * sigma[None, :, :]
                + k[:, None, None] ** 2 * mass[None, :, :])[:, :, :, None]

    source = np.zeros((nx, nz, n), dtype=np.float32)
    for i in range(n):
        ia = np.argmin(np.abs(xc - ab_np[i, 0]))
        ib = np.argmin(np.abs(xc - ab_np[i, 1]))
        source[ia, -1, i] += 1.0
        source[ib, -1, i] -= 1.0
    rhs = mx.broadcast_to(mx.array(source)[None, :, :, :]
                          * mx.array(currents, dtype=mx.float32)[None, None, None, :],
                          (len(ky_np), nx, nz, n))
    x = mx.zeros_like(rhs)
    r = rhs
    p = r / diagonal
    rz = mx.sum(r * p, axis=(1, 2), keepdims=True)
    threshold = rz * 1e-16
    # Fixed iteration bound preserves a differentiable MLX computation.
    for iteration in range(2500):
        x, r, p, rz = _cg_step(x, r, p, rz, diagonal, gx, gz, threshold)
        if iteration % 25 == 24:
            mx.eval(x, r, p, rz)

    receiver = _receiver_matrix(xc, mn_np[:, 0]) - _receiver_matrix(xc, mn_np[:, 1])
    surface = x[:, :, -1, :]
    mode_voltage = mx.sum(surface * mx.array(receiver.T)[None, :, :], axis=1)
    voltage = mx.sum(mode_voltage * mx.array(weights, dtype=mx.float32)[:, None], axis=0)
    ab_m = mx.array(ab, dtype=mx.float32)
    mn_m = mx.array(mn, dtype=mx.float32)
    a, b = ab_m[:, 0], ab_m[:, 1]
    m, npos = mn_m[:, 0], mn_m[:, 1]
    geom = (1.0 / mx.abs(a - m) - 1.0 / mx.abs(b - m)
            - 1.0 / mx.abs(a - npos) + 1.0 / mx.abs(b - npos)) / (2.0 * np.pi)
    rho = voltage / (mx.array(currents, dtype=mx.float32) * geom)
    return {"voltage": voltage, "apparent_resistivity": rho}
