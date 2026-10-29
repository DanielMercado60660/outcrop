"""Centered nodal mantle convection, with a direct discrete Stokes solve.

FFT sine transforms diagonalize the *finite-difference* Dirichlet Laplacian.
Its squared inverse applies the two Poisson solves without an intervening
round trip to physical space. All field arithmetic and time stepping use MLX
float32; NumPy is used only to construct grid-dependent constants on the host.
"""

import operator
from functools import cache

import mlx.core as mx
import numpy as np


def _temperature_array(temperature):
    t = mx.array(temperature, dtype=mx.float32)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or t.shape[0] < 5:
        raise ValueError("temperature must have shape (N+1, N+1), N >= 4")
    return t


@cache
def _kernels(n, flowing):
    """Cache constants and compiled kernels by grid size and zero-Ra branch."""
    # Construct these constants in host double precision before rounding once.
    # The DST-I eigenvalues belong to the nodal five-point stencil, rather
    # than the continuum Laplacian's -pi**2 * (kx**2 + kz**2).
    k = np.arange(1, n, dtype=np.float64)
    eigenvalues = -4.0 * n * n * np.sin(np.pi * k / (2.0 * n)) ** 2
    inverse_biharmonic = 1.0 / (eigenvalues[:, None] + eigenvalues[None, :]) ** 2
    # Each unnormalized 1D DST-I squares to 2*n times the identity. A
    # forward/inverse 2D pair therefore needs a total factor 1/(4*n*n).
    inverse_squared = mx.array((inverse_biharmonic / (4.0 * n * n)).astype(np.float32))
    weights_host = np.full(n + 1, 1.0 / n, dtype=np.float64)
    weights_host[[0, -1]] *= 0.5
    weights = mx.array(weights_host.astype(np.float32))
    mx.eval(inverse_squared, weights)

    def dst_rows(a):
        """Unnormalized DST-I along x via a length-2*n real odd extension."""
        zero = mx.zeros((n - 1, 1), dtype=mx.float32)
        odd = mx.concatenate((zero, a, zero, -a[:, ::-1]), axis=1)
        # Im FFT(0,a,0,-reverse(a))[k] = -2*sum_j a[j]*sin(pi*j*k/n).
        return -mx.imag(mx.fft.rfft(odd, axis=1))[:, 1:n]

    def dst2(a):
        return dst_rows(dst_rows(a).T).T

    def solve(t, ra):
        if not flowing:
            zero = mx.zeros_like(t)
            return zero, zero, zero

        # L q = Ra T_x, L psi = q. The two negative Laplacian
        # eigenvalues multiply to a positive squared eigenvalue.
        forcing = (t[1:-1, 2:] - t[1:-1, :-2]) * (0.5 * n) * ra
        modes = dst2(forcing)
        interior = dst2(modes * inverse_squared)
        psi = mx.pad(interior, ((1, 1), (1, 1)))

        # Odd streamfunction ghosts give twice the usual centered derivative
        # at a wall. Zero psi along each wall makes all normal components and
        # all four corner velocities exactly zero.
        u = mx.concatenate(
            (
                n * psi[1:2, :],
                (0.5 * n) * (psi[2:, :] - psi[:-2, :]),
                -n * psi[-2:-1, :],
            ),
            axis=0,
        )
        w = mx.concatenate(
            (
                -n * psi[:, 1:2],
                -(0.5 * n) * (psi[:, 2:] - psi[:, :-2]),
                n * psi[:, -2:-1],
            ),
            axis=1,
        )
        return psi, u, w

    def rate(t, ra):
        center = t[1:-1, :]
        # Reflected side ghosts: the sidewall node itself evolves. In
        # particular it is never overwritten by its first interior neighbor.
        left = mx.concatenate((center[:, 1:2], center[:, :-1]), axis=1)
        right = mx.concatenate((center[:, 1:], center[:, -2:-1]), axis=1)
        above, below = t[2:, :], t[:-2, :]

        # Sum neighbor-minus-center differences to reduce cancellation in
        # the float32 Laplacian, while retaining the five-point stencil.
        lap = (n * n) * (
            (right - center) + (left - center)
            + (above - center) + (below - center)
        )
        if flowing:
            _, u, w = solve(t, ra)
            tx = (0.5 * n) * (right - left)
            tz = (0.5 * n) * (above - below)
            lap = lap - u[1:-1, :] * tx - w[1:-1, :] * tz
        # Fixed bottom/top temperatures have zero derivative at every stage.
        return mx.pad(lap, ((1, 1), (0, 0)))

    def step(t, ra, dt):
        midpoint = t + (0.5 * dt) * rate(t, ra)
        return t + dt * rate(midpoint, ra)

    def diagnostics(t, u, w):
        top_flux = (0.5 * n) * (4.0 * t[-2] - t[-3] - 3.0 * t[-1])
        bottom_flux = (0.5 * n) * (3.0 * t[0] - 4.0 * t[1] + t[2])
        nu_top = mx.sum(weights * top_flux)
        nu_bottom = mx.sum(weights * bottom_flux)
        speed_squared = u * u + w * w
        vrms = mx.sqrt(mx.sum(weights[:, None] * speed_squared * weights[None, :]))
        return nu_top, nu_bottom, vrms

    return mx.compile(solve), mx.compile(step), mx.compile(diagnostics)


def stokes(temperature, rayleigh):
    """Return psi, horizontal u, and upward w for the specified nodal stencil."""
    t = _temperature_array(temperature)
    if rayleigh < 0:
        raise ValueError("rayleigh must be nonnegative")
    solve, _, _ = _kernels(t.shape[0] - 1, rayleigh != 0)
    psi, u, w = solve(t, mx.array(rayleigh, dtype=mx.float32))
    return {"psi": mx.contiguous(psi), "u": mx.contiguous(u), "w": mx.contiguous(w)}


def convection(temperature, rayleigh, dt, steps):
    """Advance exactly ``steps`` explicit midpoint steps, then diagnose fields."""
    t = _temperature_array(temperature)
    steps = operator.index(steps)
    if rayleigh < 0 or dt <= 0 or steps < 0:
        raise ValueError("require rayleigh >= 0, dt > 0, and steps >= 0")
    solve, step, diagnostics = _kernels(t.shape[0] - 1, rayleigh != 0)
    ra = mx.array(rayleigh, dtype=mx.float32)
    timestep = mx.array(dt, dtype=mx.float32)
    for iteration in range(steps):
        t = step(t, ra, timestep)
        # Bound the lazy graph without synchronizing after every RK stage.
        if (iteration + 1) % 32 == 0:
            mx.eval(t)
    _, u, w = solve(t, ra)
    nu_top, nu_bottom, vrms = diagnostics(t, u, w)
    return {
        "temperature": mx.contiguous(t),
        "u": mx.contiguous(u),
        "w": mx.contiguous(w),
        "nu_top": nu_top,
        "nu_bottom": nu_bottom,
        "vrms": vrms,
    }
