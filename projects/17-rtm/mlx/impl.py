"""Fourth-order acoustic Born modelling and its MLX autodiff transpose.

The spatial and CFS-PML recurrences are shared with the implementations of
projects 13/14, with separate z/x spacings. Only scatter is differentiated:
the fixed background forcing history is evaluated before tracing autodiff.
"""

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np


def _validate_acquisition(velocity, source_amplitudes, source_locations,
                          receiver_locations, data):
    """Reject invalid physical coordinates and implicit trace broadcasting."""
    nz, nx = velocity.shape
    source_shape = np.shape(source_amplitudes)
    if len(source_shape) != 3:
        raise ValueError("source_amplitudes must have shape (nshot, nsrc, nt)")
    shots, sources, nt = source_shape
    receiver_count = None
    for name, locations, count in (
        ("source_locations", source_locations, sources),
        ("receiver_locations", receiver_locations, None),
    ):
        locations = np.asarray(locations)
        if (locations.ndim != 3 or locations.shape[0] != shots
                or locations.shape[2] != 2
                or (count is not None and locations.shape[1] != count)):
            raise ValueError(
                f"{name} must have shape (nshot, "
                f"{'nsrc' if count is not None else 'nrec'}, 2) "
                "matching source_amplitudes"
            )
        if not np.issubdtype(locations.dtype, np.integer):
            raise ValueError(f"{name} must contain integer grid coordinates")
        if (np.any(locations[..., 0] < 0)
                or np.any(locations[..., 0] >= nz)
                or np.any(locations[..., 1] < 0)
                or np.any(locations[..., 1] >= nx)):
            raise ValueError(
                f"{name} coordinates must lie in the physical grid: "
                f"0 <= z < {nz}, 0 <= x < {nx}"
            )
        if name == "receiver_locations":
            receiver_count = locations.shape[1]

    expected = (shots, receiver_count, nt)
    if np.shape(data) != expected:
        raise ValueError(f"data must have shape {expected}, got {np.shape(data)}")


def _profile(n, width, spacing, dt, frequency, vmax):
    """Nodal CFS-PML coefficients; include guards when differentiating b."""
    if width == 0:
        zero = mx.zeros((n,), dtype=mx.float32)
        return zero, zero, zero

    size = n + 2 * width + 4
    i = np.arange(size, dtype=np.float64)
    fraction = np.maximum(
        np.clip((2 + width - i) / width, 0, 1),
        np.clip((i - (size - 1 - 2 - width)) / width, 0, 1),
    )
    sigma = (
        3 * vmax * math.log(1000) / (2 * width * spacing) * fraction**2
    )
    alpha = math.pi * frequency * (1 - fraction)
    decay = sigma + alpha
    a = np.where(fraction > 0, np.exp(-decay * dt), 0)
    ratio = np.divide(sigma, decay, out=np.zeros_like(sigma), where=decay != 0)
    b = np.where(fraction > 0, ratio * np.expm1(-decay * dt), 0)
    db = (
        (2 / 3) * (b[3:-1] - b[1:-3])
        - (1 / 12) * (b[4:] - b[:-4])
    ) / spacing
    return tuple(mx.array(q, dtype=mx.float32) for q in (a[2:-2], b[2:-2], db))


def _guards(field):
    """Only the physical grid and PML evolve; the two FD guards stay zero."""
    return mx.pad(field, [(0, 0), (2, 2), (2, 2)])


def _first_z(padded, spacing):
    return (
        (2 / 3) * (padded[:, 3:-1, 2:-2] - padded[:, 1:-3, 2:-2])
        - (1 / 12) * (padded[:, 4:, 2:-2] - padded[:, :-4, 2:-2])
    ) / spacing


def _first_x(padded, spacing):
    return (
        (2 / 3) * (padded[:, 2:-2, 3:-1] - padded[:, 2:-2, 1:-3])
        - (1 / 12) * (padded[:, 2:-2, 4:] - padded[:, 2:-2, :-4])
    ) / spacing


@lru_cache(maxsize=8)
def _make_step(dz, dx):
    """Compile one differentiable time step, rather than the whole survey."""

    @mx.compile
    def step(state, vdt2, forcing, profiles):
        previous, u, psi_z, psi_x, zeta_z, zeta_x = state
        az, bz, dbz, ax, bx, dbx = profiles
        padded = _guards(u)
        du_z = _first_z(padded, dz)
        du_x = _first_x(padded, dx)
        # Difference form avoids cancellation between large central terms.
        ddu_z = (
            (4 / 3) * (
                (padded[:, 3:-1, 2:-2] - u)
                + (padded[:, 1:-3, 2:-2] - u)
            )
            - (1 / 12) * (
                (padded[:, 4:, 2:-2] - u)
                + (padded[:, :-4, 2:-2] - u)
            )
        ) / (dz * dz)
        ddu_x = (
            (4 / 3) * (
                (padded[:, 2:-2, 3:-1] - u)
                + (padded[:, 2:-2, 1:-3] - u)
            )
            - (1 / 12) * (
                (padded[:, 2:-2, 4:] - u)
                + (padded[:, 2:-2, :-4] - u)
            )
        ) / (dx * dx)

        apz, apx = az * psi_z, ax * psi_x
        azz, axx = az * zeta_z, ax * zeta_x
        tz = (1 + bz) * ddu_z + dbz * du_z + _first_z(_guards(apz), dz)
        tx = (1 + bx) * ddu_x + dbx * du_x + _first_x(_guards(apx), dx)
        hz = (1 + bz) * tz + azz
        hx = (1 + bx) * tx + axx
        laplacian = hz + hx
        following = 2 * u - previous + vdt2 * laplacian + forcing
        return (
            u, following,
            bz * du_z + apz, bx * du_x + apx,
            bz * tz + azz, bx * tx + axx,
        ), laplacian

    return step


def _born_operator(velocity, source_amplitudes, source_locations,
                   receiver_locations, grid_spacing, dt, pml_width, pml_freq):
    """Set up J_v with fixed v, acquisition and PML; return a scatter closure."""
    nz, nx = velocity.shape
    width = int(pml_width)
    if width < 0:
        raise ValueError("pml_width must be nonnegative")
    dz, dx = map(float, grid_spacing)
    nza, nxa = nz + 2 * width, nx + 2 * width
    source = mx.array(source_amplitudes, dtype=mx.float32)
    shots, _, nt = source.shape
    shot_index = mx.array(np.arange(shots, dtype=np.int32)[:, None])

    def indices(locations):
        locations = np.asarray(locations)
        return (
            shot_index,
            mx.array(locations[..., 0] + width, dtype=mx.int32),
            mx.array(locations[..., 1] + width, dtype=mx.int32),
        )

    source_index = indices(source_locations)
    receiver_index = indices(receiver_locations)
    iz = mx.array(np.clip(np.arange(-width, nz + width), 0, nz - 1), dtype=mx.int32)
    ix = mx.array(np.clip(np.arange(-width, nx + width), 0, nx - 1), dtype=mx.int32)

    def extend(model):
        # Autodiff of these gathers sums replicated copies back into edges.
        return mx.take(mx.take(model, iz, axis=0), ix, axis=1)

    v = extend(mx.array(velocity, dtype=mx.float32))
    vmax = float(np.max(velocity))
    profiles = tuple(
        p.reshape(1, nza, 1)
        for p in _profile(nz, width, dz, dt, pml_freq, vmax)
    ) + tuple(
        p.reshape(1, 1, nxa)
        for p in _profile(nx, width, dx, dt, pml_freq, vmax)
    )
    vdt2 = (v * dt) ** 2
    derivative_scale = (2 * dt * dt) * v
    src_z, src_x = source_index[1:]
    source_scale = vdt2[src_z, src_x]
    derivative_source_scale = derivative_scale[src_z, src_x]
    zero = mx.zeros((shots, nza, nxa), dtype=mx.float32)
    initial = (zero, zero, zero, zero, zero, zero)
    step = _make_step(dz, dx)

    # u^{k+1} = 2u^k-u^{k-1} + v^2 dt^2 H(u^k) - v^2 dt^2 f^k.
    # Its velocity tangent is driven by dv * 2v dt^2 (H(u^k)-f^k).
    # Receiver sample k reads u^k, so forcing nt-1 has no recorded effect.
    state = initial
    born_forcing = []
    for k in range(max(nt - 1, 0)):
        amplitudes = source[..., k]
        forcing = zero.at[source_index].add(-source_scale * amplitudes)
        state, laplacian = step(state, vdt2, forcing, profiles)
        drive = derivative_scale * laplacian
        drive = drive.at[source_index].add(-derivative_source_scale * amplitudes)
        # No scatter dependency exists yet. Evaluate to release each previous
        # background graph while keeping the forcing required by its transpose.
        mx.eval(state, drive)
        born_forcing.append(drive)

    def predict(scatter):
        padded_scatter = extend(scatter)
        state = initial
        traces = []
        for k in range(nt):
            traces.append(state[1][receiver_index])
            if k < nt - 1:
                state, _ = step(
                    state, vdt2, padded_scatter * born_forcing[k], profiles,
                )
        if not nt:
            return mx.zeros((shots, receiver_locations.shape[1], 0), dtype=mx.float32)
        return mx.stack(traces, axis=-1)

    return predict


def rtm(velocity, source_amplitudes, source_locations, receiver_locations,
        data, grid_spacing, dt, pml_width, pml_freq):
    """Return the signed Euclidean Born transpose, summed over all shots."""
    _validate_acquisition(
        velocity, source_amplitudes, source_locations, receiver_locations, data,
    )
    predict = _born_operator(
        velocity, source_amplitudes, source_locations, receiver_locations,
        grid_spacing, dt, pml_width, pml_freq,
    )
    target = mx.array(data, dtype=mx.float32)

    def pairing(scatter):
        return mx.sum(predict(scatter) * target)

    image = mx.grad(pairing)(mx.zeros(velocity.shape, dtype=mx.float32))
    mx.eval(image)
    return {"image": image}


def born_misfit_and_grad(velocity, scatter, source_amplitudes, source_locations,
                         receiver_locations, data, grid_spacing, dt,
                         pml_width, pml_freq):
    """Return J_v scatter, the summed least-squares loss, and its gradient."""
    _validate_acquisition(
        velocity, source_amplitudes, source_locations, receiver_locations, data,
    )
    predict = _born_operator(
        velocity, source_amplitudes, source_locations, receiver_locations,
        grid_spacing, dt, pml_width, pml_freq,
    )
    target = mx.array(data, dtype=mx.float32)

    def objective(direction):
        predicted = predict(direction)
        residual = predicted - target
        return 0.5 * mx.sum(residual * residual), predicted

    (loss, predicted), gradient = mx.value_and_grad(objective)(
        mx.array(scatter, dtype=mx.float32)
    )
    mx.eval(loss, predicted, gradient)
    return {"predicted": predicted, "loss": loss, "gradient": gradient}
