"""Float32 acoustic C-PML propagation and velocity autodifferentiation.

The recurrence follows tests/DISCRETE_OPERATOR.md. Wavefields store only the
evolving region; padding by two zeros when taking derivatives supplies the
fixed guards, including guards for the PML memory fields.
"""

from functools import lru_cache

import mlx.core as mx
import numpy as np


def _profile(n, width, spacing, dt, frequency, max_vel):
    """Fixed nodal a, b, and Db, including profile values in the guards."""
    nfull = n + 2 * (width + 2)
    if width == 0:
        zero = np.zeros(n, dtype=np.float32)
        return tuple(mx.array(zero) for _ in range(3))

    f32 = np.float32
    index = np.arange(nfull, dtype=np.float32)
    left = np.clip((f32(width + 2) - index) / f32(width), 0, 1)
    right = np.clip(
        (index - f32(nfull - 1 - 2 - width)) / f32(width), 0, 1
    )
    distance = np.maximum(left, right)
    sigma0 = (
        -f32(3) * f32(max_vel) * np.log(f32(0.001))
        / (f32(2) * f32(width) * f32(spacing))
    )
    sigma = sigma0 * (left * left + right * right)
    alpha = f32(np.pi) * f32(frequency) * (f32(1) - distance)
    decay = sigma + alpha
    exponent = -decay * f32(dt)
    a = np.where(distance > 0, np.exp(exponent), f32(0))
    fraction = np.zeros_like(decay)
    np.divide(sigma, decay, out=fraction, where=decay != 0)
    b = fraction * np.expm1(exponent)
    c = np.zeros_like(b)
    c[2:-2] = (
        f32(2 / 3) * (b[3:-1] - b[1:-3])
        - f32(1 / 12) * (b[4:] - b[:-4])
    ) / f32(spacing)
    # Db is formed BEFORE dropping guard profile values: b is not zero there.
    return tuple(mx.array(p[2:-2], dtype=mx.float32) for p in (a, b, c))


def _guards(field):
    return mx.pad(field, [(0, 0), (2, 2), (2, 2)])


def _first_z(padded, inv_h):
    return (
        (2 / 3) * (padded[:, 3:-1, 2:-2] - padded[:, 1:-3, 2:-2])
        - (1 / 12) * (padded[:, 4:, 2:-2] - padded[:, :-4, 2:-2])
    ) * inv_h


def _first_x(padded, inv_h):
    return (
        (2 / 3) * (padded[:, 2:-2, 3:-1] - padded[:, 2:-2, 1:-3])
        - (1 / 12) * (padded[:, 2:-2, 4:] - padded[:, 2:-2, :-4])
    ) * inv_h


def _validate_locations(locations, name, expected_shape, nz, nx):
    """Check unpadded geometry before offsetting or narrowing any indices."""
    locations = np.asarray(locations)
    if locations.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.issubdtype(locations.dtype, np.integer):
        raise ValueError(f"{name} must contain integer coordinates")
    if (np.any(locations[..., 0] < 0)
            or np.any(locations[..., 0] >= nz)
            or np.any(locations[..., 1] < 0)
            or np.any(locations[..., 1] >= nx)):
        raise ValueError(f"{name} coordinates must lie inside the velocity model")
    # Widen other integer dtypes before adding the PML offset. In particular,
    # small integer dtypes must not wrap during host-side index arithmetic.
    return locations.astype(np.int64, copy=False)


@lru_cache(maxsize=8)
def _make_step(spacing):
    """Cache one compiled differentiable step, rather than a whole time graph."""
    inv_h = 1.0 / spacing
    inv_h2 = 1.0 / (spacing * spacing)

    @mx.compile
    def step(state, v2dt2, amplitudes, source_index, receiver_index, profiles):
        previous, u, psi_z, psi_x, zeta_z, zeta_x = state
        az, bz, cz, ax, bx, cx = profiles
        # Record u^k, before using forcing sample k.
        recorded = u[receiver_index]
        padded = _guards(u)
        du_z = _first_z(padded, inv_h)
        du_x = _first_x(padded, inv_h)
        ddu_z = (
            (4 / 3)
            * (padded[:, 3:-1, 2:-2] + padded[:, 1:-3, 2:-2] - 2 * u)
            - (1 / 12)
            * (padded[:, 4:, 2:-2] + padded[:, :-4, 2:-2] - 2 * u)
        ) * inv_h2
        ddu_x = (
            (4 / 3)
            * (padded[:, 2:-2, 3:-1] + padded[:, 2:-2, 1:-3] - 2 * u)
            - (1 / 12)
            * (padded[:, 2:-2, 4:] + padded[:, 2:-2, :-4] - 2 * u)
        ) * inv_h2

        tz = (1 + bz) * ddu_z + cz * du_z + _first_z(
            _guards(az * psi_z), inv_h
        )
        tx = (1 + bx) * ddu_x + cx * du_x + _first_x(
            _guards(ax * psi_x), inv_h
        )
        sz = (1 + bz) * tz + az * zeta_z
        sx = (1 + bx) * tx + ax * zeta_x
        # Scatter-add also handles several sources at the same node.
        forcing = mx.zeros_like(u).at[source_index].add(amplitudes)
        following = 2 * u - previous + v2dt2 * (sz + sx - forcing)
        next_state = (
            u,
            following,
            bz * du_z + az * psi_z,
            bx * du_x + ax * psi_x,
            bz * tz + az * zeta_z,
            bx * tx + ax * zeta_x,
        )
        return next_state, recorded

    return step


def misfit_and_grad(velocity, source_amplitudes, source_locations,
                    receiver_locations, observed, grid_spacing, dt,
                    accuracy, pml_width, pml_freq, max_vel):
    """Return traces, mean least-squares loss, and mx autodiff dJ/dvelocity."""
    if accuracy != 4:
        raise ValueError("Only fourth-order spatial accuracy is supported")
    if pml_width < 0:
        raise ValueError("pml_width must be nonnegative")

    nz, nx = velocity.shape
    shots, sources, nt = source_amplitudes.shape
    if (observed.ndim != 3 or observed.shape[0] != shots
            or observed.shape[2] != nt):
        raise ValueError("observed must have shape (shots, receivers, nt)")
    source_locations = _validate_locations(
        source_locations, "source_locations", (shots, sources, 2), nz, nx,
    )
    receiver_locations = _validate_locations(
        receiver_locations, "receiver_locations",
        (shots, observed.shape[1], 2), nz, nx,
    )
    width = int(pml_width)
    nza, nxa = nz + 2 * width, nx + 2 * width
    amplitudes = mx.array(source_amplitudes, dtype=mx.float32)
    target = mx.array(observed, dtype=mx.float32)
    model = mx.array(velocity, dtype=mx.float32)

    # Active-region coordinates omit the two guards, hence the shift by w.
    shot_index = mx.array(np.arange(shots, dtype=np.int32)[:, None])

    def indices(locations):
        return (
            shot_index,
            mx.array(locations[..., 0] + width, dtype=mx.int32),
            mx.array(locations[..., 1] + width, dtype=mx.int32),
        )

    source_index = indices(source_locations)
    receiver_index = indices(receiver_locations)
    z_profiles = _profile(nz, width, grid_spacing, dt, pml_freq, max_vel)
    x_profiles = _profile(nx, width, grid_spacing, dt, pml_freq, max_vel)
    profiles = tuple(p.reshape(1, nza, 1) for p in z_profiles) + tuple(
        p.reshape(1, 1, nxa) for p in x_profiles
    )
    step = _make_step(float(grid_spacing))

    def loss(v):
        # This extension is inside autodiff: edge and corner copies contribute
        # to the derivative of their original model cells.
        extended = mx.pad(v, [(width, width), (width, width)], mode="edge")
        v2dt2 = extended * extended * (dt * dt)
        zero = mx.zeros((shots, nza, nxa), dtype=mx.float32)
        state = (zero, zero, zero, zero, zero, zero)
        traces = []
        for k in range(nt):
            state, recorded = step(
                state, v2dt2, amplitudes[..., k], source_index,
                receiver_index, profiles,
            )
            traces.append(recorded)
        predicted = mx.stack(traces, axis=-1)
        residual = predicted - target
        objective = 0.5 * mx.mean(residual * residual)
        return objective, predicted

    # Only the supplied velocity is differentiated. Traces are auxiliary
    # output; acquisition, observed data, and the max_vel-based PML stay fixed.
    (objective, predicted), gradient = mx.value_and_grad(loss, argnums=0)(model)
    mx.eval(predicted, objective, gradient)
    return {
        "predicted": np.ascontiguousarray(np.asarray(predicted)),
        "objective": np.asarray(objective),
        "gradient": np.ascontiguousarray(np.asarray(gradient)),
    }
