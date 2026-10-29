"""Acoustic C-PML autodiff with bounded restart storage and optimal replay.

Only a single time step is traced by autodiff. Evaluating every forward and
reverse step prevents MLX's lazy graphs from retaining an implicit trajectory.
The full-storage and bounded-storage paths execute the same reverse kernel.
"""

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

_FIRST = {2: (0.5,), 4: (2 / 3, -1 / 12)}
_SECOND = {2: (1.0,), 4: (4 / 3, -1 / 12)}


def _profile(n, p, low, high, spacing, dt, frequency, vmax, first):
    """Deepwave nodal profiles, including nonzero b in the stencil guards."""
    size = n + 2 * p
    indices = np.arange(size, dtype=np.float64)
    qlow = (np.clip((p + low - indices) / low, 0, 1)
            if low else np.zeros(size))
    qhigh = (np.clip((indices - (size - 1 - p - high)) / high, 0, 1)
             if high else np.zeros(size))
    q = np.maximum(qlow, qhigh)
    scale = 3 * vmax * math.log(1000) / (2 * spacing)
    sigma = np.zeros(size, dtype=np.float64)
    if low:
        sigma += scale * qlow**2 / low
    if high:
        sigma += scale * qhigh**2 / high
    alpha = math.pi * frequency * (1 - q)
    decay = sigma + alpha
    a = np.where(q > 0, np.exp(-decay * dt), 0)
    ratio = np.divide(sigma, decay, out=np.zeros_like(sigma), where=decay != 0)
    b = np.where(q > 0, ratio * np.expm1(-decay * dt), 0)
    db = np.zeros(n, dtype=np.float64)
    for j, coefficient in enumerate(first, 1):
        db += coefficient * (b[p + j:size - p + j]
                             - b[p - j:size - p - j]) / spacing
    return tuple(mx.array(x, dtype=mx.float32) for x in (a[p:-p], b[p:-p], db))


def _derivatives(field, axis, p, spacing, first, second=None):
    """Centered derivatives on the evolving grid with fixed zero exterior."""
    padded = mx.pad(field, [(0, 0), (p, p), (p, p)])
    nz, nx = field.shape[-2:]
    d1, d2 = None, None
    for j, coefficient in enumerate(first, 1):
        if axis == 0:
            plus = padded[:, p + j:p + nz + j, p:p + nx]
            minus = padded[:, p - j:p + nz - j, p:p + nx]
        else:
            plus = padded[:, p:p + nz, p + j:p + nx + j]
            minus = padded[:, p:p + nz, p - j:p + nx - j]
        term = coefficient * (plus - minus)
        d1 = term if d1 is None else d1 + term
        if second is not None:
            term = second[j - 1] * ((plus - field) + (minus - field))
            d2 = term if d2 is None else d2 + term
    if second is None:
        return d1 / spacing
    return d1 / spacing, d2 / (spacing * spacing)


@lru_cache(maxsize=8)
def _kernels(accuracy, dz, dx):
    """Cache code only: all model, survey, state and adjoint arrays are inputs."""
    p = accuracy // 2
    first, second = _FIRST[accuracy], _SECOND[accuracy]

    @mx.compile
    def step(state, coefficient, forcing, profiles):
        previous, u, psi_z, psi_x, zeta_z, zeta_x = state
        az, bz, dbz, ax, bx, dbx = profiles
        gz, lz = _derivatives(u, 0, p, dz, first, second)
        gx, lx = _derivatives(u, 1, p, dx, first, second)
        apz, apx = az * psi_z, ax * psi_x
        azz, axx = az * zeta_z, ax * zeta_x
        tz = (1 + bz) * lz + dbz * gz + _derivatives(apz, 0, p, dz, first)
        tx = (1 + bx) * lx + dbx * gx + _derivatives(apx, 1, p, dx, first)
        hz, hx = (1 + bz) * tz + azz, (1 + bx) * tx + axx
        # A dense point-RHS avoids a velocity gather whose transpose would
        # require colliding scatter additions for sources shared by shots.
        following = 2 * u - previous + coefficient * (hz + hx)
        following = following - coefficient * forcing
        return (u, following, bz * gz + apz, bx * gx + apx,
                bz * tz + azz, bx * tx + axx)

    def pairing(state, coefficient, forcing, profiles, adjoint, data_bar,
                receiver_index):
        following = step(state, coefficient, forcing, profiles)
        value = mx.sum(following[0] * adjoint[0])
        for field, cotangent in zip(following[1:], adjoint[1:]):
            value = value + mx.sum(field * cotangent)
        return value + mx.sum(state[1][receiver_index] * data_bar)

    # This scalar contraction computes one VJP using mx.grad. Both the
    # restart and v^2 dt^2 are independent differentiable arguments.
    reverse = mx.compile(mx.grad(pairing, argnums=(0, 1)))
    return step, reverse


def _extend(model, widths):
    """Edge replication with deterministic reduction in its autodiff VJP."""
    zl, zh, xl, xh = widths
    nz, nx = model.shape
    parts = []
    if zl:
        parts.append(mx.broadcast_to(model[:1, :], (zl, nx)))
    parts.append(model)
    if zh:
        parts.append(mx.broadcast_to(model[-1:, :], (zh, nx)))
    result = mx.concatenate(parts, axis=0) if len(parts) > 1 else model
    parts = []
    if xl:
        parts.append(mx.broadcast_to(result[:, :1], (nz + zl + zh, xl)))
    parts.append(result)
    if xh:
        parts.append(mx.broadcast_to(result[:, -1:], (nz + zl + zh, xh)))
    return mx.concatenate(parts, axis=1) if len(parts) > 1 else result


def _splits(nt, capacity):
    """Bottom-up F(n,c,e), keeping the smallest minimizing integer split."""
    costs = np.zeros((2, capacity + 1, nt + 1), dtype=np.int64)
    splits = np.zeros_like(costs)
    for c in range(capacity + 1):
        for n in range(1, nt + 1):
            for e in (0, 1):
                if n == 1:
                    costs[e, c, n] = e
                elif c == 0:
                    costs[e, c, n] = n * (n - 1) // 2 + e * n
                elif c >= n - 1:
                    costs[e, c, n] = n - 1 + e
                else:
                    k = np.arange(1, n)
                    options = k + costs[e, c - 1, n - k] + costs[0, c, k]
                    best = int(np.argmin(options))
                    costs[e, c, n] = options[best]
                    splits[e, c, n] = best + 1
    return splits


def _events(nt, capacity):
    """Yield the exact recursive-split plan; no numerical states live here."""
    splits = _splits(nt, capacity) if 0 < capacity < nt - 1 else None

    def interval(start, n, slots, base, reach_end):
        # On entry the working state is at start. The base restart is
        # available outside slots (an ancestor slot, or implicit state 0).
        c = len(slots)
        if n == 1:
            if reach_end:
                yield (0, start + 1, -1)
                yield (2, start, base)
            yield (4, start, -1)
        elif c == 0:
            yield (0, start + n - 1 + reach_end, -1)
            if reach_end:
                yield (2, start, base)
                yield (0, start + n - 1, -1)
            yield (4, start + n - 1, -1)
            for t in range(start + n - 2, start - 1, -1):
                yield (2, start, base)
                if t > start:
                    yield (0, t, -1)
                yield (4, t, -1)
        elif c >= n - 1:
            for j in range(1, n):
                yield (0, start + j, -1)
                yield (1, start + j, slots[j - 1])
            if reach_end:
                yield (0, start + n, -1)
                yield (2, start + n - 1, slots[n - 2])
            yield (4, start + n - 1, -1)
            yield (3, start + n - 1, slots[n - 2])
            for j in range(n - 2, 0, -1):
                yield (2, start + j, slots[j - 1])
                yield (4, start + j, -1)
                yield (3, start + j, slots[j - 1])
            yield (2, start, base)
            yield (4, start, -1)
        else:
            k = int(splits[reach_end, c, n])
            slot = slots[0]
            yield (0, start + k, -1)
            yield (1, start + k, slot)
            yield from interval(start + k, n - k, slots[1:], slot, reach_end)
            yield (3, start + k, slot)
            yield (2, start, base)
            yield from interval(start, k, slots, base, 0)

    yield from interval(0, nt, tuple(range(capacity)), -1, 1)


def misfit_and_grad(velocity, source_amplitudes, source_locations,
                    receiver_locations, observed, grid_spacing, dt,
                    accuracy, pml_width, pml_freq, max_vel, checkpoint_budget):
    """Return summed L2 loss, physical velocity gradient and executed audit."""
    if accuracy not in _FIRST:
        raise ValueError("accuracy must be 2 or 4")
    shots, sources, nt = source_amplitudes.shape
    if nt < 1 or not -1 <= checkpoint_budget < nt:
        raise ValueError("need nt >= 1 and checkpoint_budget in [-1, nt-1]")
    widths = tuple(map(int, pml_width))
    if len(widths) != 4 or min(widths) < 0:
        raise ValueError("pml_width must contain four nonnegative widths")
    dz, dx = map(float, grid_spacing)
    if min(dz, dx, dt) <= 0:
        raise ValueError("grid spacings and dt must be positive")
    capacity = nt - 1 if checkpoint_budget == -1 else checkpoint_budget
    zl, zh, xl, xh = widths
    nz, nx = velocity.shape
    if observed.ndim != 3 or observed.shape[0] != shots or observed.shape[2] != nt:
        raise ValueError("observed must have shape [nshot, nreceiver, nt]")

    def validate_locations(locations, name, count):
        locations = np.asarray(locations)
        if locations.shape != (shots, count, 2):
            raise ValueError(f"{name} must have shape {(shots, count, 2)}")
        if not np.issubdtype(locations.dtype, np.integer):
            raise ValueError(f"{name} must contain integer cell indices")
        if (np.any(locations < 0) or np.any(locations[..., 0] >= nz)
                or np.any(locations[..., 1] >= nx)):
            raise ValueError(f"{name} must lie within the physical velocity grid")
        return locations

    # Check physical coordinates before adding PML offsets or narrowing to
    # MLX indices: an out-of-bounds location can otherwise address a PML cell.
    source_locations = validate_locations(source_locations, "source_locations", sources)
    receiver_locations = validate_locations(
        receiver_locations, "receiver_locations", observed.shape[1],
    )
    nza, nxa = nz + zl + zh, nx + xl + xh
    p = accuracy // 2
    first = _FIRST[accuracy]
    profiles = tuple(
        x.reshape(1, nza, 1)
        for x in _profile(nza, p, zl, zh, dz, dt, pml_freq, max_vel, first)
    ) + tuple(
        x.reshape(1, 1, nxa)
        for x in _profile(nxa, p, xl, xh, dx, dt, pml_freq, max_vel, first)
    )
    model = mx.array(velocity, dtype=mx.float32)
    source = mx.array(source_amplitudes, dtype=mx.float32)
    target = mx.array(observed, dtype=mx.float32)
    shot_index = mx.array(np.arange(shots, dtype=np.int32)[:, None])

    def indices(locations):
        locations = np.asarray(locations)
        return (shot_index,
                mx.array(locations[..., 0] + zl, dtype=mx.int32),
                mx.array(locations[..., 1] + xl, dtype=mx.int32))

    source_index = indices(source_locations)
    receiver_index = indices(receiver_locations)

    def model_coefficient(v):
        return (_extend(v, widths) * dt) ** 2

    coefficient = model_coefficient(model)
    zero = mx.zeros((shots, nza, nxa), dtype=mx.float32)
    initial = (zero, zero, zero, zero, zero, zero)
    state, adjoint = initial, initial
    coefficient_bar = mx.zeros_like(coefficient)
    mx.eval(coefficient, profiles, initial, source, target)
    step, reverse = _kernels(accuracy, dz, dx)
    live, schedule, traces = {}, [], []
    current, work, peak = 0, 0, 0
    first_forward_complete = False

    for op, time, slot in _events(nt, capacity):
        if op == 0:
            for t in range(current, time):
                recording = not first_forward_complete
                if recording:
                    trace = state[1][receiver_index]
                forcing = zero.at[source_index].add(source[..., t])
                state = step(state, coefficient, forcing, profiles)
                if recording:
                    mx.eval(state, trace)
                    traces.append(trace)
                else:
                    mx.eval(state)
            work += time - current
            current = time
            if current == nt and not first_forward_complete:
                predicted = mx.stack(traces, axis=-1)
                residual = predicted - target
                objective = 0.5 * mx.sum(residual * residual)
                mx.eval(predicted, residual, objective)
                traces.clear()
                first_forward_complete = True
        elif op == 1:
            # Arrays are immutable. Sharing adjacent u levels is exact;
            # each restart still owns references to all six state fields.
            live[slot] = state
            peak = max(peak, len(live))
        elif op == 2:
            state = initial if slot == -1 else live[slot]
            current = time
            mx.eval(state)
        elif op == 3:
            del live[slot]
        else:  # REVERSE, at the input state of exactly one time step
            forcing = zero.at[source_index].add(source[..., time])
            adjoint, increment = reverse(
                state, coefficient, forcing, profiles, adjoint,
                residual[..., time], receiver_index,
            )
            coefficient_bar = coefficient_bar + increment
            mx.eval(adjoint, coefficient_bar)
        # The log is appended only after the corresponding operation ran.
        schedule.append((op, time, slot))

    def coefficient_pairing(v):
        return mx.sum(model_coefficient(v) * coefficient_bar)

    # Apply the model-map VJP once, after descending time accumulation.
    # Broadcast reductions sum every replicated edge/corner contribution.
    gradient = mx.grad(coefficient_pairing)(model)
    mx.eval(gradient)
    result = {
        "objective": np.array(objective, dtype=np.float32),
        "gradient": np.array(gradient, dtype=np.float32),
        "predicted": np.array(predicted, dtype=np.float32),
        "schedule": np.asarray(schedule, dtype=np.int64).reshape(-1, 3),
        "forward_steps": np.array(work, dtype=np.int64),
        "peak_checkpoints": np.array(peak, dtype=np.int64),
        "checkpoint_bytes": np.array(peak * 6 * shots * nza * nxa * 4,
                                     dtype=np.int64),
    }
    # Return NumPy values only; neither a lazy graph nor a restart escapes.
    return result
