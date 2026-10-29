"""Second-order staggered P-SV propagation with IVM vacuum and CFS-C-PML.

The normal stresses, two velocities, shear stress, and eight derivative
memories all live on the padded grid. Only material is edge replicated;
wavefields have a zero exterior. All material and source arithmetic stays
in MLX, including the calculation differentiated by ``misfit_and_grad``.

Each internal number is represented as a sum of two float32 arrays. Keeping
the low part prevents material conversion and time-accumulation roundoff
from overwhelming small centered differences of the velocity objective.
The interface still rounds each trace, objective, and gradient to float32.
"""

import mlx.core as mx
import numpy as np

# These kernels recover ONLY the roundoff of an already computed operation.
# Disabling contraction matters for TwoSum; the product residual explicitly
# uses an FMA. Neither kernel defines physical propagation or an adjoint.
# Roundoff has zero derivative under the usual differentiation of floating
# arithmetic, so stop_gradient keeps these kernels out of the reverse graph.
_SUM_ERROR = mx.fast.metal_kernel(
    name="psv_sum_roundoff",
    input_names=["a", "b", "rounded"],
    output_names=["error"],
    header="#pragma clang fp contract(off)\n",
    source="""
        uint i = thread_position_in_grid.x;
        float z = rounded[i] - a[i];
        error[i] = (a[i] - (rounded[i] - z)) + (b[i] - z);
    """,
    compile_options={"math_mode": "safe"},
)
_PRODUCT_ERROR = mx.fast.metal_kernel(
    name="psv_product_roundoff",
    input_names=["a", "b", "rounded"],
    output_names=["error"],
    source="""
        uint i = thread_position_in_grid.x;
        error[i] = metal::fma(a[i], b[i], -rounded[i]);
    """,
    compile_options={"math_mode": "safe"},
)


def _roundoff(kernel, a, b, rounded):
    if rounded.size == 0:
        return mx.zeros_like(rounded)
    a, b = mx.broadcast_arrays(a, b)
    error = kernel(
        inputs=[a, b, rounded],
        grid=(rounded.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[rounded.shape],
        output_dtypes=[mx.float32],
    )[0]
    return mx.stop_gradient(error)


def _two_sum(a, b):
    rounded = a + b
    return rounded, _roundoff(_SUM_ERROR, a, b, rounded)


def _wide(q):
    """An exact float32 input, with no low component."""
    return q, mx.zeros_like(q)


def _constant(q):
    """Split fixed host constants before transferring them to the GPU."""
    q = np.asarray(q, dtype=np.float64)
    high = q.astype(np.float32)
    low = (q - high.astype(np.float64)).astype(np.float32)
    return mx.array(high), mx.array(low)


def _add(a, b):
    high, error = _two_sum(a[0], b[0])
    return _two_sum(high, (error + a[1]) + b[1])


def _neg(a):
    return -a[0], -a[1]


def _sub(a, b):
    return _add(a, _neg(b))


def _mul(a, b):
    high = a[0] * b[0]
    error = _roundoff(_PRODUCT_ERROR, a[0], b[0], high)
    low = ((error + a[0] * b[1]) + a[1] * b[0]) + a[1] * b[1]
    return _two_sum(high, low)


def _div(a, b):
    high = a[0] / b[0]
    # One residual correction supplies approximately 48 significand bits.
    remainder = _sub(a, _mul(b, _wide(high)))
    low = (remainder[0] + remainder[1]) / b[0]
    return _two_sum(high, low)


def _map_pair(function, pair):
    return tuple(function(q) for q in pair)


def _take(pair, index):
    return _map_pair(lambda q: q[index], pair)


def _pad(pair, padding, mode="constant"):
    return _map_pair(lambda q: mx.pad(q, padding, mode=mode), pair)


def _reshape(pair, shape):
    return _map_pair(lambda q: q.reshape(shape), pair)


def _select(mask, a, b):
    return tuple(mx.where(mask, x, y) for x, y in zip(a, b))


def _rounded(pair):
    return pair[0] + pair[1]


def _profile(n, low, high, spacing, dt, frequency, max_vel, offset):
    """Fixed, host-side PML design at integer or positive half nodes."""
    u = np.arange(n + low + high, dtype=np.float64) + offset
    sigma = np.zeros_like(u)
    depth = np.zeros_like(u)
    if low:
        s = np.clip((low - u) / low, 0.0, 1.0)
        sigma += (-3.0 * max_vel * np.log(0.001) / (2.0 * low * spacing)) * s**2
        depth = np.maximum(depth, s)
    if high:
        s = np.clip((u - (low + n - 1)) / high, 0.0, 1.0)
        sigma += (-3.0 * max_vel * np.log(0.001) / (2.0 * high * spacing)) * s**2
        depth = np.maximum(depth, s)
    active = depth > 0.0
    rate = sigma + np.pi * frequency * (1.0 - depth)
    decay = np.zeros_like(u)
    coupling = np.zeros_like(u)
    decay[active] = np.exp(-rate[active] * dt)
    coupling[active] = (sigma[active] / rate[active]) * np.expm1(-rate[active] * dt)
    return _constant(decay), _constant(coupling)


def _locations(locations, name, model_shape, shots, count=None):
    """Validate unpadded coordinates before any cast, offset, or flattening."""
    locations = np.asarray(locations)
    expected = f"({shots}, {count if count is not None else 'nreceiver'}, 2)"
    if (locations.ndim != 3 or locations.shape[0] != shots
            or locations.shape[2] != 2
            or (count is not None and locations.shape[1] != count)):
        raise ValueError(f"{name} must have shape {expected}")
    if locations.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer grid indices")
    upper = np.asarray(model_shape, dtype=np.int64) - 1
    if np.any(locations < 0) or np.any(locations >= upper):
        raise ValueError(
            f"{name} coordinates must satisfy 0 <= y < {upper[0]} and "
            f"0 <= x < {upper[1]} in the unpadded model")
    return locations.astype(np.int64, copy=False)


def _geometry(shape, source_y, source_x, source_locations, receiver_locations,
              spacing, dt, pml_width, pml_freq, max_vel):
    """Prepare fixed geometry, profiles, and collision-free injection groups."""
    ny, nx = shape
    if source_y.ndim != 3 or source_x.shape != source_y.shape:
        raise ValueError("source_y and source_x must have matching shapes "
                         "(nshot, nsource, nt)")
    shots, sources, _ = source_y.shape
    src = _locations(source_locations, "source_locations", shape, shots, sources)
    rec = _locations(receiver_locations, "receiver_locations", shape, shots)
    top, bottom, left, right = (int(w) for w in pml_width)
    height, width = ny + top + bottom, nx + left + right
    area = height * width
    source_spatial = (src[..., 0] + top) * width + src[..., 1] + left
    shot_offset = np.arange(shots, dtype=np.int64)[:, None] * area
    source_grid = (source_spatial + shot_offset).reshape(-1)
    receiver_grid = ((rec[..., 0] + top) * width + rec[..., 1] + left
                     + shot_offset)

    # One scatter per occurrence rank: within each scatter destinations are
    # unique. Coincident sources are added in input order, without racing
    # floating-point atomics. This also preserves source autodifferentiability.
    occurrence = {}
    groups = []
    for position, index in enumerate(source_grid):
        index = int(index)
        rank = occurrence.get(index, 0)
        occurrence[index] = rank + 1
        if rank == len(groups):
            groups.append([])
        groups[rank].append(position)
    injection = tuple(
        (mx.array(source_grid[positions], dtype=mx.int32),
         mx.array(positions, dtype=mx.int32))
        for positions in groups
    )

    profiles = []
    for n, lo, hi, h, axis in ((ny, top, bottom, spacing[0], 1),
                              (nx, left, right, spacing[1], 2)):
        for offset in (0.0, 0.5):
            a, b = _profile(n, lo, hi, h, dt, pml_freq, max_vel, offset)
            broadcast_shape = [1, 1, 1]
            broadcast_shape[axis] = n + lo + hi
            profiles.append((_reshape(a, broadcast_shape),
                             _reshape(b, broadcast_shape)))
    return {
        "shape": (shots, height, width),
        "padding": [(top, bottom), (left, right)],
        "source_spatial": mx.array(source_spatial.reshape(-1), dtype=mx.int32),
        "receivers": mx.array(receiver_grid, dtype=mx.int32),
        "injection": injection,
        "profiles": tuple(profiles),
        "steps": (_constant(dt), _constant(1.0 / spacing[0]),
                  _constant(1.0 / spacing[1])),
        "zero": _constant(0.0),
        "one": _constant(1.0),
        "two": _constant(2.0),
        "four": _constant(4.0),
    }


def _positive_reciprocal(q, geometry):
    """No reciprocal is evaluated at zero, even in an inactive where branch."""
    active = q[0] > 0.0
    safe = _select(active, q, geometry["one"])
    return _select(active, _div(geometry["one"], safe), geometry["zero"])


def _coefficients(vp, vs, rho, solid, geometry):
    # The vacuum mask is fixed in differentiation. In particular, a vacuum
    # density next to solid must not receive an interface-buoyancy gradient.
    vp = _wide(mx.where(solid, vp, 0.0))
    vs = _wide(mx.where(solid, vs, 0.0))
    rho = _wide(mx.where(solid, rho, 0.0))
    two = geometry["two"]
    vs_squared = _mul(vs, vs)
    mu = _mul(rho, vs_squared)
    lam = _mul(rho, _sub(_mul(vp, vp), _mul(two, vs_squared)))
    padding = geometry["padding"]
    mu = _pad(mu, padding, mode="edge")
    lam = _pad(lam, padding, mode="edge")
    rho = _pad(rho, padding, mode="edge")

    ry = _add(_take(rho, np.s_[:-1, :]), _take(rho, np.s_[1:, :]))
    rx = _add(_take(rho, np.s_[:, :-1]), _take(rho, np.s_[:, 1:]))
    by = _pad(_mul(two, _positive_reciprocal(ry, geometry)),
              [(0, 1), (0, 0)])
    bx = _pad(_mul(two, _positive_reciprocal(rx, geometry)),
              [(0, 0), (0, 1)])
    inv_mu = _positive_reciprocal(mu, geometry)
    active = ((mu[0][:-1, :-1] > 0.0) & (mu[0][1:, :-1] > 0.0)
              & (mu[0][:-1, 1:] > 0.0) & (mu[0][1:, 1:] > 0.0))
    inverse_sum = _add(
        _add(_take(inv_mu, np.s_[:-1, :-1]), _take(inv_mu, np.s_[1:, :-1])),
        _add(_take(inv_mu, np.s_[:-1, 1:]), _take(inv_mu, np.s_[1:, 1:])))
    safe = _select(active, inverse_sum, geometry["one"])
    shear = _select(active, _div(geometry["four"], safe), geometry["zero"])
    shear = _pad(shear, [(0, 1), (0, 1)])
    dt = geometry["steps"][0]
    dtby, dtbx = _mul(dt, by), _mul(dt, bx)
    source_indices = geometry["source_spatial"]
    source_weights = (_take(_reshape(dtby, (-1,)), source_indices),
                      _take(_reshape(dtbx, (-1,)), source_indices))
    return (lam, _mul(two, mu), _mul(dt, shear), dtby, dtbx), source_weights


def _dy_plus(f, invdy):
    neighbor = _map_pair(
        lambda q: mx.concatenate((q[:, 1:, :], mx.zeros_like(q[:, :1, :])), axis=1), f)
    return _mul(_sub(neighbor, f), invdy)


def _dy_minus(f, invdy):
    neighbor = _map_pair(
        lambda q: mx.concatenate((mx.zeros_like(q[:, :1, :]), q[:, :-1, :]), axis=1), f)
    return _mul(_sub(f, neighbor), invdy)


def _dx_plus(f, invdx):
    neighbor = _map_pair(
        lambda q: mx.concatenate((q[:, :, 1:], mx.zeros_like(q[:, :, :1])), axis=2), f)
    return _mul(_sub(neighbor, f), invdx)


def _dx_minus(f, invdx):
    neighbor = _map_pair(
        lambda q: mx.concatenate((mx.zeros_like(q[:, :, :1]), q[:, :, :-1]), axis=2), f)
    return _mul(_sub(f, neighbor), invdx)


def _correct(derivative, memory, profile):
    decay, coupling = profile
    memory = _add(_mul(decay, memory), _mul(coupling, derivative))
    return _add(derivative, memory), memory


def _inject(field, weight, force, injection):
    shape = field[0].shape
    field = _reshape(field, (-1,))
    for destinations, positions in injection:
        increment = _mul(_take(weight, positions), _wide(force[positions]))
        updated = _add(_take(field, destinations), increment)
        # MLX ArrayAt supports reductions, but has no .set operation.
        # Replace each selected high/low value directly: scattering the
        # increment alone would lose the compensated gather/add result.
        # Destinations are unique within each occurrence group, so this
        # functional scatter is deterministic and differentiable.
        field = tuple(mx.put_along_axis(q, destinations, value, axis=0)
                      for q, value in zip(field, updated))
    return _reshape(field, shape)


@mx.compile
def _step(state, coefficients, source_weights, profiles, steps, injection,
          receivers, force_y, force_x):
    """One complete leapfrog step; samples are taken from the incoming state."""
    (vy, vx, syy, sxx, sxy, psi_syy_y, psi_sxy_x, psi_sxy_y, psi_sxx_x,
     psi_vy_y, psi_vx_x, psi_vx_y, psi_vy_x) = state
    lam, two_mu, dt_shear, dtby, dtbx = coefficients
    y_integer, y_half, x_integer, x_half = profiles
    dt, invdy, invdx = steps

    recorded_yy = _take(_reshape(syy, (-1,)), receivers)
    recorded_xx = _take(_reshape(sxx, (-1,)), receivers)
    samples = (_rounded(_take(_reshape(vy, (-1,)), receivers)),
               _rounded(_take(_reshape(vx, (-1,)), receivers)),
               -0.5 * _rounded(_add(recorded_yy, recorded_xx)))

    dsyy_y, psi_syy_y = _correct(_dy_plus(syy, invdy), psi_syy_y, y_half)
    dsxy_x, psi_sxy_x = _correct(_dx_minus(sxy, invdx), psi_sxy_x, x_integer)
    dsxy_y, psi_sxy_y = _correct(_dy_minus(sxy, invdy), psi_sxy_y, y_integer)
    dsxx_x, psi_sxx_x = _correct(_dx_plus(sxx, invdx), psi_sxx_x, x_half)
    vy = _add(vy, _mul(dtby, _add(dsyy_y, dsxy_x)))
    vx = _add(vx, _mul(dtbx, _add(dsxy_y, dsxx_x)))

    # Inject force density after the velocity update, before differentiating
    # velocities for the stress update. There is no cell-area factor.
    weight_y, weight_x = source_weights
    vy = _inject(vy, weight_y, force_y, injection)
    vx = _inject(vx, weight_x, force_x, injection)

    ey, psi_vy_y = _correct(_dy_minus(vy, invdy), psi_vy_y, y_integer)
    ex, psi_vx_x = _correct(_dx_minus(vx, invdx), psi_vx_x, x_integer)
    dvx_y, psi_vx_y = _correct(_dy_plus(vx, invdy), psi_vx_y, y_half)
    dvy_x, psi_vy_x = _correct(_dx_plus(vy, invdx), psi_vy_x, x_half)
    w = _mul(lam, _add(ey, ex))
    syy = _add(syy, _mul(dt, _add(w, _mul(two_mu, ey))))
    sxx = _add(sxx, _mul(dt, _add(w, _mul(two_mu, ex))))
    sxy = _add(sxy, _mul(dt_shear, _add(dvx_y, dvy_x)))
    return ((vy, vx, syy, sxx, sxy, psi_syy_y, psi_sxy_x, psi_sxy_y, psi_sxx_x,
             psi_vy_y, psi_vx_x, psi_vx_y, psi_vy_x), samples)


def _propagate(vp, vs, rho, source_y, source_x, solid, geometry, evaluate):
    coefficients, source_weights = _coefficients(vp, vs, rho, solid, geometry)
    state = tuple(_wide(mx.zeros(geometry["shape"], dtype=mx.float32))
                  for _ in range(13))
    nt = source_y.shape[-1]
    traces = [[], [], []]
    for n in range(nt):
        state, samples = _step(
            state, coefficients, source_weights, geometry["profiles"],
            geometry["steps"], geometry["injection"], geometry["receivers"],
            source_y[..., n].reshape(-1), source_x[..., n].reshape(-1))
        for channel, sample in zip(traces, samples):
            channel.append(sample)
        # Bound the forward lazy graph without retaining full wavefield
        # histories. Do not evaluate during the autodiff transformation.
        if evaluate and (n + 1) % 32 == 0:
            mx.eval(state, [channel[-32:] for channel in traces])
    if nt == 0:
        shape = (*geometry["receivers"].shape, 0)
        return tuple(mx.zeros(shape, dtype=mx.float32) for _ in range(3))
    return tuple(mx.stack(channel, axis=-1) for channel in traces)


def elastic(vp, vs, rho, source_y, source_x, source_locations,
            receiver_locations, spacing, dt, pml_width, pml_freq, max_vel):
    """Return vy, vx, and in-plane pressure at the API's staggered times."""
    geometry = _geometry(vp.shape, source_y, source_x, source_locations,
                         receiver_locations, spacing, dt, pml_width, pml_freq,
                         max_vel)
    solid = mx.array(np.asarray(rho) > 0.0)
    inputs = tuple(mx.array(q, dtype=mx.float32)
                   for q in (vp, vs, rho, source_y, source_x))
    vy, vx, pressure = _propagate(*inputs, solid, geometry, evaluate=True)
    result = {"vy": vy, "vx": vx, "p": pressure}
    mx.eval(result)
    return {name: np.array(value, dtype=np.float32, order="C")
            for name, value in result.items()}


def misfit_and_grad(vp, vs, rho, source_y, source_x, source_locations,
                    receiver_locations, spacing, dt, pml_width, pml_freq,
                    max_vel, observed_y, observed_x, data_scale):
    """Differentiate all uncompressed time steps with respect to five inputs."""
    geometry = _geometry(vp.shape, source_y, source_x, source_locations,
                         receiver_locations, spacing, dt, pml_width, pml_freq,
                         max_vel)
    solid = mx.array(np.asarray(rho) > 0.0)
    obs_y = mx.array(observed_y, dtype=mx.float32)
    obs_x = mx.array(observed_x, dtype=mx.float32)
    scale = mx.array(data_scale, dtype=mx.float32)
    inputs = tuple(mx.array(q, dtype=mx.float32)
                   for q in (vp, vs, rho, source_y, source_x))

    def loss(vp, vs, rho, source_y, source_x):
        vy, vx, _ = _propagate(vp, vs, rho, source_y, source_x, solid,
                               geometry, evaluate=False)
        residual_y, residual_x = (vy - obs_y) / scale, (vx - obs_x) / scale
        squared_error = (mx.sum(residual_y * residual_y)
                         + mx.sum(residual_x * residual_x))
        return squared_error * (0.5 / observed_y.size)

    value, gradients = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(*inputs)
    names = ("vp", "vs", "rho", "source_y", "source_x")
    result = {"objective": value}
    result.update({"grad_" + name: grad for name, grad in zip(names, gradients)})
    mx.eval(result)
    result = {name: np.array(array, dtype=np.float32, order="C")
              for name, array in result.items()}
    result["objective"] = result["objective"][()]
    return result
