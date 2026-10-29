"""Velocity Born modeling and reorthogonalized CGLS in MLX.

The spatial operator is the full CFS-PML operator at every updated node.
The reverse recurrence is obtained with mx.grad of the linear time step,
including its receiver sampling. No forward scattered-state tape is needed:
the derivative of this recurrence is independent of the scattered state.
"""

import math
from collections import OrderedDict

import mlx.core as mx
import numpy as np


def _profile(n, width, h, dt, freq, vmax):
    """Host-side, fixed one-dimensional coefficients (including guards)."""
    i = np.arange(n, dtype=np.float64)
    left, right = width + 2, n - 1 - width - 2
    ql = np.clip((left - i) / width, 0, 1)
    qh = np.clip((i - right) / width, 0, 1)
    q = np.maximum(ql, qh)
    sigma = 3 * vmax * np.log(1000) / (2 * width * h) * (ql**2 + qh**2)
    alpha = np.pi * freq * (1 - q)
    decay = -(sigma + alpha) * dt
    a = np.where(q > 0, np.exp(decay), 0)
    ratio = np.divide(sigma, sigma + alpha, out=np.zeros_like(sigma),
                      where=(sigma + alpha) != 0)
    b = np.where(q > 0, ratio * np.expm1(decay), 0)
    db = np.zeros_like(b)
    db[2:-2] = ((2 / 3) * (b[3:-1] - b[1:-3])
                 - (1 / 12) * (b[4:] - b[:-4])) / h
    return [mx.array(x.astype(np.float32)) for x in (a, b, db)]


def _inside(a):
    return a[..., 2:-2, 2:-2]


def _guards(a):
    return mx.pad(a, [(0, 0)] * (a.ndim - 2) + [(2, 2), (2, 2)])


def _derivatives(q, axis, h):
    c = _inside(q)
    if axis == 0:
        plus, minus = q[..., 3:-1, 2:-2], q[..., 1:-3, 2:-2]
        plus2, minus2 = q[..., 4:, 2:-2], q[..., :-4, 2:-2]
    else:
        plus, minus = q[..., 2:-2, 3:-1], q[..., 2:-2, 1:-3]
        plus2, minus2 = q[..., 2:-2, 4:], q[..., 2:-2, :-4]
    d1 = ((2 / 3) * (plus - minus) - (1 / 12) * (plus2 - minus2)) / h
    d2 = ((4 / 3) * ((plus - c) + (minus - c))
          - (1 / 12) * ((plus2 - c) + (minus2 - c))) / (h * h)
    return _guards(d1), _guards(d2)


def _d1(q, axis, h):
    if axis == 0:
        a, b = q[..., 3:-1, 2:-2], q[..., 1:-3, 2:-2]
        c, d = q[..., 4:, 2:-2], q[..., :-4, 2:-2]
    else:
        a, b = q[..., 2:-2, 3:-1], q[..., 2:-2, 1:-3]
        c, d = q[..., 2:-2, 4:], q[..., 2:-2, :-4]
    return _guards(((2 / 3) * (a - b) - (1 / 12) * (c - d)) / h)


class _Born:
    def __init__(self, v, mask, source_amplitudes, source_locations,
                 receiver_locations, spacing, dt, pml_width, pml_freq, accuracy):
        if accuracy != 4 or pml_width <= 0:
            raise ValueError("Only fourth-order space and positive PML are supported")
        vmax = float(np.max(v))
        ratio = dt * vmax * math.sqrt(sum(float(h)**-2 for h in spacing)) / 0.6
        if math.ceil(ratio) > 1:
            raise ValueError("This API requires no internal time substeps (R=1)")
        self.shape = tuple(v.shape)
        self.mask = mx.array(mask, dtype=mx.float32)
        self.h = tuple(float(h) for h in spacing)
        offset = pml_width + 2
        nz, nx = self.shape
        self.mz, self.mx = nz + 2 * offset, nx + 2 * offset
        self.iz = mx.array(np.clip(np.arange(self.mz) - offset, 0, nz - 1),
                           dtype=mx.int32)
        self.ix = mx.array(np.clip(np.arange(self.mx) - offset, 0, nx - 1),
                           dtype=mx.int32)
        self.v = self.extend(mx.array(v, dtype=mx.float32))
        self.c = dt**2 * self.v * self.v
        self.tangent_scale = (2 * dt**2) * self.v
        self.profiles = []
        for axis, (n, h) in enumerate(zip((self.mz, self.mx), self.h)):
            prof = _profile(n, pml_width, h, dt, pml_freq, vmax)
            shape = (n, 1) if axis == 0 else (1, n)
            self.profiles.append(tuple(x.reshape(shape) for x in prof))
        self.nshot, _, self.nt = source_amplitudes.shape
        self.data_shape = (self.nshot, receiver_locations.shape[1], self.nt)
        self.zero_state = mx.zeros((6, self.nshot, self.mz, self.mx), mx.float32)
        self.zero_dv = mx.zeros((self.mz, self.mx), mx.float32)
        self.shots = mx.arange(self.nshot)[:, None]
        self.sz = mx.array(source_locations[..., 0] + offset, dtype=mx.int32)
        self.sx = mx.array(source_locations[..., 1] + offset, dtype=mx.int32)
        self.rz = mx.array(receiver_locations[..., 0] + offset, dtype=mx.int32)
        self.rx = mx.array(receiver_locations[..., 1] + offset, dtype=mx.int32)

        # Compiling a single step bounds graph size across all 680 time samples.
        self.step = mx.compile(self._step)
        self.background_step = mx.compile(self._background_step)

        def pairing(state, dv, drive, cotangent, datum):
            advanced = self._step(state, drive * dv)
            return (mx.sum(advanced * cotangent)
                    + mx.sum(self.sample(state[0]) * datum))

        # This is the exact local reverse-mode chain rule, not a reverse-time
        # approximation to the damped PDE. All six states are differentiated.
        self.reverse_step = mx.compile(mx.grad(pairing, argnums=(0, 1)))

        def extension_pairing(model, padded_gradient):
            return mx.sum(self.extend(self.mask * model) * padded_gradient)

        self.fold = mx.compile(mx.grad(extension_pairing, argnums=0))
        self.drives = []
        amplitudes = mx.array(source_amplitudes, dtype=mx.float32)
        state = self.zero_state
        for n in range(self.nt):
            state, drive = self.background_step(state, amplitudes[..., n])
            self.drives.append(drive)
            if (n + 1) % 16 == 0:
                mx.eval(state, self.drives[-16:])
        mx.eval(state, self.drives)

    def extend(self, model):
        return mx.take(mx.take(model, self.iz, axis=0), self.ix, axis=1)

    def sample(self, field):
        return field[self.shots, self.rz, self.rx]

    def _spatial(self, state):
        q = state[0]
        total = mx.zeros_like(q)
        psis, zetas = [], []
        for axis in range(2):
            a, b, db = self.profiles[axis]
            psi, zeta = state[2 + axis], state[4 + axis]
            g, d2 = _derivatives(q, axis, self.h[axis])
            t = (1 + b) * d2 + db * g + _d1(a * psi, axis, self.h[axis])
            total = total + (1 + b) * t + a * zeta
            psis.append(_guards(_inside(b * g + a * psi)))
            zetas.append(_guards(_inside(b * t + a * zeta)))
        return total, psis, zetas

    def _assemble(self, state, spatial, psis, zetas, force):
        new = _guards(_inside(2 * state[0] - state[1] + self.c * spatial + force))
        return mx.stack([new, state[0], *psis, *zetas])

    def _step(self, state, force):
        spatial, psis, zetas = self._spatial(state)
        return self._assemble(state, spatial, psis, zetas, force)

    def _background_step(self, state, amplitude):
        source = mx.zeros((self.nshot, self.mz, self.mx), mx.float32)
        source = source.at[self.shots, self.sz, self.sx].add(amplitude)
        spatial, psis, zetas = self._spatial(state)
        drive = self.tangent_scale * (spatial - source)
        return self._assemble(state, spatial, psis, zetas, -self.c * source), drive

    def forward(self, model):
        if self.nt == 0:
            return mx.zeros(self.data_shape, dtype=mx.float32)
        dv = self.extend(self.mask * model)
        state, records = self.zero_state, []
        for n, drive in enumerate(self.drives):
            # Sampling precedes injection and advancement, including at n=0.
            records.append(self.sample(state[0]))
            state = self.step(state, drive * dv)
            if (n + 1) % 16 == 0:
                mx.eval(state, records[-16:])
        result = mx.stack(records, axis=-1)
        mx.eval(result)
        return result

    def transpose(self, data):
        if self.nt == 0:
            return mx.zeros(self.shape, dtype=mx.float32)
        bars = self.zero_state
        gradient = self.zero_dv
        for count, n in enumerate(range(self.nt - 1, -1, -1)):
            bars, increment = self.reverse_step(
                self.zero_state, self.zero_dv, self.drives[n], bars, data[..., n])
            gradient = gradient + increment
            if (count + 1) % 16 == 0:
                mx.eval(bars, gradient)
        result = self.fold(mx.zeros(self.shape, mx.float32), gradient)
        mx.eval(result)
        return result


# Cache only survey/background setup, never predictions or inversion results.
# Keys contain every operator input, so changed surveys cannot reuse stale B_n.
_OPERATORS = OrderedDict()


def _operator(v, mask, source_amplitudes, source_locations, receiver_locations,
              spacing, dt, pml_width, pml_freq, accuracy):
    arrays = (v, mask, source_amplitudes, source_locations, receiver_locations)
    key = (tuple((a.shape, a.dtype.str, a.tobytes()) for a in arrays),
           tuple(spacing), dt, pml_width, pml_freq, accuracy)
    if key not in _OPERATORS:
        # Bound retained background fields when fixtures alternate surveys.
        if len(_OPERATORS) >= 2:
            _OPERATORS.popitem(last=False)
        _OPERATORS[key] = _Born(*arrays, spacing, dt, pml_width, pml_freq, accuracy)
    _OPERATORS.move_to_end(key)
    return _OPERATORS[key]


def _numpy(**arrays):
    mx.eval(*arrays.values())
    return {name: np.asarray(a, dtype=np.float32) for name, a in arrays.items()}


def born_model(v, model, mask, source_amplitudes, source_locations,
               receiver_locations, spacing, dt, pml_width, pml_freq, accuracy):
    op = _operator(v, mask, source_amplitudes, source_locations, receiver_locations,
                   spacing, dt, pml_width, pml_freq, accuracy)
    return _numpy(predicted=op.forward(mx.array(model, dtype=mx.float32)))


def misfit_and_grad(v, model, mask, observed, source_amplitudes, source_locations,
                    receiver_locations, spacing, dt, pml_width, pml_freq, accuracy):
    op = _operator(v, mask, source_amplitudes, source_locations, receiver_locations,
                   spacing, dt, pml_width, pml_freq, accuracy)
    predicted = op.forward(mx.array(model, dtype=mx.float32))
    residual = predicted - mx.array(observed, dtype=mx.float32)
    objective = 0.5 * mx.sum(residual * residual)
    gradient = op.transpose(residual)
    return _numpy(objective=objective, gradient=gradient, predicted=predicted)


@mx.compile
def _orthogonalize(p, q, pi, qi):
    beta = mx.sum(qi * q)
    return p - beta * pi, q - beta * qi


def lsrtm(v, mask, observed, source_amplitudes, source_locations,
          receiver_locations, spacing, dt, pml_width, pml_freq, accuracy, iterations):
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if not np.any(observed):
        return {"image": np.zeros_like(v, dtype=np.float32),
                "rtm": np.zeros_like(v, dtype=np.float32),
                "predicted": np.zeros_like(observed, dtype=np.float32),
                "residual_history": np.zeros(iterations + 1, dtype=np.float32)}
    op = _operator(v, mask, source_amplitudes, source_locations, receiver_locations,
                   spacing, dt, pml_width, pml_freq, accuracy)
    data = mx.array(observed, dtype=mx.float32)
    image = mx.zeros(v.shape, mx.float32)
    predicted = mx.zeros_like(data)
    residual = data
    data_norm = mx.sqrt(mx.sum(data * data))
    rtm = op.transpose(data)
    history = [mx.array(1.0, dtype=mx.float32)]
    basis = []
    for k in range(iterations):
        p = rtm if k == 0 else op.transpose(residual)
        q = op.forward(p)
        for _ in range(2):
            for pi, qi in basis:
                p, q = _orthogonalize(p, q, pi, qi)
            mx.eval(p, q)
        scale = mx.sqrt(mx.sum(q * q))
        scale_value = scale.item()
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            raise RuntimeError(
                f"CGLS breakdown at iteration {k + 1}: "
                "search-direction norm is zero or non-finite"
            )
        p, q = p / scale, q / scale
        alpha = mx.sum(residual * q)
        image = image + alpha * p
        mx.eval(image, p, q)
        basis.append((p, q))
        predicted = op.forward(image)
        residual = data - predicted
        history.append(mx.sqrt(mx.sum(residual * residual)) / data_norm)
        mx.eval(residual, history[-1])
    return _numpy(image=image, rtm=rtm, predicted=predicted,
                  residual_history=mx.stack(history))
