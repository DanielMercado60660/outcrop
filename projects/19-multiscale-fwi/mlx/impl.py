"""MLX acoustic wave propagation and frequency-continuation inversion.

Propagation, filtering, autodiff, and optimizer arithmetic use float32 on
MLX's default device (GPU). NumPy is used only for input validation, index
and mask setup, and float32 API outputs; there is no CPU float64 path.
"""
import mlx.core as mx
import numpy as np


def _coordinates(name, coordinates, shots, nz, nx, count=None):
    """Validate physical indices before adding padding or converting dtype."""
    coordinates = np.asarray(coordinates)
    if (coordinates.ndim != 3 or coordinates.shape[0] != shots
            or coordinates.shape[2] != 2
            or (count is not None and coordinates.shape[1] != count)):
        raise ValueError(f"{name} must have shape (shots, count, 2) matching the data")
    if not np.issubdtype(coordinates.dtype, np.integer):
        raise ValueError(f"{name} must contain integer [z, x] grid coordinates")
    if (np.any(coordinates < 0) or np.any(coordinates[..., 0] >= nz)
            or np.any(coordinates[..., 1] >= nx)):
        raise ValueError(f"{name} coordinates must lie in the physical grid: 0 <= z < {nz}, 0 <= x < {nx}")
    return coordinates.astype(np.int64, copy=False)


def _profile(n, dx, dt):
    # Fixed CPML coefficients, also evaluated in float32 on the default device.
    # The physical edge is at 14 = 12 PML cells + 2 stencil halo cells.
    x = mx.arange(n, dtype=mx.float32)
    f = mx.clip(mx.maximum((14 - x) / 12, (x - (n - 15)) / 12), 0, 1)
    sigma = (-3 * 3000 / (24 * dx)) * mx.log(mx.array(0.001, dtype=mx.float32)) * f**2
    alpha = np.pi * 12 * (1 - f)
    a = mx.exp(-(sigma + alpha) * dt)
    b = sigma / (sigma + alpha) * (a - 1)
    a = mx.where(f == 0, 0, a)
    db = (8 * (b[3:-1] - b[1:-3]) - (b[4:] - b[:-4])) / (12 * dx)
    return a, b, mx.pad(db, [(2, 2)])


def _derivatives(u, axis, dx):
    if axis == 1:
        c, m1, p1, m2, p2 = (u[:, 2:-2, :], u[:, 1:-3, :], u[:, 3:-1, :], u[:, :-4, :], u[:, 4:, :])
        pads = [(0, 0), (2, 2), (0, 0)]
    else:
        c, m1, p1, m2, p2 = (u[:, :, 2:-2], u[:, :, 1:-3], u[:, :, 3:-1], u[:, :, :-4], u[:, :, 4:])
        pads = [(0, 0), (0, 0), (2, 2)]
    d1 = (8 * (p1 - m1) - (p2 - m2)) / (12 * dx)
    # The fourth-order stencil written as differences annihilates constant
    # fields exactly, even in float32. Summing rounded coefficients times u
    # instead introduces a small DC forcing, amplified at low cutoffs.
    near = (p1 - c) + (m1 - c)
    far = (p2 - c) + (m2 - c)
    d2 = (16 * near - far) / (12 * dx * dx)
    return mx.pad(d1, pads), mx.pad(d2, pads)


@mx.compile
def _step(state, v2dt, forcing, active, profiles, dx):
    u, increment, pz, px, zz, zx = state
    az, bz, dz, ax, bx, dxx = profiles
    uz, uzz = _derivatives(u, 1, dx)
    ux, uxx = _derivatives(u, 2, dx)
    dpz, _ = _derivatives(az * pz, 1, dx)
    dpx, _ = _derivatives(ax * px, 2, dx)
    tz = (1 + bz) * uzz + dz * uz + dpz
    tx = (1 + bx) * uxx + dxx * ux + dpx
    lap = (1 + bz) * tz + az * zz + (1 + bx) * tx + ax * zx
    # Store u[n]-u[n-1] as the second state variable. This is the same
    # second-order recurrence as 2*u[n]-u[n-1]+acceleration, while avoiding
    # subtraction of full-amplitude fields at every step. All six state
    # arrays and their autodiff paths remain float32 on the default device.
    increment = (increment + v2dt * (lap - forcing)) * active
    nxt = (u + increment) * active
    return (nxt, increment, (bz * uz + az * pz) * active,
            (bx * ux + ax * px) * active, (bz * tz + az * zz) * active,
            (bx * tx + ax * zx) * active)


class _Problem:
    def __init__(self, basis, source, sources, receivers, observed, dx, dt, cutoff):
        basis, source, observed = np.asarray(basis), np.asarray(source), np.asarray(observed)
        if basis.ndim != 3 or any(size == 0 for size in basis.shape):
            raise ValueError("basis must have nonempty shape (parameters, nz, nx)")
        if source.ndim != 3:
            raise ValueError("source must have shape (shots, sources_per_shot, time)")
        shots, q, self.nt = source.shape
        if self.nt == 0:
            raise ValueError("source must contain at least one time sample")
        nz, nx = basis.shape[1:]
        sources = _coordinates("sources", sources, shots, nz, nx, count=q)
        receivers = _coordinates("receivers", receivers, shots, nz, nx)
        if observed.shape != (shots, receivers.shape[1], self.nt):
            raise ValueError("observed must have shape (shots, receivers_per_shot, source_time_samples)")
        for name, value in (("dx", dx), ("dt", dt), ("cutoff", cutoff)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.basis = mx.array(basis, dtype=mx.float32)
        self.source = mx.array(source, dtype=mx.float32)
        self.observed = mx.array(observed, dtype=mx.float32)
        self.dx, self.dt = mx.array(dx, dtype=mx.float32), dt
        self.nz, self.nx = nz + 28, nx + 28
        self.zi = mx.array(np.clip(np.arange(self.nz, dtype=np.int64) - 14, 0, nz - 1))
        self.xi = mx.array(np.clip(np.arange(self.nx, dtype=np.int64) - 14, 0, nx - 1))
        self.sz = mx.array(sources[:, :, 0] + 14)
        self.sx = mx.array(sources[:, :, 1] + 14)
        active = np.zeros((1, self.nz, self.nx), np.float32)
        active[:, 2:-2, 2:-2] = 1
        self.active = mx.array(active, dtype=mx.float32)
        self.rshot = mx.array(np.arange(shots, dtype=np.int64)[:, None])
        self.rz = mx.array(receivers[:, :, 0] + 14)
        self.rx = mx.array(receivers[:, :, 1] + 14)
        az, bz, dz = _profile(self.nz, dx, dt)
        ax, bx, dxx = _profile(self.nx, dx, dt)
        self.az, self.bz, self.dz = az[None, :, None], bz[None, :, None], dz[None, :, None]
        self.ax, self.bx, self.dxx = ax[None, None, :], bx[None, None, :], dxx[None, None, :]
        self.profiles = (self.az, self.bz, self.dz, self.ax, self.bx, self.dxx)
        # Parseval gives the same length-nt periodic real-FFT filter energy:
        #   ||H x||^2 = sum_k multiplicity[k] * H[k]^2 * |DFT(x)[k]|^2 / nt.
        # DC and (for even nt) Nyquist occur once; other bins occur twice.
        bins = mx.arange(self.nt // 2 + 1, dtype=mx.float32)
        frequencies = bins / (self.nt * dt)
        response = 1 / mx.sqrt(1 + (frequencies / cutoff)**8)
        multiplicity = np.full(bins.size, 2.0, dtype=np.float32)
        multiplicity[0] = 1
        if self.nt % 2 == 0:
            multiplicity[-1] = 1
        self.spectral_scale = response * mx.sqrt(mx.array(multiplicity) / self.nt)
        self.denominator = self.filtered_energy(self.observed)
        energy = float(self.denominator.item())
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("observed data must have finite, strictly positive filtered energy")
        self.shots = shots

    def filtered_energy(self, x):
        """Squared norm of the specified periodic, zero-phase FFT filter."""
        # The native float32/complex64 FFT supports autodiff. Explicit length
        # and backward normalization preserve the original periodic record.
        spectrum = mx.fft.rfft(x, n=self.nt, axis=-1, norm="backward")
        real = mx.real(spectrum) * self.spectral_scale
        imag = mx.imag(spectrum) * self.spectral_scale
        return mx.sum(real * real + imag * imag)

    def predict(self, p):
        v = mx.sum(p[:, None, None] * self.basis, axis=0)
        v = v[self.zi[:, None], self.xi[None, :]]
        v2dt = v * v * (self.dt * self.dt)
        shape = (self.shots, self.nz, self.nx)
        zero = mx.zeros(shape, dtype=mx.float32)
        state = (zero, zero, zero, zero, zero, zero)
        records = []
        for t in range(self.nt):
            records.append(state[0][self.rshot, self.rz, self.rx])
            # Scatter-add preserves coincident sources without a grid-sized
            # mask per source. Velocity-dependent scaling remains in _step.
            injection = zero.at[self.rshot, self.sz, self.sx].add(self.source[:, :, t])
            state = _step(state, v2dt, injection, self.active, self.profiles, self.dx)
        return mx.stack(records, axis=-1)

    def loss_and_prediction(self, p):
        prediction = self.predict(p)
        energy = self.filtered_energy(prediction - self.observed)
        return 0.5 * energy / self.denominator, prediction

    def objective(self, p):
        energy = self.filtered_energy(self.predict(p) - self.observed)
        return 0.5 * energy / self.denominator


def misfit_and_grad(*, parameters, basis, source, sources, receivers, observed, dx, dt, cutoff):
    problem = _Problem(basis, source, sources, receivers, observed, dx, dt, cutoff)
    p = mx.array(parameters, dtype=mx.float32)
    (value, prediction), gradient = mx.value_and_grad(problem.loss_and_prediction)(p)
    mx.eval(value, prediction, gradient)
    return {"predicted": np.asarray(prediction, dtype=np.float32),
            "objective": np.asarray(value, dtype=np.float32),
            "gradient": np.asarray(gradient, dtype=np.float32)}


def invert(*, initial_parameters, basis, source, sources, receivers, observed, dx, dt, cutoffs, steps_per_stage):
    # Keep parameters, gradients, updates, and Armijo arithmetic on the same
    # float32 device as propagation. Only scalar decisions return to Python.
    p = mx.array(initial_parameters, dtype=mx.float32)
    trajectory = [p]
    history = []
    for cutoff in cutoffs:
        problem = _Problem(basis, source, sources, receivers, observed, dx, dt, cutoff)
        grad_fn = mx.value_and_grad(problem.objective)
        stalled = False
        for _ in range(steps_per_stage):
            if not stalled:
                value, gradient = grad_fn(p)
                mx.eval(value, gradient)
                if value.item() <= 1e-10:
                    stalled = True
                else:
                    direction = -80 * gradient / mx.maximum(mx.max(mx.abs(gradient)), 1e-30)
                    accepted = False
                    for k in range(16):
                        candidate = mx.clip(p + (2.0**-k) * direction, 1500, 2900)
                        trial = problem.objective(candidate)
                        bound = value + 1e-4 * mx.sum(gradient * (candidate - p))
                        if (trial <= bound).item():
                            p = candidate
                            accepted = True
                            break
                    if not accepted:
                        stalled = True
            history.append(value)
            trajectory.append(p)
    states = mx.stack(trajectory)
    values = mx.stack(history) if history else mx.zeros((0,), dtype=mx.float32)
    mx.eval(p, states, values)
    return {"parameters": np.asarray(p, dtype=np.float32),
            "trajectory": np.asarray(states, dtype=np.float32),
            "objective_history": np.asarray(values, dtype=np.float32)}
