"""Cell-centred 2.5-D DC resistivity modelling and inversion.

Material mapping, assembly, solves, quadrature, objective, and autodiff use
MLX float32 on the default device. NumPy/SciPy only prepare fixed geometry
and drive the optimizer; no model-dependent calculation uses a CPU solve.
Solver contractions use explicit float32 products and reductions, independent
of MLX's process-wide reduced-precision matrix-multiplication setting.
"""

import mlx.core as mx
import numpy as np
from scipy.optimize import minimize
from scipy.special import k0e, k1e


def _matmul_float32(a, b):
    """Batched matrix product without reduced-mantissa GEMM dispatch.

    MLX 0.32 can route float32 ``matmul`` through reduced-precision hardware.
    That is especially damaging to iterative refinement: an inaccurate A@x
    makes a correction spoil an already accurate factorization solve. The
    same issue affects the implicit derivative's adjoint outer products.

    Elementwise float32 multiplication followed by a float32 reduction keeps
    all operand bits. Blocking output rows bounds the temporary storage; the
    final two axes have ordinary matrix-multiplication semantics. Do not
    replace this with matmul/einsum or compile it into a GEMM. No environment
    override is needed, even if MLX was used before this module was imported.
    """
    blocks = []
    for start in range(0, a.shape[-2], 16):
        products = a[..., start:start + 16, :, None] * b[..., None, :, :]
        blocks.append(mx.sum(products, axis=-2))
    return mx.concatenate(blocks, axis=-2)


def _ldlt(a):
    """Factor batched SPD matrices using only GPU-supported MLX operations.

    MLX 0.32's Metal LU and Cholesky backends are not implemented. Symmetric
    elimination gives A = L diag(d) L.T without pivoting for this positive
    definite Robin problem. Store each subdiagonal column of unit-diagonal L.
    The outer product is formed symmetrically to preserve symmetry in float32.
    """
    diagonal, lower = [], []
    for _ in range(a.shape[-1] - 1):
        pivot = a[..., 0, 0]
        column = a[..., 1:, 0]
        diagonal.append(pivot)
        lower.append(column / pivot[..., None])
        update = (column[..., :, None] * column[..., None, :]) / pivot[..., None, None]
        a = a[..., 1:, 1:] - update
    diagonal.append(a[..., 0, 0])
    return diagonal, lower


def _substitute(diagonal, lower, b):
    """Forward substitution, diagonal division, then backward substitution."""
    rows = []
    for pivot, column in zip(diagonal[:-1], lower):
        first = b[..., 0, :]
        rows.append(first / pivot[..., None])
        b = b[..., 1:, :] - column[..., :, None] * first[..., None, :]
    rows.append(b[..., 0, :] / diagonal[-1][..., None])
    x = rows[-1][..., None, :]
    for row, column in zip(reversed(rows[:-1]), reversed(lower)):
        first = row - mx.sum(column[..., :, None] * x, axis=-2)
        x = mx.concatenate((first[..., None, :], x), axis=-2)
    return x


def _gpu_solve(a, b):
    """Equilibrated float32 direct solve, with two residual corrections.

    Diagonal equilibration handles the wide range of cell sizes and Fourier
    mass terms. Reusing the factors for residual corrections improves the
    solution of the original assembled system, including scaling roundoff.
    Residuals must use full-mantissa products, not reduced-precision GEMM.
    There is no residual-based early exit and no CPU/fp64 fallback.
    """
    scale = mx.sqrt(mx.diagonal(a, axis1=-2, axis2=-1))
    scaled = a / (scale[..., :, None] * scale[..., None, :])
    diagonal, lower = _ldlt(scaled)

    def apply(rhs):
        return _substitute(diagonal, lower, rhs / scale[..., :, None]) / scale[..., :, None]

    x = apply(b)
    for _ in range(2):
        x = x + apply(b - _matmul_float32(a, x))
    return x


@mx.custom_function
def _solve(a, b):
    """Differentiable batched solve for multiple source columns."""
    return _gpu_solve(a, b)


@_solve.vjp
def _solve_vjp(primals, cotangent, output):
    a, _ = primals
    # The implicit solve derivative is independent of the factorization and
    # residual-correction algorithm. MLX differentiates the full material and
    # assembly graph through this primitive, including all log10 chain rules.
    adj = _gpu_solve(mx.swapaxes(a, -1, -2), cotangent)
    return -_matmul_float32(adj, mx.swapaxes(output, -1, -2)), adj


class _Problem:
    def __init__(self, hx, hz, origin, quadrature, electrodes, abmn, current):
        # Model-independent host setup only. Accumulate coordinates and evaluate
        # Bessel ratios here, then round the fixed coefficients once to float32.
        hx = np.asarray(hx, dtype=np.float32)
        hz = np.asarray(hz, dtype=np.float32)
        origin = np.asarray(origin, dtype=np.float32)
        electrodes = np.asarray(electrodes, dtype=np.float32)
        abmn = np.asarray(abmn, dtype=np.int32)
        quad = np.asarray(quadrature, dtype=np.float32)
        nx, nz = len(hx), len(hz)
        n = nx * nz
        xn = origin[0] + np.r_[0., np.cumsum(hx)]
        zn = origin[1] + np.r_[0., np.cumsum(hz)]
        xc = (xn[:-1] + xn[1:]) / 2
        zc = (zn[:-1] + zn[1:]) / 2
        self.n = n
        self.nk = len(quad)
        self.quad = mx.array(quad[:, 0], dtype=mx.float32)
        self.weights = mx.array(quad[:, 1], dtype=mx.float32)
        self.volume = mx.array(np.outer(hz, hx).ravel(), dtype=mx.float32)

        left, right, hleft, hright, area = [], [], [], [], []
        for iz in range(nz):
            for ix in range(nx - 1):
                left.append(iz * nx + ix)
                right.append(iz * nx + ix + 1)
                hleft.append(hx[ix] / 2)
                hright.append(hx[ix + 1] / 2)
                area.append(hz[iz])
        for iz in range(nz - 1):
            for ix in range(nx):
                left.append(iz * nx + ix)
                right.append((iz + 1) * nx + ix)
                hleft.append(hz[iz] / 2)
                hright.append(hz[iz + 1] / 2)
                area.append(hx[ix])
        left = np.array(left, dtype=np.int32)
        right = np.array(right, dtype=np.int32)
        self.hl = mx.array(np.asarray(hleft, np.float32), dtype=mx.float32)
        self.hr = mx.array(np.asarray(hright, np.float32), dtype=mx.float32)
        self.area = mx.array(np.asarray(area, np.float32), dtype=mx.float32)
        # Each interior face contributes +g at both diagonal entries and -g
        # at the two off-diagonal entries. Index 0 denotes a missing face.
        # Gather the stencil directly instead of multiplying dense incidence
        # matrices, retaining the same face conductance in every entry.
        edge_at = np.zeros((n, n), dtype=np.int32)
        cell_faces = np.zeros((n, 4), dtype=np.int32)
        degree = np.zeros(n, dtype=np.int32)
        for face, (i, j) in enumerate(zip(left, right), start=1):
            edge_at[i, j] = edge_at[j, i] = face
            cell_faces[i, degree[i]] = face
            cell_faces[j, degree[j]] = face
            degree[i] += 1
            degree[j] += 1
        self.edge_at = mx.array(edge_at)
        self.cell_faces = mx.array(cell_faces)
        self.left = mx.array(left)
        self.right = mx.array(right)

        cells, face_area, half_width, distance, normal = [], [], [], [], []
        xmid, ztop = np.median(xn), zn[-1]
        for iz in range(nz):
            for ix, xface, sign in ((0, xn[0], -1), (nx - 1, xn[-1], 1)):
                dx, dz = xface - xmid, zc[iz] - ztop
                r = np.hypot(dx, dz)
                cells.append(iz * nx + ix)
                face_area.append(hz[iz])
                half_width.append(hx[ix] / 2)
                distance.append(r)
                normal.append(sign * dx / r)
        for ix in range(nx):
            dx, dz = xc[ix] - xmid, zn[0] - ztop
            r = np.hypot(dx, dz)
            cells.append(ix)
            face_area.append(hx[ix])
            half_width.append(hz[0] / 2)
            distance.append(r)
            normal.append(-dz / r)
        kr = quad[:, 0, None] * np.array(distance)[None, :]
        alpha = quad[:, 0, None] * (k1e(kr) / k0e(kr)) * np.array(normal)[None, :]
        coefficients = np.array(face_area)[None, :] / (np.array(half_width)[None, :] + 1 / alpha)
        boundary = np.zeros((len(quad), n), dtype=np.float32)
        for j, cell in enumerate(cells):
            boundary[:, cell] += coefficients[:, j]
        self.boundary = mx.array(boundary, dtype=mx.float32)
        self.eye = mx.eye(n, dtype=mx.float32)

        sources, source_map = np.unique(abmn[:, :2], axis=0, return_inverse=True)
        rhs = np.zeros((n, len(sources)), dtype=np.float32)
        nearest_x = np.abs(electrodes[:, 0, None] - xc).argmin(axis=1)
        nearest_z = np.abs(electrodes[:, 1, None] - zc).argmin(axis=1)
        nearest = nearest_z * nx + nearest_x
        for j, (a, b) in enumerate(sources):
            rhs[nearest[a], j] += current
            rhs[nearest[b], j] -= current
        self.rhs = mx.array(rhs, dtype=mx.float32)
        receiver = np.zeros((len(abmn), n), dtype=np.float32)
        for i, row in enumerate(abmn):
            for electrode, sign in ((row[2], 1), (row[3], -1)):
                parts = []
                for centers, value in zip((xc, zc), electrodes[electrode]):
                    if len(centers) == 1:
                        lo, hi, t = 0, 0, 0.0
                    else:
                        hi = int(np.clip(np.searchsorted(centers, value), 1, len(centers) - 1))
                        lo = hi - 1
                        t = np.clip((value - centers[lo]) / (centers[hi] - centers[lo]), 0, 1)
                    parts.append(((lo, 1-t), (hi, t)))
                for iz, wz in parts[1]:
                    for ix, wx in parts[0]:
                        receiver[i, iz * nx + ix] += sign * wx * wz
        self.receiver = mx.array(receiver, dtype=mx.float32)
        self.source_map = mx.array(source_map.astype(np.int32))

    def predict(self, model):
        model = model.astype(mx.float32)
        rho = mx.power(10.0, model.reshape((-1,)))
        g = self.area / (mx.take(rho, self.left) * self.hl
                         + mx.take(rho, self.right) * self.hr)
        faces = mx.concatenate((mx.zeros((1,), dtype=mx.float32), g))
        diagonal = mx.sum(mx.take(faces, self.cell_faces), axis=1)
        lap = -mx.take(faces, self.edge_at) + self.eye * diagonal[None, :]
        diagonal = (self.boundary + self.quad[:, None] ** 2 * self.volume[None, :]) / rho[None, :]
        matrix = lap[None, :, :] + self.eye[None, :, :] * diagonal[:, None, :]
        rhs = mx.broadcast_to(self.rhs[None, :, :], (self.nk, self.n, self.rhs.shape[1]))
        u = _solve(matrix, rhs)
        # Form receiver differences before summing over wavenumbers to
        # avoid cancellation between separately integrated potentials.
        selected = mx.transpose(mx.take(u, self.source_map, axis=2), (0, 2, 1))
        voltage = mx.sum(selected * self.receiver[None, :, :], axis=2)
        return mx.sum(voltage * self.weights[:, None], axis=0)


def dc_predict(hx, hz, origin, quadrature, electrodes, abmn, current, log10_rho):
    problem = _Problem(hx, hz, origin, quadrature, electrodes, abmn, current)
    model = mx.array(log10_rho, dtype=mx.float32)
    return {'predicted': problem.predict(model)}


def _objective(problem, observed, standard_deviation, reference, beta):
    obs = mx.array(observed, dtype=mx.float32)
    sd = mx.array(standard_deviation, dtype=mx.float32)
    ref = mx.array(reference, dtype=mx.float32)

    def objective(model):
        model = model.astype(mx.float32)
        pred = problem.predict(model)
        delta = model - ref
        regularization = (mx.sum(delta ** 2) + mx.sum((delta[1:, :] - delta[:-1, :]) ** 2)
                          + mx.sum((delta[:, 1:] - delta[:, :-1]) ** 2))
        return mx.mean(((pred - obs) / sd) ** 2) / 2 + beta * regularization / 2

    return objective


def dc_misfit_and_grad(hx, hz, origin, quadrature, electrodes, abmn, current,
                       log10_rho, observed, standard_deviation, reference, beta):
    problem = _Problem(hx, hz, origin, quadrature, electrodes, abmn, current)
    model = mx.array(log10_rho, dtype=mx.float32)
    objective = _objective(problem, observed, standard_deviation, reference, beta)
    value, gradient = mx.value_and_grad(objective)(model)
    return {'objective': value, 'gradient': gradient,
            'predicted': problem.predict(model)}


def dc_invert(hx, hz, origin, quadrature, electrodes, abmn, current,
              observed, standard_deviation, initial, active, beta,
              lower_bound, upper_bound, max_iterations):
    problem = _Problem(hx, hz, origin, quadrature, electrodes, abmn, current)
    initial = np.asarray(initial, dtype=np.float32)
    indices = np.flatnonzero(np.asarray(active).ravel())
    objective = _objective(problem, observed, standard_deviation, initial, beta)
    value_grad = mx.value_and_grad(objective)

    def evaluate(x):
        # SciPy owns host optimizer bookkeeping, but every trial, objective,
        # and gradient is evaluated in float32 on the default MLX device.
        model = initial.copy().ravel()
        model[indices] = x
        value, gradient = value_grad(mx.array(model.reshape(initial.shape), dtype=mx.float32))
        mx.eval(value, gradient)
        return float(value.item()), np.asarray(gradient).ravel()[indices].astype(np.float64)

    if len(indices):
        result = minimize(evaluate, initial.ravel()[indices].astype(np.float64), jac=True,
                          method='L-BFGS-B', bounds=[(lower_bound, upper_bound)] * len(indices),
                          options={'maxiter': int(max_iterations), 'ftol': 1e-10, 'gtol': 1e-6,
                                   'maxls': 30, 'maxcor': 20})
        model = initial.copy().ravel()
        model[indices] = result.x
        model = model.reshape(initial.shape)
    else:
        model = initial.copy()
    predicted = problem.predict(mx.array(model, dtype=mx.float32))
    return {'log10_rho': model, 'predicted': predicted}
