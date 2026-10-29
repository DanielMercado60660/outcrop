"""Joint normalized gravity and magnetic inversion using MLX.

Models retain east, north, depth C order. The supplied matrices already encode
the physical units, signs, and survey normalization; no rescaling is applied.
Public results are C-contiguous NumPy float32 arrays.
"""

import mlx.core as mx
import numpy as np


def _array(value):
    return mx.array(value, dtype=mx.float32)


def _numpy_outputs(result):
    """Convert public results without promoting scalar arrays to vectors."""
    return {name: np.array(value, dtype=np.float32, order="C")
            for name, value in result.items()}


def _cross(a, b):
    return mx.stack((a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]))


def predict(density, susceptibility, gravity_matrix, magnetic_matrix):
    return _numpy_outputs({
        "gravity": _array(gravity_matrix) @ _array(density),
        "magnetic": _array(magnetic_matrix) @ _array(susceptibility),
    })


def objective_and_grad(density, susceptibility, gravity_matrix, magnetic_matrix,
                       gravity_data, magnetic_data, derivatives, alpha, smooth,
                       coupling):
    ag, am = _array(gravity_matrix), _array(magnetic_matrix)
    dg, dm, d = _array(gravity_data), _array(magnetic_data), _array(derivatives)
    models = mx.stack((_array(density), _array(susceptibility)))

    def objective(x):
        u, v = x[0], x[1]
        gu, gv = d @ u, d @ v
        cross = _cross(gu, gv)
        data = 0.5 * (mx.sum(mx.square(ag @ u - dg))
                      + mx.sum(mx.square(am @ v - dm)))
        ridge = (0.5 * alpha) * (mx.sum(mx.square(u)) + mx.sum(mx.square(v)))
        grad_penalty = (0.5 * smooth) * (mx.sum(mx.square(gu))
                                         + mx.sum(mx.square(gv)))
        cross_penalty = (0.5 * coupling) * mx.sum(mx.square(cross))
        terms = mx.stack((data, ridge, grad_penalty, cross_penalty))
        return mx.sum(terms), terms, cross

    (value, terms, cross), gradient = mx.value_and_grad(objective)(models)
    return _numpy_outputs({
        "objective": value,
        "terms": terms,
        "grad_density": gradient[0],
        "grad_susceptibility": gradient[1],
        "cross_gradient": cross,
    })


def _cross_matrix(d, model):
    """Matrix mapping q to the component-first stacked Dq cross Dmodel."""
    g = d @ model
    return mx.concatenate((g[2, :, None] * d[1] - g[1, :, None] * d[2],
                           g[0, :, None] * d[2] - g[2, :, None] * d[0],
                           g[1, :, None] * d[0] - g[0, :, None] * d[1]),
                          axis=0)


def _solve(matrix, rhs):
    """Solve an SPD system by Cholesky and two triangular solves."""
    factor = mx.linalg.cholesky(matrix, upper=False, stream=mx.cpu)
    intermediate = mx.linalg.solve_triangular(
        factor, rhs, upper=False, stream=mx.cpu
    )
    return mx.linalg.solve_triangular(
        factor.T, intermediate, upper=True, stream=mx.cpu
    )


def invert(gravity_matrix, magnetic_matrix, gravity_data, magnetic_data,
           derivatives, alpha, smooth, coupling, sweeps):
    """Apply exactly the specified density-first quadratic block sweeps."""
    # Keep assembly on the same backend as the factorization. Selecting the
    # CPU only inside _solve does not move upstream Gram products to the CPU;
    # their roundoff is amplified by these ill-conditioned normal systems.
    # This scoped context restores the caller's stream on exit and keeps all
    # inversion operations in MLX float32, including the coupling matrices.
    with mx.stream(mx.cpu):
        ag, am = _array(gravity_matrix), _array(magnetic_matrix)
        dg, dm, d = _array(gravity_data), _array(magnetic_data), _array(derivatives)
        n = ag.shape[1]
        regularizer = alpha * mx.eye(n, dtype=mx.float32)
        for axis in range(3):
            regularizer = regularizer + smooth * (d[axis].T @ d[axis])
        hg, hm = ag.T @ ag + regularizer, am.T @ am + regularizer
        bg, bm = ag.T @ dg, am.T @ dm
        u, v = _solve(hg, bg), _solve(hm, bm)
        mx.eval(u, v)

        for _ in range(sweeps):
            cv = _cross_matrix(d, v)
            u = _solve(hg + coupling * (cv.T @ cv), bg)
            cu = _cross_matrix(d, u)
            v = _solve(hm + coupling * (cu.T @ cu), bm)
            mx.eval(u, v)

    return _numpy_outputs({"density": u, "susceptibility": v})
