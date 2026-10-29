"""Weighted, unconstrained finite-fault slip inversion in float32 MLX.

Bulk arithmetic and objective autodiff run on the GPU. Only the QR and
triangular-solve primitives use the CPU, as documented below.
"""

import mlx.core as mx


@mx.custom_function
def _least_squares(a, b):
    """Solve full-column-rank least squares with a matrix right-hand side.

    A direct QR solve also preserves an exactly zero right-hand side.
    """
    with mx.stream(mx.gpu):
        # MLX's QR/triangular solver needs the CPU backend. This is the isolated
        # float32 solve exception: QR preserves small independent columns that
        # normal equations can lose. All surrounding arithmetic stays on GPU.
        q, r = mx.linalg.qr(a, stream=mx.cpu)
        n = a.shape[1]
        q, r = q[:, :n], r[:n, :]
        return mx.linalg.solve_triangular(r, q.T @ b, upper=True, stream=mx.cpu)


@_least_squares.vjp
def _least_squares_vjp(primals, cotangent, output):
    """Implicit derivative of A.T @ (A @ slip - b) = 0, using QR."""
    a, b = primals
    with mx.stream(mx.gpu):
        # The adjoint uses the same isolated float32 CPU solve exception.
        # Its residual and matrix products are evaluated on the GPU.
        q, r = mx.linalg.qr(a, stream=mx.cpu)
        n = a.shape[1]
        q, r = q[:, :n], r[:n, :]
        # R.T @ y = cotangent; R @ z = y. Never construct A.T @ A.
        y = mx.linalg.solve_triangular(r.T, cotangent, upper=False, stream=mx.cpu)
        z = mx.linalg.solve_triangular(r, y, upper=True, stream=mx.cpu)
        grad_b = q @ y
        residual = b - a @ output
        return residual @ z.T - grad_b @ output.T, grad_b


def invert_slip(green, data, sigma, regularizer, damping, areas, shear_modulus):
    """Return signed slip, displacement predictions, and scalar moment in N m.

    Solve [G / sigma; damping * L] @ slip = [data / sigma; 0]
    in least squares. Rows and columns retain their supplied ordering.
    """
    with mx.stream(mx.gpu):
        g = mx.array(green, dtype=mx.float32)
        d = mx.array(data, dtype=mx.float32)
        sig = mx.array(sigma, dtype=mx.float32)
        reg = mx.array(regularizer, dtype=mx.float32)
        # Whiten observation rows by standard deviation, not variance.
        weighted_g = g / sig[:, None]
        weighted_d = d / sig
        # Squaring this block in the objective supplies damping**2.
        penalty = damping * reg
        augmented_g = mx.concatenate((weighted_g, penalty), axis=0)
        augmented_d = mx.concatenate(
            (weighted_d, mx.zeros((reg.shape[0],), dtype=mx.float32)), axis=0
        )
        slip = _least_squares(augmented_g, augmented_d[:, None])[:, 0]
        predicted = g @ slip
        moment = shear_modulus * mx.sum(mx.array(areas, dtype=mx.float32) * mx.abs(slip))
        return {"slip": slip, "predicted": predicted, "moment": moment}


def misfit_and_grad(green, data, sigma, regularizer, damping, slip):
    """Return scalar J and its float32 autodiff gradient with respect to slip."""
    with mx.stream(mx.gpu):
        g = mx.array(green, dtype=mx.float32)
        d = mx.array(data, dtype=mx.float32)
        sig = mx.array(sigma, dtype=mx.float32)
        reg = mx.array(regularizer, dtype=mx.float32)

        def objective(s):
            residual = (g @ s - d) / sig
            penalty = damping * (reg @ s)
            return 0.5 * (mx.sum(residual * residual) + mx.sum(penalty * penalty))

        value, gradient = mx.value_and_grad(objective)(mx.array(slip, dtype=mx.float32))
        return {"objective": value, "gradient": gradient}
