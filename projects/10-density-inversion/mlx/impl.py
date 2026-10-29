"""Density inversion with MLX autodiff and a fixed quadratic penalty.

The supplied sensitivity already contains the gravity units and upward-gz sign.
Only mesh topology is constructed here; no gravity kernel or oracle is used.
"""

import mlx.core as mx


def _regularization(shape, cell_size, depth_offset, alpha_xyz):
    """Return depth weights and each interior face, in x-fastest ordering."""
    nx, ny, nz = shape
    n_cells = nx * ny * nz
    # A C-order (z, y, x) view is the specified Fortran-order (x, y, z) view.
    cells = mx.arange(n_cells, dtype=mx.int32).reshape(nz, ny, nx)
    # Cancel a common length scale before converting either length to float32.
    # At least one scaled length is 1, so the denominator stays positive even
    # if the other is too small for float32. This also avoids large depths.
    length_scale = max(cell_size, depth_offset)
    scaled_cell = cell_size / length_scale
    scaled_offset = depth_offset / length_scale
    depth = (nz - mx.arange(nz, dtype=mx.float32) - 0.5) * scaled_cell
    layer_weights = (0.5 * scaled_cell + scaled_offset) / (depth + scaled_offset)
    weights = mx.broadcast_to(layer_weights[:, None, None], (nz, ny, nx))
    weights = weights.reshape(-1)
    squared = mx.square(weights)
    alpha = mx.array(alpha_xyz, dtype=mx.float32)

    left = (cells[:, :, :-1], cells[:, :-1, :], cells[:-1, :, :])
    right = (cells[:, :, 1:], cells[:, 1:, :], cells[1:, :, :])
    starts, ends, coefficients = [], [], []
    for axis, (lo, hi) in enumerate(zip(left, right)):
        i, j = lo.reshape(-1), hi.reshape(-1)
        starts.append(i)
        ends.append(j)
        coefficients.append(0.5 * alpha[axis] * (squared[i] + squared[j]))
    return (
        weights,
        mx.concatenate(starts),
        mx.concatenate(ends),
        mx.concatenate(coefficients),
    )


def _model_penalty(delta, weights, starts, ends, coefficients):
    """Full squared penalty, including the reference in every difference."""
    smallness = mx.sum(mx.square(weights * delta))
    differences = delta[ends] - delta[starts]
    smoothness = mx.sum(coefficients * mx.square(differences))
    return smallness + smoothness


def _objective_function(matrix, data, uncertainty, reference, regularization, beta):
    """Make one objective for public gradients and solver residual refinement."""
    weights, starts, ends, coefficients = regularization

    def objective(model):
        residual = (matrix @ model - data) / uncertainty
        phi_d = mx.sum(mx.square(residual))
        phi_m = _model_penalty(
            model - reference, weights, starts, ends, coefficients
        )
        return phi_d + beta * phi_m, phi_d, phi_m

    return objective


def apply_operator(sensitivity, vector, transpose=False):
    """Apply G or G.T without changing its units or adding data weights."""
    matrix = mx.array(sensitivity, dtype=mx.float32)
    vector = mx.array(vector, dtype=mx.float32)
    return {"result": (matrix.T if transpose else matrix) @ vector}


def misfit_and_grad(
    sensitivity, data, uncertainty, model, reference, shape,
    cell_size, depth_offset, alpha_xyz, beta,
):
    """Evaluate the specified full squared objective and its MLX gradient."""
    matrix = mx.array(sensitivity, dtype=mx.float32)
    observed = mx.array(data, dtype=mx.float32)
    sigma = mx.array(uncertainty, dtype=mx.float32)
    density = mx.array(model, dtype=mx.float32)
    center = mx.array(reference, dtype=mx.float32)
    regularization = _regularization(shape, cell_size, depth_offset, alpha_xyz)
    objective = _objective_function(
        matrix, observed, sigma, center, regularization, beta
    )
    (value, phi_d, phi_m), gradient = mx.value_and_grad(objective)(density)
    return {
        "objective": value,
        "phi_d": phi_d,
        "phi_m": phi_m,
        "gradient": gradient,
        "weights": regularization[0],
    }


def invert(
    sensitivity, data, uncertainty, reference, shape, cell_size,
    depth_offset, alpha_xyz, beta,
):
    """Solve the unconstrained least-squares problem at the supplied beta.

    An augmented QR avoids forming ill-conditioned normal equations. MLX's
    factorization and triangular solves run on its CPU stream; operator,
    objective and autodiff arithmetic use the default MLX device. Four residual
    corrections use the original objective to remove float32 assembly error.
    """
    matrix = mx.array(sensitivity, dtype=mx.float32)
    observed = mx.array(data, dtype=mx.float32)
    sigma = mx.array(uncertainty, dtype=mx.float32)
    center = mx.array(reference, dtype=mx.float32)
    regularization = _regularization(shape, cell_size, depth_offset, alpha_xyz)
    weights, starts, ends, coefficients = regularization
    n_cells = matrix.shape[1]

    # L.T @ L is precisely the smallness plus interior-face precision matrix.
    identity = mx.eye(n_cells, dtype=mx.float32)
    smallness = weights[:, None] * identity
    edges = mx.sqrt(coefficients)[:, None] * (identity[ends] - identity[starts])
    penalty_factor = mx.concatenate((smallness, edges), axis=0)
    augmented = mx.concatenate(
        (
            matrix / sigma[:, None],
            mx.sqrt(mx.array(beta, dtype=mx.float32)) * penalty_factor,
        ),
        axis=0,
    )
    # Solve for delta, but always predict using the full model reference+delta.
    target = mx.concatenate(
        (
            (observed - matrix @ center) / sigma,
            mx.zeros((penalty_factor.shape[0],), dtype=mx.float32),
        )
    )
    q, r = mx.linalg.qr(augmented, stream=mx.cpu)
    delta = mx.linalg.solve_triangular(r, q.T @ target, upper=True, stream=mx.cpu)
    model = center + delta

    objective = _objective_function(
        matrix, observed, sigma, center, regularization, beta
    )
    value_gradient = mx.value_and_grad(objective)
    for _ in range(4):
        _, gradient = value_gradient(model)
        # Hessian(objective) = 2 * R.T @ R. Do not use a rounded H @ m - rhs
        # residual: cancellation there would compromise model cross-parity.
        temporary = mx.linalg.solve_triangular(
            r.T, -0.5 * gradient, upper=False, stream=mx.cpu
        )
        correction = mx.linalg.solve_triangular(
            r, temporary, upper=True, stream=mx.cpu
        )
        model = model + correction

    return {"model": model, "predicted": matrix @ model}
