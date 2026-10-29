"""Weighted equivalent-source gridding implemented with MLX arrays."""

import mlx.core as mx
import numpy as np


def _jacobian(coordinates, points):
    delta = coordinates[:, None, :] - points[None, :, :]
    return mx.rsqrt(mx.sum(delta * delta, axis=-1))


@mx.custom_function
def _pseudoinverse(matrix):
    """SVD pseudoinverse with a float32 numerical-rank threshold."""
    # MLX's SVD runs on the CPU and has no built-in autodiff rule. All
    # arithmetic stays in MLX; the rule below differentiates the pseudoinverse
    # itself, avoiding singular-vector derivatives at repeated singular values.
    left, singular, right = mx.linalg.svd(matrix, stream=mx.cpu)
    count = min(matrix.shape)
    cutoff = max(matrix.shape) * float(np.finfo(np.float32).eps) * mx.max(singular)
    retained = singular > cutoff
    safe_singular = mx.where(retained, singular, 1.0)
    inverse = mx.where(retained, 1.0 / safe_singular, 0.0)
    return (right[:count, :].T * inverse[None, :]) @ left[:, :count].T


@_pseudoinverse.vjp
def _pseudoinverse_vjp(primals, cotangent, output):
    """Moore-Penrose differential, valid on each fixed-rank stratum."""
    (matrix,) = primals
    inverse = output
    # Transpose of dP = -P dA P + P P.T dA.T (I - A P)
    #                         + (I - P A) dA.T P.T P.
    # Apply the projectors without forming an observation-square identity.
    transpose = cotangent.T
    gradient = (
        -inverse.T @ (cotangent @ inverse.T)
        + (transpose - matrix @ (inverse @ transpose)) @ (inverse @ inverse.T)
        + inverse.T @ (inverse @ (transpose - transpose @ (inverse @ matrix)))
    )
    return (gradient,)


def _fit(jacobian, data, weights, damping):
    """Solve scaled ridge LS; use minimum-norm LS when undamped."""
    count = jacobian.shape[1]
    if count == 0 or jacobian.shape[0] == 0:
        return mx.zeros((count,), dtype=mx.float32)
    centered = jacobian - mx.mean(jacobian, axis=0, keepdims=True)
    variance = mx.mean(centered * centered, axis=0)
    scale = mx.sqrt(mx.where(variance == 0, mx.ones_like(variance), variance))
    root_weights = mx.sqrt(weights)
    matrix = jacobian / scale[None, :] * root_weights[:, None]
    rhs = data * root_weights
    if damping == 0:
        return (_pseudoinverse(matrix) @ rhs) / scale
    # Augmentation yields the ridge objective without squaring the condition
    # number. Keeping the transformations in MLX preserves autodifferentiation.
    matrix = mx.concatenate(
        [matrix, mx.sqrt(mx.array(damping, dtype=mx.float32)) * mx.eye(count)],
        axis=0,
    )
    rhs = mx.concatenate([rhs, mx.zeros((count,), dtype=mx.float32)])
    augmented = mx.concatenate([matrix, rhs[:, None]], axis=1)
    for column in range(count):
        tail = augmented[column:, column:]
        vector = tail[:, 0]
        norm = mx.sqrt(mx.sum(vector * vector))
        sign = mx.where(vector[0] >= 0, 1.0, -1.0)
        vector = mx.concatenate([(vector[0] + sign * norm)[None], vector[1:]])
        denominator = mx.sum(vector * vector)
        safe_denominator = mx.where(denominator == 0, 1.0, denominator)
        transformed = tail - (2.0 / safe_denominator) * vector[:, None] * (
            vector @ tail
        )[None, :]
        # Only the active submatrix is needed by later transformations.
        augmented = mx.concatenate(
            [augmented[:column, :], mx.concatenate(
                [augmented[column:, :column], transformed], axis=1
            )], axis=0
        )

    # Back substitution, expressed without array mutation.
    solution = mx.zeros((0,), dtype=mx.float32)
    for row in range(count - 1, -1, -1):
        remainder = mx.sum(augmented[row, row + 1:count] * solution)
        diagonal = augmented[row, row]
        value = (augmented[row, count] - remainder) / diagonal
        solution = mx.concatenate([value[None], solution])
    return solution / scale


def equivalent_sources(coordinates, data, points, evaluation, weights, damping,
                       method, data_indices, data_offsets, source_indices,
                       source_offsets):
    coordinates = mx.array(coordinates, dtype=mx.float32)
    data = mx.array(data, dtype=mx.float32)
    points = mx.array(points, dtype=mx.float32)
    evaluation = mx.array(evaluation, dtype=mx.float32)
    weights = mx.array(weights, dtype=mx.float32)

    if method == "standard":
        jacobian = _jacobian(coordinates, points)
        evaluation_jacobian = _jacobian(evaluation, points)
        coefficients = _fit(jacobian, data, weights, damping)
        prediction = evaluation_jacobian @ coefficients
    elif method == "boosted":
        data_offsets = np.asarray(data_offsets)
        source_offsets = np.asarray(source_offsets)
        data_indices = np.asarray(data_indices)
        source_indices = np.asarray(source_indices)
        residual = data
        prediction = mx.zeros((evaluation.shape[0],), dtype=mx.float32)
        for window in range(len(data_offsets) - 1):
            row_indices = data_indices[data_offsets[window]:data_offsets[window + 1]]
            col_indices = source_indices[source_offsets[window]:source_offsets[window + 1]]
            # Either empty membership gives a zero update, including the
            # minimum-norm solution for an empty undamped fit.
            if row_indices.size == 0 or col_indices.size == 0:
                continue
            rows = mx.array(row_indices, dtype=mx.int32)
            cols = mx.array(col_indices, dtype=mx.int32)
            window_points = mx.take(points, cols, axis=0)
            window_jacobian = _jacobian(coordinates, window_points)
            coefficients = _fit(mx.take(window_jacobian, rows, axis=0),
                                data=mx.take(residual, rows, axis=0),
                                weights=mx.take(weights, rows, axis=0),
                                damping=damping)
            residual = residual - window_jacobian @ coefficients
            prediction = prediction + _jacobian(evaluation, window_points) @ coefficients
            # Finish each update so lazy graphs do not retain the Jacobians
            # for the entire window schedule during ordinary prediction.
            mx.eval(residual, prediction)
    else:
        raise ValueError(f"Unknown equivalent-sources method: {method}")
    return {"prediction": prediction.astype(mx.float32)}
