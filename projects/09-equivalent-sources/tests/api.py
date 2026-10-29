"""Array-only contract shared by the MLX and Metal implementations."""


def equivalent_sources(coordinates, data, points, evaluation, weights, damping,
                       method, data_indices, data_offsets, source_indices,
                       source_offsets):
    """Fit equivalent point sources and predict at evaluation coordinates.

    coordinates : float32 ndarray, shape (n, 3)
        Observation coordinates in metres, columns easting, northing, upward.
    data : float32 ndarray, shape (n,)
        Observed scalar field, in arbitrary but consistent field units.
    points : float32 ndarray, shape (m, 3)
        Explicit source coordinates in metres, same axes as coordinates.
    evaluation : float32 ndarray, shape (k, 3)
        Prediction coordinates in metres, same axes; returned in this order.
    weights : float32 ndarray, shape (n,)
        Positive least-squares sample weights (inverse variance convention).
    damping : float
        Nonnegative ridge coefficient in the objective below.
    method : str
        'standard' for one global fit; 'boosted' for sequential window fits.
    data_indices, source_indices : int32 ndarray, shape (total memberships,)
        Zero-based observation and source row indices respectively, concatenated
        in processing order. Indices may recur across overlapping windows.
    data_offsets, source_offsets : int32 ndarray, shape (number of windows + 1,)
        Unitless offsets into their respective index arrays: window j uses
        indices[offsets[j]:offsets[j + 1]]. Both start at zero. For standard
        fits the index arrays are empty and each offsets array is [0].

    For each fit form J[i,j] = 1 / distance(coordinates[i], points[j]).
    Let s[j] be the unweighted population standard deviation (ddof=0) of
    column j about its unweighted column mean; use 1 for a zero scale. Do
    not subtract that mean from J. With S = diag(s), solve with no intercept:

        minimize_beta sum_i weights[i] * (data[i] - (J S^-1 beta)[i])**2
                      + damping * ||beta||_2**2,
        coefficients c = S^-1 beta.

    For boosted fits, repeat this solve in the supplied order using each
    window's data rows, source columns, weights, and current residual. Recompute
    the unweighted scales and apply damping independently in every window.
    Add its coefficients to the corresponding sources and subtract its field
    from the residual at every observation before processing the next window.

    Returns
    -------
    dict[str, numpy.ndarray]
        {'prediction': float32 ndarray of shape (k,)} in the same field units
        as data, computed as the evaluation Jacobian times the accumulated
        source coefficients. Distances use all three Cartesian coordinates.
    """
    raise NotImplementedError
