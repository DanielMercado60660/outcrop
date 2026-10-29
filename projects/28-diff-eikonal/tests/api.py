"""Frozen common API for the MLX autodiff and Rust/Metal adjoint paths."""


def misfit_and_grad(slowness, spacing, sources, receivers, observed, weights):
    """Solve 2D first-arrival eikonal equations and differentiate an L2 objective.

    Parameters
    ----------
    slowness : numpy.ndarray, float32, shape (nz, nx)
        Strictly positive node slowness in seconds/metre. Axes are (z, x),
        z positive down, x positive right; grid origin is (0, 0).
    spacing : list of two Python floats
        [dz, dx] in metres, both positive; rectangular cells are supported.
    sources : numpy.ndarray, int32, shape (ns, 2)
        In-bounds (iz, ix) node indices. Each row is a SEPARATE shot with its
        own T[source]=0 Dirichlet condition, independent of local slowness.
    receivers : numpy.ndarray, int32, shape (nr, 2)
        In-bounds (iz, ix) indices shared by all shots. Sampling is at nodes;
        repeated receivers accumulate separately in the loss and adjoint.
    observed : numpy.ndarray, float32, shape (ns, nr)
        Observed first-arrival times in seconds.
    weights : numpy.ndarray, float32, shape (ns, nr)
        Nonnegative dimensionless loss weights (zero is permitted).

    Returns
    -------
    dict of numpy.ndarray
        "traveltime": float32 (ns, nz, nx), seconds.
        "predicted": float32 (ns, nr), seconds, sampled from traveltime.
        "objective": float32 scalar, seconds squared,
            0.5 * sum(weights * (predicted - observed)**2), WITHOUT averaging.
        "gradient": float32 (nz, nx), derivative of objective with respect
            to slowness (units seconds*metres), summed across shots.

    Discretization
    --------------
    At every nonsource node solve
        sum_d (max(T_i - min(T_{i-e_d}, T_{i+e_d}), 0) / h_d)**2 = s_i**2.
    Missing exterior neighbors are +infinity (no periodic wrap or ghost
    sources). Use the first-order causal Godunov root and converge fast
    sweeping to this fixed point. Evaluate the root in FLOAT32 USING THE
    INCREMENT over the smaller axis minimum, as specified below. Source times
    are exactly zero, not a finite-radius level set. No smoothing,
    higher-order stencils, continuum
    ray gradients, regularization, or velocity chain rule is included.
    The gradient is the derivative of the converged discrete solution;
    MLX uses autodiff through sweeping, Metal uses its discrete adjoint.
    At equal minimizing neighbors choose the lower index; test perturbations
    avoid active nondifferentiable ties. All inputs must remain unmodified.

    Required numerically stable local update
    ----------------------------------------
    Sort the two axis minima and their corresponding spacings as (a, ha),
    (b, hb), with a <= b. If a is infinite, retain infinity. If
    a + s_i*ha <= b, return a + s_i*ha. Otherwise set delta = b - a and
        A = 1/ha**2 + 1/hb**2
        B = -2*delta/hb**2
        C = delta**2/hb**2 - s_i**2
        u = (-B + sqrt(B**2 - 4*A*C)) / (2*A)
        T_new = a + u.
    Algebraically equivalent stable increment formulas are permitted. Do not
    expand the quadratic in absolute T: cancellation in that evaluation can
    satisfy field parity yet corrupt float32 centered loss differences above
    the required 1e-3 gradient-check error. This arithmetic requirement is part
    of the API, in addition to the discrete equation and output tolerances.

    Repeated calls with identical slowness, spacing and sources must return
    bitwise-identical traveltime and predicted arrays, including when only
    observed or weights change. In particular, using a returned predicted
    array as observed must yield exactly zero objective and gradient. Use a
    deterministic sweep schedule; unordered floating-point updates must not
    change the converged arrivals between calls.
    """
    raise NotImplementedError
