"""Common NumPy API: both MLX and Rust/Metal implement all functions.

All array inputs/outputs are float32, C-contiguous NumPy arrays. Scalar inputs
are Python floats. Coordinates are local Cartesian metres: x east, y north,
z down (station elevation is negative z). Times are relative seconds, never
Unix timestamps. B is the event count and N the station count. One constant
positive P-wave speed and independent Gaussian pick errors are assumed.
Implementations must handle every event independently; no random data generation.
"""


def misfit_and_grad(stations, arrivals, sigma, velocity, models):
    """Evaluate weighted arrival-time misfit and its model gradient.

    stations: float32 (N,3), receiver [east,north,down] in m.
    arrivals: float32 (B,N), observed picks in s.
    sigma: float32 (N,), known positive pick standard deviations in s.
    velocity: positive Python float, homogeneous speed in m/s.
    models: float32 (B,4), [x,y,z,t0] (m,m,m,s). No source coincides
        with a station. Predicted pick is t0 + ||source-station||/velocity.

    Return dict: predicted float32 (B,N) in s; objective float32 (B,),
    dimensionless 0.5*sum(((predicted-arrivals)/sigma)**2); gradient float32
    (B,4), derivative of each event's objective w.r.t. its own [x,y,z,t0],
    in 1/m for the first three entries and 1/s for the last. Sum, not mean.
    MLX uses autodiff; Metal may use the analytic Jacobian.
    """
    raise NotImplementedError


def grid_search(stations, arrivals, sigma, velocity, x, y, z):
    """Exhaustive spatial grid search with analytically profiled origin time.

    stations, arrivals, sigma, velocity: as in misfit_and_grad.
    x,y,z: float32 (Nx,), (Ny,), (Nz,), increasing coordinate axes in m.
    Search every Cartesian product node. At each node use
    t0=sum((arrivals-distance/velocity)/sigma**2)/sum(1/sigma**2).
    Minimize the same half chi-square objective as misfit_and_grad.
    Ties choose the first C-order (Nx,Ny,Nz) flattened node (z fastest).

    Return dict: models float32 (B,4), [x,y,z,t0] (m,m,m,s), and
    objective float32 (B,), dimensionless minimum half chi-square.
    """
    raise NotImplementedError


def geiger(stations, arrivals, sigma, velocity, initial):
    """Joint nonlinear least squares for hypocentre and origin time.

    stations, arrivals, sigma, velocity: as in misfit_and_grad.
    initial: float32 (B,4), starting [x,y,z,t0] in m,m,m,s. Solve the
    unconstrained homogeneous half chi-square objective to convergence with
    Geiger's iterative linearization (Gauss-Newton; damping is permitted).
    The supplied cases have full rank and starts in the local basin.

    Return dict: models float32 (B,4), fitted [x,y,z,t0]; objective float32
    (B,), dimensionless half chi-square; covariance float32 (B,4,4),
    inverse(J.T @ diag(1/sigma**2) @ J) evaluated at the fitted model,
    with J=d(predicted picks)/d[x,y,z,t0]. Units are outer products of
    (m,m,m,s). Use known sigma: do NOT rescale by the residual variance,
    fix origin time, or retain optimizer damping in this covariance.
    ellipse float32 (B,2,2) in m**2 is chi2_2(0.95)*covariance[:,:2,:2].
    Its eigenvectors are horizontal ellipse axes, eigenvalues squared
    semiaxis lengths. A horizontal error d is inside iff d.T@inv(ellipse)@d
    <= 1. This is a marginal 2D 95% ellipse, not a 3D confidence ellipsoid
    or a conditional slice with depth/time held fixed.
    """
    raise NotImplementedError
