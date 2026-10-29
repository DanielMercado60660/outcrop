"""Frozen project-facing eikonal solver API."""


def travel_time(phi, speed, spacing):
    """Solve the first-order isotropic eikonal equation on a Cartesian grid.

    Args:
        phi: float32 ndarray, shape (n0, n1) or (n0, n1, n2), meters.
            Its zero level set defines the source. Negative and positive samples
            straddle the source. Axis order is NumPy indexing; grid origins
            are immaterial. At each node with phi == 0, initialize T = 0.
            At every other node adjacent to a sign-changing edge or a zero
            node, calculate d_k = spacing * phi_i / (phi_i - phi_neighbor)
            for each such edge. For each axis k retain the smallest d_k if
            both neighbors cross. Initialize T_i = 1 / (speed_i *
            sqrt(sum_k 1 / d_k**2)), summing over axes with a crossing.
            Speed is sampled at the initialized node, not the crossing.
            All initialized nodes remain fixed throughout the sweeps.
        speed: float32 ndarray of the same shape, strictly positive, m/s,
            sampled at grid nodes. The equation is |grad(T)| = 1/speed.
        spacing: Python float, positive uniform grid spacing in meters, equal
            on all axes. Domain boundaries are nonperiodic.

    Returns:
        Dict containing ``travel``: float32 ndarray of the input shape, in
        seconds, nonnegative on both sides of the zero level set. Solve to the
        converged first-order Godunov upwind solution: on each axis use the
        smaller available neighbor time, and solve the local quadratic using
        only neighbors whose times are below the resulting update. At the
        boundary only in-domain neighbors are available.
    """
    raise NotImplementedError
