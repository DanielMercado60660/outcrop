"""Public interface for the 2D and 3D fast-marching solvers."""


def travel_time(phi, speed, spacing, order):
    """Solve |grad T| = 1 / speed away from the zero contour of phi.

    Args:
        phi: NumPy float32 array, shape (n0, n1) or (n0, n1, n2),
            signed level-set samples at grid nodes, in arbitrary level-set units. Its
            piecewise linearly interpolated zero contour is the source.
            Both signs are present. Travel time is nonnegative on both sides.
        speed: NumPy float32 array with the same shape as phi, strictly
            positive propagation speed in length/time units at each node.
        spacing: list of positive Python floats, one grid spacing in length
            units per array axis, in NumPy axis order. Grid origin is arbitrary.
        order: Python int, 1 or 2, the upwind spatial discretization order.

    Initialization and marching rules:
        For order 1, crossings are STRICT sign changes: phi_i*phi_neighbor < 0.
        An exactly zero-valued neighbor is NOT a crossing. For order 2 only,
        edges from nonzero phi nodes to zero-valued neighbors also count as
        crossings (the established second-order variant).
        Along each axis retain the nearer crossing,
        with distance d = h * abs(phi_i / (phi_i - phi_neighbor)). Combine
        crossing axes as distance = 1 / sqrt(sum(1 / d**2)); divide by the
        speed at that node. Independently, initialize every phi == 0 node at
        time zero and freeze it. Freeze all crossing-initialized nodes too.
        At order 1, neighbors of exact zeros that have no strict crossing are
        initially unfrozen and get their times by marching from the frozen band.
        Boundary edges outside the grid are absent.
        All other nodes start at infinity. Solve using the speed at the node
        being updated, never an interpolated or averaged speed.

        Use a minimum-time priority queue. Initially update all unfrozen
        neighbors of the frozen band; subsequently accept the lowest tentative
        time and update its immediate unfrozen neighbors. Ties are resolved by
        lowest C-order flat index. Discard stale queue entries and only lower
        tentative times. On each axis choose the frozen immediate neighbor with
        lowest time; ties choose the negative direction. Axes without one are
        omitted. For order 1 that axis contributes (a, w) = (T1, 1/h**2).
        Include EVERY available axis in the quadratic below at order 1; do
        not drop axes even when the larger root is below an included a.
        The candidate time is the larger root of
        sum(w*(T-a)**2) = 1/speed_i**2 over the included terms.

        For order 2, use the second neighbor in that same direction only when
        it is in bounds, frozen, has T2 <= T1, and phi_i * phi_second >= 0.
        Then (a, w) = ((4*T1-T2)/3, 9/(4*h**2)); otherwise use order 1 on
        that axis. For order 2 only, sort available terms by a, adding them
        in order while the current root is at least the next a. Use the same
        larger-root equation. If its discriminant is negative, redo the
        entire update with order-1 terms,
        retaining this order-2 dimension-dropping rule in that fallback.
        A shifted quadratic centered on the smallest a avoids float32
        cancellation. Only immediate neighbors are revisited on acceptance.

    Returns:
        dict with key "time": NumPy float32 array of phi.shape, travel time
        in time units from the interpolated zero contour. Zero-valued phi
        nodes have zero time. The grids are nonperiodic and unmasked.
    """
    raise NotImplementedError
