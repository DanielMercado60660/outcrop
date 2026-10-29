"""Contract for homogeneous-medium double-difference relocation."""


def relocate(stations, initial_locations, initial_origins, pairs, observed, velocity, damping):
    """Minimize the stated damped double-difference least-squares objective.

    stations: float32 (S, 3), station x, y, depth in km, local Cartesian.
    initial_locations: float32 (N, 3), event x, y, depth in km, depth positive.
    initial_origins: float32 (N,), event origin times in seconds, common epoch.
    pairs: int32 (P, 2), zero-based event indices (i, j), i != j.
    observed: float32 (P, S), observed arrival time i minus j in seconds.
    velocity: Python float, homogeneous isotropic propagation speed in km/s.
    damping: positive Python float, dimensionless regularization weight.

    For event e and station s, t[e,s] = origin[e] + distance[e,s]/velocity.
    Minimize sum over p,s of (t[i,s]-t[j,s]-observed[p,s])**2 plus
    damping * sum over events of (||location-initial_location||**2 /
    velocity**2 + (origin-initial_origin)**2). The damping also fixes the
    unobservable common origin-time shift. Return a dict containing float32
    locations (N, 3), km, and origins (N,), seconds, in input event order.
    """
    raise NotImplementedError
