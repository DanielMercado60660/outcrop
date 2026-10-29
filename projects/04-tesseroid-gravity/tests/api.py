"""Frozen project interface; implementations live in mlx/impl.py and metal/."""


def tesseroid_gravity(coordinates, tesseroids, density, field, radial_adaptive):
    """Calculate a constant-density tesseroid gravity field.

    Parameters
    ----------
    coordinates : float32 ndarray, shape (N, 3)
        Geocentric spherical longitude and latitude in degrees, radius in m.
    tesseroids : float32 ndarray, shape (M, 6)
        West, east, south, north in degrees and bottom, top radii in m.
        West < east, south < north, and 0 < bottom < top. Observations are
        outside all source interiors. This API covers non-wrapping sources.
    density : float32 ndarray, shape (M,)
        Constant density contrast in kg/m^3, signed or zero.
    field : str
        "potential" for positive gravitational potential in J/kg, or "g_z"
        for the downward radial acceleration in mGal (1 mGal = 1e-5 m/s^2).
    radial_adaptive : bool
        If true split all three dimensions adaptively; otherwise split only
        longitude and latitude. Use the discretization specified below.

    Discretization
    --------------
    Independently for each observation and nonzero-density source, recursively
    split the current cell. Its center is the arithmetic mean of each pair of
    bounds. Distance d is the straight-line Euclidean distance between the
    observation and this center, both expressed in geocentric spherical
    coordinates. Let lambda and phi be radians, phi_c the center latitude,
    and R the cell's top radius. The three sizes are
    R * 2 asin(cos(phi_c) sin((east-west)/2)), R * (north-south), and
    top-bottom. For each enabled dimension, split it into equal halves if and
    only if d / size < q, where q is 1 for potential and 2.5 for g_z. Split
    all qualifying dimensions simultaneously and recurse until none qualify.

    For each final cell use the tensor product of two-point Gauss-Legendre
    nodes +/- 1/sqrt(3), each with weight 1, mapped to each cell interval.
    A node at longitude lambda', latitude phi', radius r' has mass
    rho * ((east-west)*(north-south)*(top-bottom)/8) * r'**2 * cos(phi'),
    with angular extents in radians. For a point at radius r, define
    cos(psi) = sin(phi) sin(phi') + cos(phi) cos(phi') cos(lambda-lambda')
    and distance L = sqrt(r**2 + r'**2 - 2*r*r'*cos(psi)). Sum G*m/L
    for potential and G*m*(r-r'*cos(psi))/L**3 * 1e5 for g_z, where
    G = 6.6743e-11 m^3 kg^-1 s^-2. Evaluate equivalent formulas stably in
    source-local coordinates: direct subtraction of Earth-scale squared
    radii and arccos of near-one values are inaccurate in float32. For example,
    use the haversine for 1-cos(psi), and form radial differences locally.

    Returns
    -------
    dict
        {"field": float32 ndarray of shape (N,)} in the requested units.
        Contributions from all sources are summed at each observation.
    """
    raise NotImplementedError
