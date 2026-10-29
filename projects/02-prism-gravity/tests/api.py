"""Public contract for both implementations."""


def prism_gravity(coordinates, prisms, density, field):
    """Compute the summed field of axis-aligned homogeneous rectangular prisms.

    coordinates: float32 array (N, 3), observation positions in metres, with
        columns easting, northing, upward (positive upward); N is positive.
    prisms: float32 array (P, 6), west, east, south, north, bottom, top in
        metres in the same Cartesian frame. Bounds are strictly ordered; P > 0.
    density: float32 array (P,), density contrasts in kg/m^3; may be negative
        or zero. Contributions sum linearly.
    field: string, one of 'potential', 'g_e', 'g_n', 'g_z'. Potential is
        positive for positive mass, in J/kg. Acceleration is in mGal (1 mGal =
        1e-5 m/s^2); g_e and g_n are eastward and northward, while g_z is
        downward, opposite to the upward coordinate. These fields are defined
        at interior points and on faces, edges and vertices.

    Returns exactly {field: values}, values a float32 NumPy array of shape
    (N,). Evaluate in source-local coordinates to limit cancellation.
    """
    raise NotImplementedError
