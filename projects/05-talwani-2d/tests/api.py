"""Public interface for both implementations of 2-D polygon gravity."""


def polygon_gravity(vertices, observations, density):
    """Return the downward gravity of one infinitely long polygonal body.

    Parameters
    ----------
    vertices : numpy.ndarray, float32, shape (n, 2)
        Simple, non-self-intersecting polygon in metres. Columns are horizontal
        distance x and depth z, positive downward. Vertices may be clockwise or
        counterclockwise, with no duplicated closing vertex; n >= 3.
    observations : numpy.ndarray, float32, shape (m, 2)
        Observation x and positive-down z in metres, in the same coordinates.
        Tested points are strictly outside the polygon and away from edges.
    density : float
        Uniform density contrast in kg/m^3, possibly negative or zero.

    Returns
    -------
    dict
        ``{"gz": array}`` with shape (m,), dtype float32, in mGal. Positive
        gz is downward. The body extends infinitely in the strike direction.
        Use gravitational constant 6.67430e-11 m^3 kg^-1 s^-2.
    """
    raise NotImplementedError
