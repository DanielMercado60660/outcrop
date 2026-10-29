"""Shared implementation contract for project 08."""


def normal_gravity(latitude, height):
    """Return WGS84 normal gravity at exterior geodetic positions.

    Parameters
    ----------
    latitude : numpy.ndarray, float32
        Geodetic latitude in degrees, in [-90, 90]. Any nonempty shape;
        north is positive. Longitude is irrelevant by axial symmetry.
    height : numpy.ndarray, float32
        Geometric height in meters above the WGS84 reference ellipsoid,
        nonnegative, with exactly the same shape as latitude.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``gravity`` is a float32 array with the input shape, in m/s². It is
        the signed normal-gravity value returned by Boule 0.6.0's WGS84
        ``normal_gravity`` in geodetic coordinates (positive toward the
        ellipsoid in ordinary near-Earth conditions). Its off-surface formula
        is the ellipsoidal-harmonic u component, not the full vector norm.
        WGS84 constants are a=6378137 m, 1/f=298.257223563,
        GM=398600441800000 m³/s², and omega=7.292115e-5 rad/s.

    The reference formula is valid only on or outside the ellipsoid. Inputs
    are finite. No behavior is specified for other inputs.
    """
    raise NotImplementedError
