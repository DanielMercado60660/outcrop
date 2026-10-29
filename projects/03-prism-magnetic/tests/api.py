"""Public forward-model contract shared by the MLX and Metal implementations."""


def prism_magnetic(coordinates, prisms, magnetization, inducing_field):
    """Return prism magnetic induction and its linear total-field anomaly.

    Parameters
    ----------
    coordinates : float32 ndarray, shape (N, 3)
        Observation coordinates in metres, columns easting, northing, upward.
    prisms : float32 ndarray, shape (P, 6)
        Prism boundaries in metres, columns west, east, south, north, bottom,
        top; west < east, south < north, bottom < top. Same Cartesian frame as
        coordinates. Observations in this contract are outside all prisms and
        do not lie on their boundaries.
    magnetization : float32 ndarray, shape (P, 3)
        Uniform magnetization in A/m, columns east, north, upward, per prism.
    inducing_field : float32 ndarray, shape (3,)
        Unit vector in the direction of the ambient field, east, north, upward;
        dimensionless. Upward is positive (an inclination down is negative up).

    Returns
    -------
    dict of float32 ndarrays, each shape (N,)
        Keys ``b_e``, ``b_n``, ``b_u`` are induction components in nT, and
        ``total_field`` is their dot product with inducing_field, in nT. This
        is the standard linear projection, not the exact difference of field
        magnitudes. Contributions from all prisms are summed.
    """
    raise NotImplementedError
