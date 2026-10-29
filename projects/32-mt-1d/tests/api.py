"""Contract for a 1D magnetotelluric forward response."""


def mt1d(resistivity, thickness, frequency):
    """Compute surface apparent resistivity and phase for a layered earth.

    Parameters are NumPy float32 arrays: resistivity, shape (L,), positive
    ohm m, ordered from the surface downward; thickness, shape (L-1,),
    positive metres, in the same order (the last layer is a half-space);
    frequency, shape (F,), positive Hz, in the requested output order.

    Return a dict with float32 arrays apparent_resistivity (F,), in ohm m,
    and phase (F,), in degrees. Use nonmagnetic permeability mu0 =
    1.25663706212e-6 H/m, exp(+i omega t), and the positive impedance
    convention Z = -Z_yx, so a half-space has phase +45 degrees. Apparent
    resistivity is |Z|^2/(mu0 * 2*pi*frequency), phase is arg(Z) in degrees.
    """
    raise NotImplementedError
