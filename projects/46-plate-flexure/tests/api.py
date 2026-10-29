"""Frozen interface for the plate-flexure implementations."""


def plate_flexure(x, rigidity, foundation, load, plate):
    """Compute line-load deflection of a plate on a linear foundation.

    x : float32 NumPy array, shape (n,), m. Coordinates relative to the load
        at x=0; nonnegative for a broken plate, signed for a continuous plate.
    rigidity : Python float, N m. Positive constant flexural rigidity D.
    foundation : Python float, N/m^3. Positive foundation stiffness k.
    load : Python float, N/m. Vertical line force, positive downward.
    plate : str. "continuous" for an infinite plate, "broken" for a
        semi-infinite plate with an edge force and zero edge bending moment.

    Returns {"deflection": float32 NumPy array of shape (n,), in m},
    positive downward. The solution decays at infinity and obeys
    D*w'''' + k*w = 0 away from the load. The continuous plate has a
    shear jump D*(w'''(0+) - w'''(0-)) = P. For the broken plate,
    x increases into the plate and w''(0)=0, D*w'''(0)=P. The full
    line force P acts on this one edge; it is not split between halves.
    """
    raise NotImplementedError
