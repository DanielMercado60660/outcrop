"""Frozen, implementation-neutral interface for project 01."""


def gravity(coordinates, centers, masses, radii):
    """Compute Newtonian gravitational potential and acceleration.

    Args:
        coordinates: float32 array of shape ``(..., 3)``, observation positions
            in metres. The leading dimensions describe the observation grid.
        centers: float32 array of shape ``(S, 3)``, source positions in metres.
        masses: float32 array of shape ``(S,)``, source masses in kilograms.
        radii: float32 array of shape ``(S,)``, source radii in metres; zero
            denotes a point mass, positive denotes a homogeneous solid sphere.

    All Cartesian vectors use (x, y, z), in a common right-handed frame with
    z increasing upward. Masses are positive, radii are nonnegative, and no
    observation coincides with a point mass. Use G = 6.67430e-11 SI. The
    potential is zero at infinity; acceleration is minus its spatial gradient.
    Contributions from all sources are summed, including inside spheres.

    Returns:
        Dict with float32 NumPy arrays: ``potential`` of shape ``(...)`` in
        m^2/s^2, and ``acceleration`` of shape ``(..., 3)`` in m/s^2.
    """
    raise NotImplementedError
