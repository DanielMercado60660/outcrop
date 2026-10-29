"""Public forward API shared by both implementations."""


def okada(coordinates, source, depth, dip_deg, bounds, slip, alpha):
    """Compute displacement and horizontal spatial derivatives in a half-space.

    coordinates: float32 (N, 3), observation (x, y, z), in length units;
        z <= 0, with z positive up and x along strike.
    source: "point" or "finite_rectangle". Point observations are at z = 0;
        rectangles support z <= 0.
    depth: Python float, positive depth in the same length units as coordinates,
        of the source reference point (0, 0, -depth).
    dip_deg: Python float, signed dip in degrees, including +/-90 degrees.
    bounds: float32 (4,), [AL1, AL2, AW1, AW2] in length units. A rectangle
        spans these along-strike and up-dip coordinates (xi, eta), at position
        (xi, eta*cos(dip), -depth + eta*sin(dip)). Ignored for a point source.
    slip: float32 (3,), strike, dip, tensile components, in displacement units
        for a rectangle, or potency (displacement times area) for a point.
        These are hanging-wall motion relative to footwall: for 0 < dip < 90,
        positive strike is left-lateral, dip is reverse, tensile is opening.
    alpha: Python float, dimensionless (lambda + mu)/(lambda + 2*mu).
        All published cases use lambda=mu, hence alpha=2/3.

    Return a dict with float32 arrays: displacement (N, 3), in displacement
    units, and horizontal_derivatives (N, 3, 2), displacement per length unit.
    Axes are observation, displacement component (x,y,z), derivative direction
    (x,y); derivatives are raw partials with no strain sign flips. Outputs include
    the supplied slip/potency amplitude; point potencies already include area.
    All lengths must use one consistent unit (e.g. metres), with rectangle slip
    in metres and point potency in cubic metres for metre-valued displacement.
    Physical source singularities are excluded;
    removable singularities, including vertical dip cases, must be handled.
    """
    raise NotImplementedError
