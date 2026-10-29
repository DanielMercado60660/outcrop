"""Float64 Okada (1985) surface rectangular-dislocation solution.

Equations (25)-(30), with Chinnery's corner difference. Local x is along
strike, y is horizontal up-dip, z is up. The rectangle is [0,L] x [0,W]
up-dip from the corner (0,0,-depth): depth is the BOTTOM edge depth.
Only buried rectangles and surface observations are supported. This module
is judge-only; no implementation imports it.
"""
import numpy as np


def displacement(x, y, depth, dip_deg, length, width, slip, nu=0.25):
    """Return (...,3) local surface displacement for (U1,U2,U3) slip.

    Lengths and slips use the same arbitrary length unit. +U1 is left-lateral,
    +U2 reverse for positive dip, +U3 opening. Negative vertical dip uses the
    identical +90 degree rectangle at depth+width, with strike slip reversed
    (the two vertical planes have opposite normals). No other negative dips
    or surface-breaking sources are needed here.
    """
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    slip = np.asarray(slip, np.float64)
    if dip_deg == -90:
        return displacement(x, y, depth + width, 90, length, width,
                            slip * [-1, 1, 1], nu)
    if not 0 < dip_deg <= 90:
        raise ValueError("supported dips: (0,90] or -90 degrees")
    sd, cd = np.sin(np.deg2rad(dip_deg)), np.cos(np.deg2rad(dip_deg))
    vertical = dip_deg == 90  # exact checklist input; snap its trig symmetry
    if vertical:
        sd, cd = 1.0, 0.0
    if depth - width * sd <= 0:
        raise ValueError("rectangle must be strictly buried")
    p, q = y * cd + depth * sd, y * sd - depth * cd
    factor = 1 - 2 * nu

    def corner(xi, eta):
        r = np.sqrt(xi**2 + eta**2 + q**2)
        yt, dt = eta * cd + q * sd, eta * sd - q * cd
        re, rd = r + eta, r + dt
        # Our positive-dip buried rectangles never reach R+eta=0. The -90
        # checklist geometry is handled by exact geometric equivalence above.
        if np.any(re <= 0) or np.any(rd <= 0):
            raise ValueError("unsupported singular corner")
        log_re = np.log(re)
        angle = np.zeros_like(r)
        np.divide(xi * eta, q * r, out=angle, where=q != 0)
        angle = np.arctan(angle)  # principal atan, NOT atan2
        if vertical:
            i1 = -factor / 2 * xi * q / rd**2
            i3 = factor / 2 * (eta / rd + yt * q / rd**2 - log_re)
            i4 = -factor * q / rd
            i5 = -factor * xi * sd / rd
        else:
            xx = np.sqrt(xi**2 + q**2)
            ratio = np.zeros_like(r)
            np.divide(eta * (xx + q * cd) + xx * (r + xx) * sd,
                      xi * (r + xx) * cd, out=ratio, where=xi != 0)
            i5 = factor * 2 / cd * np.arctan(ratio)
            i4 = factor / cd * (np.log(rd) - sd * log_re)
            i3 = factor * (yt / (cd * rd) - log_re) + sd / cd * i4
            i1 = -factor * xi / (cd * rd) - sd / cd * i5
        i2 = -factor * log_re - i3
        ss = np.stack([xi*q/(r*re) + angle + i1*sd,
                       yt*q/(r*re) + q*cd/re + i2*sd,
                       dt*q/(r*re) + q*sd/re + i4*sd], axis=-1)
        ds = np.stack([q/r - i3*sd*cd,
                       yt*q/(r*(r+xi)) + cd*angle - i1*sd*cd,
                       dt*q/(r*(r+xi)) + sd*angle - i5*sd*cd], axis=-1)
        tf = np.stack([q*q/(r*re) - i3*sd**2,
                       -dt*q/(r*(r+xi)) - sd*(xi*q/(r*re)-angle) - i1*sd**2,
                       yt*q/(r*(r+xi)) + cd*(xi*q/(r*re)-angle) - i5*sd**2], axis=-1)
        return (-slip[0]*ss - slip[1]*ds + slip[2]*tf) / (2*np.pi)

    return corner(x, p) - corner(x, p-width) - corner(x-length, p) + corner(x-length, p-width)
