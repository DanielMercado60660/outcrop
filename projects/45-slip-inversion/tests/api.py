"""MLX-only inversion API; Green matrices are supplied, never generated.

All array inputs/outputs are NumPy float32, finite, C-order. No reference
library is needed by implementations. Coordinates used to construct the supplied
matrix are east/north/up (ENU), metres, surface z=0. A column describes unit
signed slip in a fixed rake direction; rows can mix LOS and GPS projections.
Positive LOS displacement points toward the satellite. No offset is estimated.
The inverse is unconstrained: signed slips are allowed, no positivity clipping.
"""


def invert_slip(green, data, sigma, regularizer, damping, areas, shear_modulus):
    """Minimize J(s) = .5*sum(((G@s-d)/sigma)**2)
    + .5*damping**2*sum((L@s)**2), using the same objective as misfit_and_grad.

    Args:
        green: (nobs,npatch) float32 G, metres displacement per metre slip
            (dimensionless), rows in exactly the supplied observation order.
        data: (nobs,) float32 d, signed displacement in metres.
        sigma: (nobs,) float32, strictly positive displacement standard deviation
            in metres; diagonal independent noise covariance sigma**2.
        regularizer: (nreg,npatch) float32 L, units 1/metre. Supplied explicitly;
            do not infer a grid, boundary condition or normalization.
        damping: Python float >=0, dimensionless multiplier of L (squared in J).
        areas: (npatch,) float32 positive finite patch areas in square metres
            (along-strike length times down-dip width, not projected area).
        shear_modulus: Python float >0, uniform shear modulus in pascals.

    The augmented weighted system [G/sigma; damping*L] has full column
    rank in every fixture, so the minimizer is unique even at zero damping.
    Return dict with float32 arrays: slip (npatch,) in metres, predicted
    (nobs,) = G@slip in metres, moment scalar shape () in newton metres,
    M0 = shear_modulus * sum(areas * abs(slip)). Moment is nonnegative even
    for negative rake amplitudes. Implementations receive no true slip.
    """
    raise NotImplementedError


def misfit_and_grad(green, data, sigma, regularizer, damping, slip):
    """Return the weighted quadratic objective and its slip gradient.

    green (nobs,npatch), data (nobs,), sigma (nobs,), regularizer
    (nreg,npatch), damping: same dtypes, units, conventions as invert_slip.
    slip: (npatch,) float32 signed trial slip in metres, unrestricted.

    Return dict of float32 arrays: objective scalar shape (), dimensionless,
    J = .5*|| (green@slip-data)/sigma ||_2**2
        + .5*damping**2*||regularizer@slip||_2**2;
    gradient (npatch,), units 1/metre, derivative of this exact J with
    respect to slip, including the regularization term. There is no division
    by nobs, no hidden damping, no noise re-estimation. MLX uses autodiff.
    """
    raise NotImplementedError
