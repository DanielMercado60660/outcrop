"""Locked numeric criteria, including the independent scheme check.

Also read by judge/make_goldens.py so metadata uses the same numeric source.
"""

# Project-wide float32 parity, measured as max absolute error / max |reference|.
PARITY_RTOL = 1.0e-4
# Two independent float32 paths may lie on opposite sides of the float64 oracle.
CROSS_RTOL = 2.0e-4
# Project-wide centered finite-difference gradient criterion for float32 arithmetic.
GRADIENT_RTOL = 1.0e-3
# Float64 oracle identities and independent scheme replay must be much tighter
# than implementation criteria. This also covers rounded float32 observations.
ORACLE_RTOL = 1.0e-7
# The objective is quadratic: no centered-difference truncation error; a 2 m/s
# perturbation avoids subtracting nearly equal float32 objectives (v ~1800 m/s).
FD_STEP = 2.0
# Dimensionless multiplier for the float64 nonlinear scalar(v +/- eps*dv)
# derivative check. Perturbations stay below 0.1 m/s in the small fixture.
BORN_LINEARIZATION_STEP = 1.0e-3
# Literal project requirement on residual L2 norm (not its square).
MAX_RESIDUAL_RATIO = 0.5
# Literal project requirement on each vertical wave-packet envelope FWHM.
MAX_WIDTH_RATIO = 0.8
# Project 17 localization rule guards against narrowing a peak at the wrong depth.
MAX_DEPTH_ERROR_CELLS = 2.0
# Roundoff allowance for a mathematically nonincreasing CGLS residual history.
HISTORY_SLACK = 1.0e-6

REASONS = {
    "PARITY_RTOL": "Campaign float32 same-discretization max-norm parity.",
    "CROSS_RTOL": "Triangle bound for two paths each within PARITY_RTOL.",
    "GRADIENT_RTOL": "Campaign float32 centered finite-difference criterion.",
    "ORACLE_RTOL": "Float64 oracle self-check, three orders tighter than parity.",
    "FD_STEP": "2 m/s; quadratic objective has no centered-difference truncation error.",
    "BORN_LINEARIZATION_STEP": "Float64 scalar/Born central difference; velocity change below 0.1 m/s.",
    "MAX_RESIDUAL_RATIO": "Campaign: residual L2 norm reduced by at least 50%.",
    "MAX_WIDTH_RATIO": "Campaign: each reflector vertical analytic-envelope FWHM reduced by at least 20%; review route (a) resolves the raw-lobe sampling floor.",
    "MAX_DEPTH_ERROR_CELLS": "Project 17 reflector localization guard.",
    "HISTORY_SLACK": "Only permits float32 reduction roundoff in residual monotonicity.",
}
