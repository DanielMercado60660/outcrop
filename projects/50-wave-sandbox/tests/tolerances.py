"""Locked numeric criteria; metadata imports this file without copying values."""

# Project-wide float32 parity at identical grid/stencil/dt, normalized by max |ref|.
PARITY_REL = 1.0e-4
# Each red/blue channel is a 1/render_scale Lipschitz map of pressure (AU).
# Propagate the project-wide pressure budget PARITY_REL * max|reference pressure|
# through that gain; clipping and max(q,0) cannot amplify it. Use the physical
# final field for rgba, and all post-update receiver pressures for probes.
# Four float32 epsilons cover division/storage evaluation near unit color.
# Constants green=0/alpha=1, range and disjoint red/blue remain exact checks.
RENDER_ROUND_ABS = 4.0 * 2.0**-23
# Oracle validation requires BOTH independent float32 evaluations to leave
# at least 3x headroom in this propagated color budget, per the adopted rule.
RENDER_AUDIT_MARGIN_MIN = 3.0


def render_abs_tolerance(pressure_peak, render_scale):
    """Absolute RGBA error from pressure parity, in dimensionless color units."""
    return PARITY_REL * pressure_peak / render_scale + RENDER_ROUND_ABS


# Float64 oracle sanity check of source injection against the exact recurrence.
ORACLE_CALIBRATION_REL = 1.0e-14
# Oracle-only audit of the written recurrence vs Deepwave: reordered float64
# stencil operations accumulate roundoff over 160 steps; no GPU limit changes.
ORACLE_SCHEME_REL = 1.0e-10
# Zero references, projected memory interiors, and render green/alpha constants
# remain exact; no absolute floor can hide leakage into a zero field.
EXACT_ABS = 0.0
# Project-wide gradient rule; centered perturbations of 2 m/s avoid float32 loss.
GRAD_FD_REL = 1.0e-3
FD_STEP_MPS = 2.0
# Project 16's gradient direction requirement is also checked for the live path.
GRAD_COS_MIN = 0.999
# Fourth-order space, second-order time with dt proportional to h: total order 2.
CONVERGENCE_ORDER = 2.0
CONVERGENCE_ORDER_SLACK = 0.2
# Exact project 50 requirement, measured independently around the harness call.
MIN_STEPS_PER_SECOND = 60.0
# This is a workload size, not a timing allowance: no startup time is subtracted.
PERFORMANCE_STEPS = 240
PERFORMANCE_GRID_SIZE = 1000
# A conservative line search is frozen by the float64 oracle; descent must occur.
DESCENT_RATIO_MAX = 0.99
