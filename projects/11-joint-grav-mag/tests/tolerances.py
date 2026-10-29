"""Locked numeric criteria; shared by tests and golden metadata."""

# Project-wide float32 parity: global max absolute error / max absolute reference.
PARITY_RTOL = 1e-4
# Solver model parity only: cond(Hg)=1.18e4, cond(Hm)=7.80e3 amplify
# float32 roundoff. 3e-3 covers about 2*eps32*max(cond), including the
# reviewed MLX CPU solve's 1.6e-3 error. All five cases are audited in judge/.
# Predicted data from EACH returned model still must meet PARITY_RTOL.
# The 10% recovery threshold is unchanged.
SOLVER_MODEL_RTOL = 3e-3
# Only used for analytically zero objective/gradient components.
ZERO_ATOL = 1e-6
# Project-wide centered finite-difference gradient criterion for float32 outputs.
GRAD_RTOL = 1e-3
# Normalized model perturbation: balances float32 cancellation and O(h^2) error.
FD_STEP = 2e-3
# Float64 oracle derivative self-check (centered differences with step below).
ORACLE_GRAD_RTOL = 1e-6
ORACLE_FD_STEP = 1e-5
# Roundoff in the float64 affine-related cross-gradient identity.
ORACLE_ZERO_ATOL = 1e-14
# The project's required improvement for EACH property, without rounding.
MAX_ERROR_RATIO = 0.90
