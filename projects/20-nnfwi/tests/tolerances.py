"""Locked numeric criteria, shared by tests and the golden generator."""

# Project-wide float32 parity against the *same* discrete float64 problem.
PARITY_RTOL = 1e-4
# Project-wide centered finite-difference gradient requirement for float32 code.
FD_RTOL = 1e-3
# Dimensionless parameter step; large enough to resolve float32 loss differences.
FD_STEP = 1e-2
# The actual project pass criterion: at least 10% lower full-model L2 error.
MAX_ERROR_RATIO = 0.90
# Additional sanity: neural inversion must improve on the starting velocity too.
MAX_INITIAL_ERROR_RATIO = 1.0
# Float64 finite differences must be appreciably more accurate than GPU checks.
ORACLE_FD_RTOL = 1e-4
# Avoid a meaningless relative check of a numerically unresolved derivative.
MIN_DIRECTIONAL_DERIVATIVE = 1e-4
# Contract limit on all committed golden data together (bytes).
MAX_GOLDEN_BYTES = 20_000_000

REASONS = {
    "PARITY_RTOL": "Project-wide float32 limit, same stencil/grid/dt/PML; max norm relative error.",
    "FD_RTOL": "Project-wide float32 centered-difference gradient limit, separately per direction.",
    "FD_STEP": "Dimensionless 0.01 step balances float32 loss cancellation and O(h^2) truncation; calibrated in the oracle.",
    "MAX_ERROR_RATIO": "Unchanged project pass criterion: full physical-grid model error <= 0.90 times plain FWI.",
    "MAX_INITIAL_ERROR_RATIO": "Additional recovery sanity: lower model error than the common initial velocity.",
    "ORACLE_FD_RTOL": "Float64 derivative calibration is ten times tighter than the implementation FD requirement.",
    "MIN_DIRECTIONAL_DERIVATIVE": "At h=0.01, the loss change must resolve float32 arithmetic; no near-zero direction checks.",
    "MAX_GOLDEN_BYTES": "Frozen contract: committed project goldens below 20 MB.",
}
