"""Locked numerical criteria. All error thresholds and FD steps live here."""

# Project-wide float32 parity limit; geometry gets a submillimetre zero floor for
# cancellation at almost coincident corner crossings on a 1.2 km grid.
PARITY_RTOL = 1.0e-4
GEOMETRY_ATOL_M = 2.5e-4
# Spec's 1e-5 small-system requirement, interpreted as relative max error
# (stricter than an absolute 1e-5 s/m bound for these ~5e-4 s/m models).
SMALL_SYSTEM_RTOL = 1.0e-5
# The specified checkerboard resolution threshold; no reduction is proposed.
CHECKERBOARD_MIN_CORRELATION = 0.8
# Project-wide gradient requirement. Quadratic objectives have no truncation error;
# a 1e-5 s/m centered step avoids float32 subtractive cancellation.
GRADIENT_FD_RTOL = 1.0e-3
FD_STEP_S_PER_M = 1.0e-5
# Judge-only consistency checks: float64 arithmetic and independently derived
# hand solutions / normal equations should agree to near machine precision.
ORACLE_RTOL = 1.0e-11
ORACLE_ATOL = 1.0e-12

TOLERANCE_METADATA = {
    "PARITY_RTOL": {"value": PARITY_RTOL, "reason": "Project-wide float32 relative parity, max absolute error / max absolute reference."},
    "GEOMETRY_ATOL_M": {"value": GEOMETRY_ATOL_M, "reason": "0.25 mm near-zero floor for float32 intersection cancellation at nearly coincident corners on a 1.2 km grid. Also require global relative max error <=1e-4."},
    "SMALL_SYSTEM_RTOL": {"value": SMALL_SYSTEM_RTOL, "reason": "Spec 1e-5, interpreted as relative max error in recovered slowness."},
    "CHECKERBOARD_MIN_CORRELATION": {"value": CHECKERBOARD_MIN_CORRELATION, "reason": "Spec Pearson correlation >= 0.8, on a geometry-only frozen mask."},
    "GRADIENT_FD_RTOL": {"value": GRADIENT_FD_RTOL, "reason": "Project-wide centered finite-difference gradient relative error <= 1e-3."},
    "FD_STEP_S_PER_M": {"value": FD_STEP_S_PER_M, "reason": "Quadratic objective: no centered truncation error; 2% of background slowness avoids float32 cancellation. Actual float32 endpoint separation is used."},
    "ORACLE_RTOL": {"value": ORACLE_RTOL, "reason": "Judge-only float64 hand-solution and normal-equation sanity checks."},
    "ORACLE_ATOL": {"value": ORACLE_ATOL, "reason": "Judge-only float64 zero floor: reversing kilometre rays changes nearly zero segments by ~1.5e-13 m. Never a relaxed implementation tolerance."},
}
