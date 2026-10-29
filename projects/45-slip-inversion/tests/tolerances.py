"""Locked numeric criteria; oracle metadata reads this same source."""

# Project-wide float32 parity, measured with max-reference norm to handle zeros.
PARITY_RTOL = 1e-4
# Project-wide centered finite-difference requirement for float32 gradients.
GRADIENT_RTOL = 1e-3
# 1 cm gives a measurable float32 objective change; quadratic central FD is exact
# in real arithmetic. The judge records and verifies this step in float64.
FD_STEP_M = 1e-2
# Required scientific recovery thresholds, unchanged.
MIN_SLIP_CORRELATION = 0.8
MAX_MOMENT_REL_ERROR = 0.10
# Permit only roundoff-sized negative improvement in the minimized objective.
OBJECTIVE_RTOL = 1e-4
# Dimensionless floor for an almost-zero noiseless optimum: one unit of
# whitened squared residual; absolute allowance is still only 1e-4.
OBJECTIVE_FLOOR = 1.0
# Published checklist numbers have four significant figures. Half of the last
# printed decimal unit is the rounding interval, not a float32 parity relaxation.
OKADA_PRINTED_HALF_UNIT = 0.5
# Covers float64 corner cancellation and numerical derivatives in the checklist;
# exact printed symmetry zeros use this absolute allowance alone.
OKADA_CHECK_ATOL = 1e-9
# Dimensionless Table 2 coordinate step; centered differences of displacements
# check the published horizontal derivatives without implementing a second API.
OKADA_CHECK_FD_STEP = 1e-4

REASONS = {
    "PARITY_RTOL": "Project-wide float32 max-norm parity against float64, including zeros.",
    "GRADIENT_RTOL": "Project-wide centered finite-difference relative gradient error.",
    "FD_STEP_M": "0.01 m: quadratic centered FD has no truncation error; avoids float32 cancellation.",
    "MIN_SLIP_CORRELATION": "Project pass: Pearson slip correlation >= 0.8.",
    "MAX_MOMENT_REL_ERROR": "Project pass: scalar seismic moment within 10% of truth.",
    "OBJECTIVE_RTOL": "Float32 roundoff allowance when comparing minimized objectives.",
    "OBJECTIVE_FLOOR": "Unit whitened-residual scale for a nearly zero noiseless optimum.",
    "OKADA_PRINTED_HALF_UNIT": "Half the last printed unit of each four-significant-digit Table 2 value.",
    "OKADA_CHECK_ATOL": "Float64 numerical derivative/corner cancellation allowance; also for exact symmetry zeros.",
    "OKADA_CHECK_FD_STEP": "Dimensionless central derivative step for published surface checklist validation.",
}
