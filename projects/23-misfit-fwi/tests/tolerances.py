"""All numerical acceptance bounds and fixed experiment controls."""

# Project-wide float32 parity limit, with global infinity-norm scaling.
PARITY_RTOL = 1e-4
# Only protects exact-zero comparisons; does not mask small gradients.
ZERO_ATOL = 1e-12
# Project-wide centered finite-difference gradient requirement.
GRAD_RTOL = 1e-3
# L2 is quadratic: a larger step reduces float32 cancellation without truncation.
FD_L2_STEP = 1e-2
# Unit-amplitude envelope traces: reduce subtraction noise while retaining O(h²)
# truncation well below the gradient bound (checked independently in float64).
FD_ENVELOPE_STEP = 5e-3
# Below epsilon, envelope subtraction cancels; a larger step resolves the change.
# The generator audits both literal float32 arithmetic and float64 truncation.
FD_SMALL_ENVELOPE_STEP = 1e-2
# Require feasibility audits to have at least 3x margin below the project-wide FD bound.
FD_AUDIT_RTOL = GRAD_RTOL / 3
# Float64 oracle derivative verification, independent of implementation precision.
ORACLE_GRAD_RTOL = 2e-6
ORACLE_FD_STEP = 1e-5
# Arrival recovered to 1/4 of the 2 ms trace sample interval.
RECOVERY_SECONDS = 5e-4
# Spec's mandatory envelope/L2 contiguous attraction-basin ratio.
MIN_BASIN_RATIO = 1.5
# Prevent a degenerate experiment with an L2 baseline that never converges.
MIN_L2_BASIN_SECONDS = 0.01
# Fixed local optimizer: steps 1/50 of a 10 Hz carrier period cannot hop cycles.
SEARCH_STEP_SECONDS = 0.002
# Stationary-point localization is far tighter than the recovery criterion.
SEARCH_STOP_SECONDS = 1e-6
SEARCH_MAX_STEPS = 100
SEARCH_BISECTIONS = 18
# Strictly positive smoothing in trace amplitude units; avoids sqrt singularities.
ENVELOPE_EPSILON = 1e-3
# Waveforms use a 10 Hz carrier and Gaussian width 0.05 s, dt = 0.002 s.
TRACE_DT = 0.002
