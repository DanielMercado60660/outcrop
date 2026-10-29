"""Locked numerical criteria; reasons apply to oracle self-checks as well."""

# Project-wide float32 ceiling against the identical float64 discrete problem.
PARITY_RTOL = 1.0e-4
# Volts: only protects cancellation at nearly zero voltage; 0.1 nV is below noise.
VOLTAGE_ATOL = 1.0e-10
# Dimensionless objective absolute floor for a zero-valued objective.
OBJECTIVE_ATOL = 1.0e-7
# Objective per dex: absolute floor for nearly zero gradient components.
GRADIENT_ATOL = 1.0e-6
# Project-wide centered finite-difference relative-error ceiling for float32 gradients.
FD_RTOL = 1.0e-3
# Dex: 0.01 gives ~6--26 objective-unit differences at objectives ~1700.
# eps32*objective/(h*|slope|) is ~7e-5 or less, below FD_RTOL=1e-3;
# a full float32 assembly/solve/autodiff replay checks rounding AND truncation.
FD_STEP = 1.0e-2
# Float64 adjoint vs centered difference: independent oracle sanity check.
ORACLE_FD_RTOL = 1.0e-6
# Dex: float64 FD step gives negligible roundoff and second-order truncation.
ORACLE_FD_STEP = 1.0e-5
# Target is reduced chi squared <= 1.1 (RMS <= sqrt(1.1)); finite noisy survey
# has stochastic chi-square fluctuations. Proposed numeric meaning of "target".
TARGET_CHI2 = 1.1
# Project spec: Euclidean contrast-weighted centroid distance in core cell units.
LOCATION_CELLS = 2.0
# Project spec, applied to EVERY true anomaly cell rather than just its average.
ANOMALY_LOG10_MAX = 0.3
# Broad background changes must not masquerade as a localized recovered anomaly.
# 0.15 dex is half the allowed anomaly error and below half the true contrast.
BACKGROUND_LOG10_RMS = 0.15
# Float32 model bounds and inactive-cell preservation allow only rounding.
MODEL_ATOL = 1.0e-6
# Recovery support uses half the prescribed contrast, avoiding tiny noise tails.
SUPPORT_CONTRAST_FRACTION = 0.5
# Runtime centered differences can use this denominator floor for near-zero slopes.
FD_SLOPE_FLOOR = 1.0e-6
# Float64 solve linearity/reciprocity sanity checks; no approximation is involved.
ORACLE_LINEAR_RTOL = 1.0e-10
# Tight float64 optimizer termination makes oracle recovery reproducible; these
# are reference-solver settings, not required implementation iteration parity.
ORACLE_OPT_FTOL = 1.0e-9
# Tight float64 parameter-step termination; reference solver setting only.
ORACLE_OPT_XTOL = 1.0e-9
# Tight float64 projected-gradient termination; reference solver setting only.
ORACLE_OPT_GTOL = 1.0e-9
# Oracle-only half-space sanity bounds, ohm m. Coarse near-electrode cells can
# bias apparent resistivity; tight identical-mesh parity remains 1e-4.
SANITY_HALFSPACE_RHO_MIN = 50.0
# Oracle-only upper sanity bound in ohm m on the same coarse inversion mesh.
SANITY_HALFSPACE_RHO_MAX = 150.0
# Oracle-only guard: a homogeneous starting model must decisively miss target.
MIN_INITIAL_CHI2_FACTOR = 10.0
