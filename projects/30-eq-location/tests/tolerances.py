"""Locked numerical thresholds; shared by tests and golden metadata."""

# Project-wide float32 parity ceiling; normwise comparisons avoid zero-entry ratios.
PARITY_RTOL = 1.0e-4
# A grid candidate must be the same node (millimetre allowance for coordinates).
GRID_POSITION_ATOL_M = 1.0e-3
# Origin time profiling uses sums of seconds in float32.
TIME_ATOL_S = 2.0e-6
# Near-zero dimensionless least-squares losses cannot use relative error alone.
LOSS_ATOL = 1.0e-4
# Required noise-free physical recovery limits (project specification).
GRID_MAX_CELLS = 1.0
GEIGER_MAX_POSITION_M = 10.0
GEIGER_MAX_TIME_S = 0.01
# Inclusive specified limits: 180 through 198 of the fixed 200 trials.
COVERAGE_MIN = 0.90
COVERAGE_MAX = 0.99
NOISY_TRIALS = 200
# Horizontal marginal Gaussian ellipse: chi-square(2) inverse CDF at 0.95.
ELLIPSE_CHI2_95 = 5.991464547107979
# Project-wide gradient limit, checked separately for each parameter's units.
GRADIENT_RTOL = 1.0e-3
# Central differences: 8 m spatial, 2 ms temporal. These resolve float32
# objective differences while keeping nonlinear truncation well below 1e-3.
FD_STEPS = (8.0, 8.0, 8.0, 0.002)
# Float64 oracle self-check; far tighter than any implementation requirement.
ORACLE_GRADIENT_RTOL = 1.0e-6
ORACLE_FD_STEPS = (0.1, 0.1, 0.1, 1.0e-5)
# Check the recorded analytic chi-square constant against SciPy in float64.
ORACLE_QUANTILE_ATOL = 1.0e-14
