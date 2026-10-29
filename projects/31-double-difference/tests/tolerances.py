"""Locked thresholds; all numerical pass criteria live here."""

# Project-wide float32 relative error bound, scaled by the largest reference value.
SOLUTION_REL = 1e-4
# DD values subtract arrival times of several seconds in float32; 20 microseconds
# accommodates cancellation and optimizer stopping, while resolving 2 ms noise.
RESIDUAL_ABS_SECONDS = 2e-5
# The project's specified minimum reduction in standard deviation.
MAX_SPREAD_RATIO = 0.5
