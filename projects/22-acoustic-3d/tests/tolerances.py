"""Numerical criteria for project 22."""

# Accumulated float32 rounding versus Deepwave float64 at the same stencil and dt.
# For plane waves, normalize by at least the initial amplitude: final fields are
# near a zero crossing although rounding accumulated at unit amplitude.
PARITY_RTOL = 1e-4
# The same initial-amplitude floor applies to plane-wave cross-path comparisons.
# Float32 implementations must agree to the project-wide default relative tolerance.
CROSS_PATH_RTOL = 1e-4
# A second-order scheme should converge within 0.2 of its asymptotic order.
ORDER_MIN = 1.8
ORDER_MAX = 2.2
# Absolute bound for the dimensionless unit-amplitude analytic plane wave, even on the coarsest grid.
ANALYTIC_ERROR_MAX = 0.08
