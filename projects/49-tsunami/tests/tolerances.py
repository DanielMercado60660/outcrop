"""All acceptance thresholds; errors use a maximum reference scale."""

# Float32 arithmetic compared with the same-grid float64 reference.
PARITY = 1e-4
# Same numerical scheme, independent float32 backends and operation ordering.
CROSS_PATH = 1e-4
# Project-wide gradient limit against centered float64 finite differences.
GRADIENT = 1e-3
# Scalar objective may be cancellation-sensitive, so normalize by max(1, |reference|).
OBJECTIVE = 1e-4
# Conservative sums can accumulate float32 rounding over time.
MASS = 1e-5
# First-order Euler/Rusanov smooth-wave convergence, allowing pre-asymptotic error.
LINEAR_ORDER_MIN = 0.85
# A shock is smeared over several cells; global L1 rate may be below one.
DAM_ORDER_MIN = 0.45
# Required criterion for measured linear long-wave speed.
SPEED_RELATIVE = 0.01
# Prevent division by zero for an identically zero reference component.
SCALE_FLOOR = 1e-30
# Float32 perturbation-state simulation at 1000 and 4000 m, including final
# rounding to total depth, stays below the project-wide same-grid limit.
DEEP_PARITY = 1e-4
# Two independently rounded float32 depth outputs can differ by one ULP at
# 4000 m even when both pass the reference test; 2e-4 follows by triangle bound.
DEEP_CROSS_PATH = 2e-4
