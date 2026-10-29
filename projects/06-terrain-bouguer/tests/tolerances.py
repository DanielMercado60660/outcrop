"""Locked numerical tolerances for float32 implementations."""

# Float32 prism corner cancellation grows with distance; the far case uses
# 1e-3 and is reported separately from the ordinary 1e-4 project-wide bound.
NEAR_RTOL = 1e-4
FAR_RTOL = 1e-3
# mGal floor for cancellation near zero and independent summation order.
ABS_MGAL = 2e-5
# Exact slab and algebraic output relations involve only float32 arithmetic.
SLAB_RTOL = 2e-6
ALGEBRA_ATOL = 2e-6
