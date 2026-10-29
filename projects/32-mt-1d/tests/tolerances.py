"""Locked numerical thresholds for float32 implementations."""

# Project-wide rule: elementwise relative error at most 1e-4 against float64.
RESPONSE_RTOL = 1e-4
# Two independently rounded float32 paths may each deviate from the reference.
CROSS_RTOL = 1e-4
