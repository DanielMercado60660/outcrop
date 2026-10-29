"""Numerical thresholds, normalized by the reference peak amplitude."""

# Project-wide float32 parity limit, including accumulated time-step rounding.
PARITY = 1e-4
# Independent float32 implementations can differ from accumulated rounding.
CROSS_PATH = 1e-4
# Project requirement: measured order within 0.2 of second order.
ORDER_DEVIATION = 0.2
# Project requirement: reflected amplitude below 1% of incident peak.
REFLECTION = 0.01
