"""All numerical acceptance thresholds for project 25."""

# Project-wide float32 parity limit; max error normalized by max reference time.
REFERENCE_REL_MAX = 1e-4
# Point-source first-order error has logarithmic corrections; oracle rates are 0.67-0.72.
MIN_OBSERVED_ORDER = 0.65
# Finite-grid allowance around first order, including source initialization.
MAX_OBSERVED_ORDER = 1.35
