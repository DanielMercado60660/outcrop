"""Locked numeric thresholds and edge exclusion for FFT filter parity."""

# Project-wide limit: maximum absolute error normalized by interior reference peak.
REFERENCE_RELATIVE = 1e-4
# Two cells exclude the original grid boundary, for all transformations.
EDGE_BAND = 2
