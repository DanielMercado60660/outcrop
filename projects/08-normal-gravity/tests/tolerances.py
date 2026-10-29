"""Locked numeric thresholds for normal gravity."""

# Project-wide pointwise float32 relative error requirement against Boule float64.
GRAVITY_RELATIVE = 1e-4
# Rounded published WGS84 surface values have about 1e-10 m/s² precision.
SURFACE_SANITY_ABSOLUTE = 1e-9
