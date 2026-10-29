"""All numerical pass thresholds, with reasons."""

# Same discrete operator as the float64 judge: float32 accumulation only.
REFERENCE_RTOL = 1.0e-4
# Each backend must independently meet 1e-4; triangle inequality allows 2e-4.
CROSS_RTOL = 2.0e-4
# The project's exact imaging requirement, in vertical grid cells.
DEPTH_CELLS = 2
# Centered FD on a quadratic objective; float32 subtraction sets the limit.
GRADIENT_RTOL = 1.0e-3
# Binary-exact half m/s step and dyadic directions avoid input rounding;
# a quadratic Born objective has no centered-difference truncation error.
FD_STEP = 0.5
# Independent float64 oracle self-checks, well above double roundoff.
ORACLE_RTOL = 1.0e-8
# A zero input has exactly zero output in this linear system.
ZERO_ATOL = 0.0
# A global amplitude floor prevents accepting a zero image at a known layer.
# This is only an anti-degeneracy gate; full-image parity remains mandatory.
MIN_LAYER_FRACTION = 1.0e-3
