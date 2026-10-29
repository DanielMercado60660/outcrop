"""Numerical thresholds for the locked plate-flexure tests."""

# Project-wide 1e-4 float32 accuracy, normalized by peak reference magnitude:
# pointwise relative error is singular at the flexural nodes.
RELATIVE_MAX_NORM = 1e-4
# Relative error is undefined at zero load; 0.1 micrometre is negligible
# relative to the nonzero cases and catches spurious output.
ZERO_ABSOLUTE_M = 1e-7
# Cross-path agreement uses the same project-wide float32 accuracy target.
CROSS_PATH_RELATIVE_MAX_NORM = 1e-4
