"""All numeric acceptance thresholds for the project."""

# Project-wide float32 requirement: elementwise relative error for nonzero reference values.
RELATIVE_ERROR = 1e-4
# Exactly zero components need an absolute scale; allow float32 cancellation.
ZERO_SCALE_ERROR = 1e-6
# Two independently rounded outputs can each deviate by the reference limit.
CROSS_RELATIVE_ERROR = 2e-4
# Zero-component cross-path differences can contain both paths' rounding error.
CROSS_ZERO_SCALE_ERROR = 2e-6
# Project-wide requirement for centered finite-difference gradient comparisons.
GRADIENT_RELATIVE_ERROR = 1e-3
