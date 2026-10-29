"""Locked numeric criteria, shared with the golden metadata generator."""

# Project-wide float32 relative accuracy for analytic forward and path parity.
FORWARD_RTOL = 1e-4
# Exact surface boundary is zero degrees C; this is a negligible float32 allowance.
TEMPERATURE_ATOL_C = 1e-5
# Allows near-zero deep heat flow without relaxing meaningful flux values (W/m²).
FLUX_ATOL_W_M2 = 1e-9
# Required age-recovery threshold, relative to true age.
AGE_RECOVERY_RTOL = 0.05
# Noise-free analytic inversion has only float32 rounding error, so use the
# project-wide 1e-4 analytic accuracy limit in addition to the 5% pass criterion.
AGE_EXACT_RTOL = 1e-4
# Project-wide gradient threshold; centered float64 differences use a 0.01 Myr step.
GRADIENT_RTOL = 1e-3
# Small absolute allowance for the dimensionless objective.
OBJECTIVE_ATOL = 1e-7
# All tested gradients are nonzero: enforce the project-wide relative limit strictly.
GRADIENT_ATOL_PER_MYR = 0.0
# Float32 path-to-path allowance for age output, in Myr.
AGE_ATOL_MYR = 1e-5
