"""Locked numeric criteria, shared by the judge and tests."""

# Project-wide float32 parity against the float64 analytic layered-earth recursion.
FORWARD_RTOL = 1e-4
# Absolute floors only protect cancellation near zero in log10(rhoa).
FORWARD_ATOL = 1e-6
# Weighted objectives/gradients amplify float32 response rounding slightly.
OBJECTIVE_RTOL = 1e-4
OBJECTIVE_ATOL = 1e-6
GRADIENT_RTOL = 1e-3  # Project-wide gradient criterion, measured in relative L2 norm.
GRADIENT_ATOL = 1e-5  # At exact zero the relative error is undefined.
FD_STEP = 2e-3  # log10(ohm m); centered O(h^2), large enough for float32 subtraction.
# Judge-only derivative uses float64 and a much smaller centered step.
JUDGE_FD_STEP = 1e-5
JUDGE_FD_RTOL = 1e-5  # Step-halving check of the float64 objective derivative.
TARGET_RMS = 1.0  # sqrt(mean(((prediction-observation)/sigma)**2)).
MISFIT_SLACK = 1e-4  # Float32 allowance at the active RMS=1 constraint.
MODEL_MAX_ERROR = 0.2  # Spec: maximum absolute log10 resistivity error, not mean error.
# Constrained nonlinear solvers need not return identical models. Enforce that
# their roughness is within 5% of an independent float64 constrained minimum.
ROUGHNESS_RTOL = 0.05
ROUGHNESS_ATOL = 1e-5  # Handles the constant halfspace's zero minimum roughness.
MODEL_LOWER = 0.0  # Fixed inversion bounds: 1 ohm m.
MODEL_UPPER = 4.0  # Fixed inversion bounds: 10,000 ohm m.
MAX_ITERATIONS = 150  # Small 8-parameter problem; stopping budget, not parity accuracy.
# Oracle sanity checks: analytic limits and monotonic physical magnitudes.
ORACLE_RTOL = 1e-10
# Float64 optimizer stopping tolerances; substantially tighter than float32 gates.
ORACLE_LEAST_SQUARES_TOL = 1e-12
ORACLE_CONSTRAINED_TOL = 1e-11
