"""Locked numeric criteria; float32 implementations, float64 judge.

Changing these values after lock requires the maintainer's approval.
"""

# The project's explicit pass criterion; applied separately to every source.
MEDIAN_TIME_REL = 0.01
# Proposed guard against sacrificing almost half the receivers to the median.
# A PINN is approximate, so a 5% tail allowance is distinct from rounding parity.
P95_TIME_REL = 0.05
# Project-wide float32 parity for fixed network evaluations and objective derivatives.
NETWORK_REL = 1e-4
# A loss is a squared residual and cancels near the exact homogeneous solution.
# Compare its square root with this float32 residual floor (dimensionless).
OBJECTIVE_SQRT_ATOL = 1e-6
# Project-wide centered finite-difference gradient criterion (vector relative L2).
GRADIENT_FD_REL = 1e-3
# Seconds: the factored ansatz imposes T(source, source)=0 algebraically.
SOURCE_TIME_ATOL = 1e-7
# Dimensionless parameter step: balances float32 subtraction and O(h^2) error.
PARAMETER_FD_STEP = 0.002
# Metres: 0.05% of the 2-km domain, resolving float32 travel-time differences.
COORDINATE_FD_STEP = 1.0
# Reference-only sanity bounds: fine-grid point-source FMM has discretization
# error near its singularity; these are NOT implementation parity tolerances.
FMM_HOMOGENEOUS_MEDIAN_REL = 0.005
FMM_HOMOGENEOUS_P95_REL = 0.01
# Float64 analytic network derivatives vs float64 centered differences.
ORACLE_FD_REL = 1e-4

REASONS = {
    "MEDIAN_TIME_REL": "Spec: median pointwise travel-time relative error <= 1%, per source and medium.",
    "P95_TIME_REL": "Proposed extra guard: 95th percentile <= 5%; PINN approximation, not float32 rounding.",
    "NETWORK_REL": "Project-wide float32 parity <= 1e-4 for fixed smooth network/objective/gradient.",
    "OBJECTIVE_SQRT_ATOL": "Dimensionless RMS residual floor; squaring near-zero float32 residuals destroys relative loss accuracy.",
    "GRADIENT_FD_REL": "Project-wide centered finite differences: relative vector L2 error <= 1e-3.",
    "SOURCE_TIME_ATOL": "Seconds; factored travel time vanishes algebraically at the source.",
    "PARAMETER_FD_STEP": "Dimensionless, centered h=0.002; float64 truncation audited in the oracle.",
    "COORDINATE_FD_STEP": "Metres, centered h=1; small relative to domain, large enough for float32 subtraction.",
    "FMM_HOMOGENEOUS_MEDIAN_REL": "Fine-grid FMM sanity only: point-source discretization, far from singularity.",
    "FMM_HOMOGENEOUS_P95_REL": "Fine-grid FMM sanity only: 1% tail vs distance/velocity, not a solver parity test.",
    "ORACLE_FD_REL": "Float64 oracle analytic/autodiff derivatives vs centered differences at recorded steps.",
}
