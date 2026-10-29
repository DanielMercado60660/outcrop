"""All numerical acceptance thresholds; reasons also copied to golden/meta.json."""

# Global infinity-norm relative error avoids meaningless division near field zeros.
# 1e-4 is the project-wide float32/reference ceiling at identical discretization.
PARITY_RTOL = 1.0e-4
# Additive allowance only for analytically zero outputs and wall roundoff.
ZERO_ATOL = 1.0e-6
# Pairwise float32 paths can differ on opposite sides of the reference by 1e-4.
CROSS_RTOL = 2.0e-4
# Two zero-reference paths may each use the absolute zero allowance above.
CROSS_ZERO_ATOL = 2.0e-6
# Long integration accumulates float32 roundoff; a 1e-4 normalized final-field
# ceiling is retained and checked against the float64 identical-step trajectory.
TRAJECTORY_RTOL = 1.0e-4
# Case 1a's explicitly mandated physical benchmark accuracy (discretization
# dominates float32); apply separately to both wall Nu values and RMS velocity.
BENCHMARK_RTOL = 0.01
# Second-order stencil refinement should reduce error by ~4; 1.8 permits the
# small-grid asymptotic correction and float32 noise without accepting order 1.
MIN_SPATIAL_ORDER = 1.8
# RK2 error ratios approach four as dt halves. 1.8 allows modest finite-dt error.
MIN_TEMPORAL_ORDER = 1.8
# Reject vacuous/nonconvergent tiny errors in the manufactured tests; this cap
# is far above float32 noise but below the coarse-grid second-order error.
FINE_STOKES_ERROR = 0.003
FINE_DIFFUSION_ERROR = 0.003
# A steady benchmark must have a small discrete thermal residual, relative to
# RMS diffusion. 5e-4 is tighter than the 1% diagnostic target and achievable
# above float32 cancellation in the five-point Laplacian at N=128.
STEADY_RESIDUAL_RTOL = 5.0e-4
# The centered-advection discrete scheme only balances wall flux to O(h^2).
# 0.5% is below the 1% physical benchmark target on the locked N=128 grid.
HEAT_BALANCE_RTOL = 0.005
# Centered streamfunction derivatives commute in the interior; allowance scales
# with max speed/h because derivative subtraction is done in float32.
DIVERGENCE_RTOL = 2.0e-6
# Physical maximum principle on this resolved benchmark, with float32 allowance.
TEMPERATURE_ATOL = 1.0e-5

REASONS = {
    "PARITY_RTOL": "Project-wide float32 ceiling, global infinity norm, identical grid/stencil/dt.",
    "ZERO_ATOL": "Float32 additive allowance for zero fields and exact walls only.",
    "CROSS_RTOL": "Two independent float32 paths, each within 1e-4 of float64.",
    "CROSS_ZERO_ATOL": "Sum of two 1e-6 absolute zero-reference allowances.",
    "TRAJECTORY_RTOL": "Retain project-wide 1e-4 ceiling for identical-step long trajectories.",
    "BENCHMARK_RTOL": "Explicit case 1a 1% published-value requirement; FD error dominates.",
    "MIN_SPATIAL_ORDER": "Second-order scheme; allow small-grid corrections, reject first order.",
    "MIN_TEMPORAL_ORDER": "Midpoint RK2 expected order 2; allow finite-dt corrections.",
    "FINE_STOKES_ERROR": "Fine-grid analytic Stokes error bound, discretization not roundoff.",
    "FINE_DIFFUSION_ERROR": "Fine-grid analytic heat error bound, discretization not roundoff.",
    "STEADY_RESIDUAL_RTOL": "Small discrete heat residual relative to RMS diffusion at N=128.",
    "HEAT_BALANCE_RTOL": "O(h^2) flux imbalance of centered discretization, below benchmark budget.",
    "DIVERGENCE_RTOL": "Centered derivative roundoff normalized by max speed / h.",
    "TEMPERATURE_ATOL": "Float32 allowance on physical benchmark temperature range [0,1].",
}
