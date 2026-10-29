"""All numeric acceptance thresholds (locked)."""

# Project-wide float32 limit: global max-absolute error / reference max-absolute
# amplitude PER component, with no pointwise relative errors near trace zeros.
# For the nominally absent plane-wave channels, roundoff/finite-sheet tails
# have no meaningful relative scale: normalize by dominant velocity amplitude,
# or rho*vp times that amplitude for pressure (physical impedance conversion).
PARITY = 1.0e-4
# Two independently rounded float32 paths can each spend the parity budget.
CROSS = 2.0e-4
# Project-wide centered FD limit; relative multiplicative perturbations avoid
# incompatible physical scales among speeds, density and force density.
GRAD_FD = 1.0e-3
FD_STEP = 1.0e-3
# Float64 oracle adjoint check: h=1e-4, truncation O(h^2), CPU rounding tiny.
ORACLE_FD_STEP = 1.0e-4
ORACLE_GRAD_FD = 2.0e-5
# Explicit project pass criterion, applied to each measured Rayleigh arrival.
RAYLEIGH_TIME = 0.01
# O(h^2) expected; allow order >=1.8 for finite-grid pre-asymptotic effects
# and <=2.3 to catch a broken analytic comparison/vanishing numerical errors.
ORDER_MIN = 1.8
ORDER_MAX = 2.3
# Finest plane-wave relative L2 error <=0.35% includes discretization AND the
# FULL allowed max-norm parity budget. By the triangle inequality its bound is
# E_ref + PARITY*peak_ref*sqrt(nt)/norm(analytic). The oracle records this bound
# for each mode and asserts it fits below 0.0035 (also checked by the integrity
# test). An earlier bound of 0.003 sat only 0.8% above S's E_ref=0.002976, insufficient
# for independently ordered float32 arithmetic. Convergence order is unchanged.
PLANE_FINE_ERROR = 3.5e-3
# C-PML late reflected L2 norm <=1% of incident full-record L2 norm vs a large
# domain; approximate finite absorbing layers cannot have exact zero echo.
PML_REFLECTION = 0.01
# Negative control must reflect at least 10% and C-PML improve it >=20x.
REFLECTIVE_MIN = 0.10
PML_IMPROVEMENT = 20.0
# Float64 analytic polynomial root and physical speed sanity (not GPU parity).
RAYLEIGH_ROOT_RESIDUAL = 1.0e-10
# Float64 Brent solver termination, far below the arrival-time error budget.
RAYLEIGH_ROOT_XTOL = 1.0e-14
# Output of a zero initial-state zero-source linear propagator is exactly zero.
ZERO = 0.0
# Frozen contract requires total stored goldens under 20 MB (decimal bytes).
GOLDEN_BYTES_LIMIT = 20_000_000
