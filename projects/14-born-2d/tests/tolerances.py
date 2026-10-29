"""Locked tolerances; errors use a single global reference maximum."""

# Float32 time stepping against the same float64 discrete Deepwave problem.
PARITY = 1e-4
# Both implementations accumulate float32 rounding independently.
CROSS = 1e-4
# Adjoint versus centered directional finite differences: project-wide gradient rule.
GRADIENT = 1e-3
# Independent float64 oracle finite differences have second-order truncation.
ORACLE_FD = 2e-6
# Second-order spatial/time scheme: halving h should reduce error by ~4;
# 3.2 allows pre-asymptotic effects and float32 accumulation.
CONVERGENCE_RATIO = 3.2
# Upper bound permits rounding while ruling out a misleading anomalous order.
CONVERGENCE_RATIO_MAX = 5.0
# Fine-grid absolute modal error has over 2x the reference error as margin.
CONVERGENCE_FINE = 0.0001
