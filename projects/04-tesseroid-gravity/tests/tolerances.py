"""All numerical pass thresholds for project 04."""

# Project-wide maximum relative error against matched float64 discretization.
MATCHED_REL = 1e-4
# Independent paths each have float32 rounding; retain the project-wide threshold.
CROSS_REL = 1e-4
# Adaptive GLQ is approximate; compared with a substantially pre-refined model.
# This bound is justified by the measured discretization error recorded in meta.
FINE_REL = 3e-3
