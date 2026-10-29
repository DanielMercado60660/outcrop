"""All numerical acceptance thresholds for this project."""

# Source-local near and intermediate observations: project-wide float32 bound.
NEAR_REL = 1e-4
# At 8–12 prism widths, alternating corner terms cancel strongly in float32;
# 2% is a far-field bound accepted after review, not a claim of 1e-4 accuracy.
FAR_REL = 2e-2
# Two float32 implementations can err on opposite sides of the oracle.
CROSS_NEAR_REL = 2 * NEAR_REL
CROSS_FAR_REL = 2 * FAR_REL
# Projection uses already rounded float32 output components and a dot product.
PROJECTION_REL = 2e-6
# Absolute nT floor for symmetry/zero fields and projection cancellation.
ZERO_ABS_NT = 2e-5
