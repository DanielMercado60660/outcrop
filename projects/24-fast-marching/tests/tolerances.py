"""All numeric acceptance thresholds, centralized for review and locking."""

# Maximum absolute difference normalized by the maximum reference travel time.
# This is the project-wide 1e-4 float32 parity allowance, including rounding.
PARITY_REL = 1e-4
# Same normalized allowance for independent float32 paths; each also has to
# meet the reference, so this deliberately requires closer mutual agreement.
CROSS_REL = 1e-4
# The first-order analytic errors must shrink by at least 20% on each halving;
# the ideal asymptotic first-order ratio is 0.5, with room for source effects.
REFINEMENT_RATIO = 0.8
# Guard against accidental tiny errors making the ratio test meaningless.
MIN_COARSE_ERROR = 1e-5
# The published second-order variant uses different near-interface and trial
# update rules from scikit-fmm; this is a coarse oracle sanity check, not parity.
ORDER2_SKFMM_SANITY_REL = 5e-2
# Judge-only max absolute error, in time units, between the independent
# float64 solver following tests/api.py and each saved oracle. About 1e-10
# is far below float32 parity, while allowing accumulated float64 roundoff
# from scikit-fmm's unshifted quadratic on the finest grids.
DOCUMENTED_RULES_ATOL = 1e-10
