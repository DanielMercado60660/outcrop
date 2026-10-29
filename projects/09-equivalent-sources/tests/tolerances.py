"""Locked numerical thresholds for equivalent-source predictions."""

# The project specification permits 1e-3 for this approximate, regularized inverse problem:
# float32 least-squares and sequential boosted residual updates accumulate error.
GRID_REL_ERR = 1e-3
# Both implementations approximate the same solution; use the same
# prediction threshold for their direct comparison as well.
CROSS_PATH_REL_ERR = 1e-3
