"""Locked numerical limits; all comparisons are against max reference scale."""

# Project-wide float32 limit, also encompassing the small analytic geometry limits.
RELATIVE_ERROR = 1e-4
# Two independent float32 paths may each err in opposite directions by 1e-4.
CROSS_PATH_RELATIVE_ERROR = 2e-4
# Exact zero density must produce exact zero; no cancellation or scale involved.
ZERO_ABSOLUTE_ERROR_MGAL = 0.0
