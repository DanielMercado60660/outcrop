"""Locked numerical limits for float32 comparisons."""

# Project-wide float32 limit, per-ray endpoint distance error / receiver distance.
ENDPOINT_REL = 1e-4
# Project-wide float32 limit, per-ray relative travel-time error.
TIME_REL = 1e-4
# One-radian scale for angle errors, including vertical rays.
ANGLE_REL = 1e-4
ANGLE_SCALE = 1.0
# Two implementations can each differ from the reference by the allowed error.
CROSS_FACTOR = 2.0
