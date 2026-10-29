"""Relative thresholds use the maximum absolute reference value in each group."""

# Float32 rounding near the source, including nominally zero components.
NEAR_REL = 1e-4
# Translation uses exactly representable float32 coordinates.
TRANSLATION_REL = 1e-4
# Two independently rounded implementations can each incur an oracle error.
CROSS_REL = 2e-4
# Corner-term cancellation grows with distance (2, 4, 8, 16 prism widths).
# Bounds include at least 1.9x margin over the independently measured worst
# NumPy/MLX float32 kernel error in each group; see golden/meta.json.
DISTANCE_REL = (1e-4, 7e-4, 5e-3, 4e-2)
# Both implementations may err in opposite directions at each distance.
DISTANCE_CROSS_REL = tuple(2 * x for x in DISTANCE_REL)
# Float64 oracle centered differences incur truncation; float32 fields add rounding.
GRADIENT_REL = 1e-3
