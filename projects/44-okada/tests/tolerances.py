"""All numeric comparison thresholds used by this judge."""

# Add an absolute cancellation allowance to each
# printed interval, rather than applying relative error to a cancelled result.
# corner_scale is saved by the judge: absolute UA/UB/z*UC contributions before
# real/image and four-corner cancellation, per unit L1 slip (or potency).
# Largest |f32-f64| over two trig rounding choices is 3.44e-7. Its largest
# ratio to eps32*L1(slip)*corner_scale is 6.217; choose the next power of two,
# 8, rather than the old 1e-9 slack. Fixed per-input floors range 0..7.30e-6
# (exact symmetry zeros keep their separate bound below). This is a reviewed
# exception to literal printed-precision and universal 1e-4 relative parity;
# small final values suffer absolute four-corner cancellation.
FLOAT32_EPS = 2.0**-23
TABLE_CANCELLATION_ULPS = 8.0
# Published zeros are exact symmetries; allow float32 cancellation.
TABLE_ZERO_ABS = 1e-6
# Manual rounding interval, plus its REAL*4 computation error and the same
# slip/corner-scaled implementation allowance. The float64 displacements are
# (-37.898142817, 63.178935180, 14.960684758); their maximum discrepancy from
# the REAL*4 published decimals is 4.282e-5. The two float32 trig variants have
# max |f32-f64|=2.548e-4, or 1.920*eps32*L1(slip)*corner_scale. Factor 4 gives
# >2x that measured normalized error. Final absolute limits (ux,uy,uz) are
# (9.0908e-4, 6.3085e-4, 3.8298e-4), including both 5e-5 allowances.
# This replaces the infeasible 1e-4 criterion.
MANUAL_PRINTED_HALF_UNIT = 5e-5
MANUAL_REFERENCE_ABS = 5e-5
MANUAL_CANCELLATION_ULPS = 4.0
# Float32 path parity, with an absolute floor for cancellation near zero.
PARITY_REL = 1e-4
PARITY_ABS = 1e-6
# Length units; point case needs no corner subtraction, rectangle needs a
# larger step to reduce the displacement noise amplified by 1/(2h).
FD_STEPS = (0.01, 0.05)
# Project-wide relative gradient criterion. The absolute interval includes the
# measured float64 centered-FD truncation (point 7.53e-9, rectangle 1.842e-6).
# The point's existing 1e-6 floor already covers its 1.42e-8 subtraction noise.
FD_REL = 1e-3
FD_ABS = (1e-6, 2e-6)
# Rectangle displacement subtraction still has noise up to 4.961e-6 at h=.05.
# Add 2*eps32*L1(slip)*fd_corner_scale, where the saved scale is the sum of
# plus/minus displacement block scales divided by actual 2h, plus the base
# derivative block scale. Measured combined derivative/FD noise is at most
# 1.178 units of this scale in both trig variants. Factor 2 provides 70% room
# and explicitly accounts for amplification by 1/(2h), without a relative
# allowance tied to the small cancelled displacement. Point bound is unchanged.
FD_CANCELLATION_ULPS = (0.0, 2.0)
# Only coordinate quantization is tested with this bound (not physics).
FD_SPACING_ABS = 2e-7
