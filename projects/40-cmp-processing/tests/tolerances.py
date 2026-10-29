# Project-wide float32 comparison against float64 away from zero.
ARRAY_REL = 1e-4
# Four sample-index ulps times maximum trace slope allow moveout and interpolation rounding.
NMO_INDEX_ULPS = 4
# Two independently rounded paths can each incur that index error.
CROSS_INDEX_ULPS = 2 * NMO_INDEX_ULPS
# Float32 reduction error in energetic semblance windows, supported by simulations.
SEMBLANCE_ABS = 1e-5
# Two independent semblance errors can add.
CROSS_SEMBLANCE_ABS = 2 * SEMBLANCE_ABS
# Project requirement for picked stacking velocities.
VELOCITY_REL = 0.03
