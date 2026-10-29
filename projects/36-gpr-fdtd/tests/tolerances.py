# Accumulated float32 arithmetic relative to the same-grid float64 reference.
PARITY = 1e-4
# Independent implementations have comparable float32 accumulation errors.
CROSS = 1e-4
# In the x-uniform reflection case Hz is analytically zero; an absolute
# 1e-6 V/m bound admits float32 roundoff relative to the unit input pulse.
ZERO_HZ_ABSOLUTE = 1e-6
# A second-order method should reduce errors by approximately four per halving.
CONVERGENCE_RATIO_MAX = 0.30
# Normal-incidence Fresnel amplitude is discretized at a dielectric interface.
REFLECTION_RELATIVE = 0.02
