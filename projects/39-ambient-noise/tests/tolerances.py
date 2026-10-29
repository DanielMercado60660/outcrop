# Peak-normalized error allows float32 FFT roundoff, including near zeros.
FFT_RELATIVE = 1e-4
# The two float32 FFT implementations may round differently.
CROSS_PATH_RELATIVE = 1e-4
# Required by the project specification for Green-function recovery.
GREEN_CORRELATION_MIN = 0.9
