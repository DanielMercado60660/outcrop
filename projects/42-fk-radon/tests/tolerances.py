"""Locked numerical thresholds shared by the oracle and tests."""

# FFT and complex projection accumulate float32 rounding; normalized max error.
PARITY = 1e-4
# Synthetic component power must meet the required 20 dB suppression target.
MAX_RESIDUAL_POWER = 0.01
# The specification permits at most 1 dB change in primary power.
PRIMARY_POWER_MIN = 10 ** (-1 / 10)
PRIMARY_POWER_MAX = 10 ** (1 / 10)
