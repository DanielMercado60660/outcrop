"""Project API for the Metal implementation."""


def classic_sta_lta(trace, nsta, nlta):
    """Return {'characteristic': float32 array, shape (N,)} (dimensionless).

    trace is a finite float32 array (N,), amplitude in arbitrary consistent
    units. nsta and nlta are Python integers in samples, N >= nlta >= nsta >= 1.
    Each value is the ratio of trailing means of squared samples, including the
    current sample. The first nlta-1 values are exactly zero. Each full long
    window must have positive energy. No time origin or sampling rate is assumed.
    """
    raise NotImplementedError


def trigger_onset(characteristic, on, off):
    """Return {'triggers': int64 array (K, 2)} of inclusive sample indices.

    characteristic is a finite nonnegative float32 array (N,), dimensionless.
    on and off are Python floats, dimensionless, on > off > 0. Activate at
    >= on; end at the last sample >= off before the next sample < off. An
    active event at the end ends at N-1. Merge on crossings while active.
    Pairs are zero-based [on, off]; K can be zero. No duration limit applies.
    """
    raise NotImplementedError
