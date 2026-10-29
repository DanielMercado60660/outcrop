"""Frozen CMP interface. All input and output arrays are float32."""


def nmo(gather, offsets, dt, velocity):
    """Return {'corrected': (n_offset, n_time) float32 amplitudes}.

    gather is (n_offset, n_time) float32 amplitudes, sampled at t=j*dt from
    time zero; offsets is (n_offset,) float32 signed source-receiver distances in metres. dt is
    a positive Python float in seconds, velocity a positive Python float in
    metres/second. At each output time t0 sample each input trace by linear
    interpolation at sqrt(t0**2+(offset/velocity)**2); values past the last
    input sample are zero. Apply no stretch mute or amplitude scaling.
    """
    raise NotImplementedError


def semblance(gather, offsets, dt, velocities, half_window):
    """Return {'semblance': (n_velocity, n_time) float32 dimensionless}.

    gather, offsets and dt follow nmo. velocities is (n_velocity,) float32 positive
    trial speeds in metres/second. half_window is a nonnegative Python int in
    samples. For each velocity and time, compute nmo and use the inclusive
    time window clipped to the trace: sum_k(sum_i a[i,k])**2 divided by
    n_offset * sum_k sum_i a[i,k]**2. Return zero for a zero denominator.
    Zero-filled samples count in n_offset; do not mute or scale amplitudes.
    """
    raise NotImplementedError


def stack(gather, offsets, dt, velocity):
    """Return {'stack': (n_time,) float32 amplitudes}, mean over NMO traces.

    All arguments, units and conventions are identical to nmo. Divide by the
    total number of offsets, including any zero-filled out-of-range samples.
    """
    raise NotImplementedError
