def ambient_noise(a, b, dt, low, high, eps):
    """Process paired real windows and return dimensionless float32 arrays.

    a, b: float32 arrays (windows, samples), equal shape. dt: sampling
    interval in seconds; low and high: inclusive frequency limits in Hz;
    eps: cutoff on the magnitude of the unnormalized forward FFT, all Python
    floats. Whiten each window independently: divide each real FFT bin by
    its magnitude when in band and magnitude strictly exceeds eps; set other
    bins to zero. The inverse FFT has 1/N normalization.

    Circular correlation is inverse FFT of conjugate(FFT(a_whitened)) times
    FFT(b_whitened). Shift zero lag to index N//2; positive lag means b is
    delayed relative to a. Stack is the arithmetic mean over windows.
    Return dict keys whitened_a, whitened_b, correlations, all (windows,
    samples), and stack (samples,). No gradient is required.
    """
    raise NotImplementedError
