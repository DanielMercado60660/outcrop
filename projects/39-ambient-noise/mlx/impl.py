"""Windowed spectral whitening and circular ambient-noise correlation."""

import mlx.core as mx
import numpy as np


def ambient_noise(a, b, dt, low, high, eps):
    """Whiten paired windows, correlate them, and average their correlations."""
    a = mx.array(a, dtype=mx.float32)
    b = mx.array(b, dtype=mx.float32)
    n = a.shape[-1]

    # Evaluate the bin frequencies in host float64, including both endpoints.
    frequencies = np.arange(n // 2 + 1, dtype=np.float64) / (n * dt)
    band = mx.array((frequencies >= low) & (frequencies <= high))

    def whiten(values):
        spectrum = mx.fft.rfft(values, axis=-1)
        magnitude = mx.abs(spectrum)
        keep = band & (magnitude > eps)
        denominator = mx.where(keep, magnitude, 1.0)
        whitened_spectrum = mx.where(keep, spectrum / denominator, 0.0)
        return mx.fft.irfft(whitened_spectrum, n=n, axis=-1).astype(mx.float32)

    whitened_a = whiten(a)
    whitened_b = whiten(b)
    spectrum_a = mx.fft.rfft(whitened_a, axis=-1)
    spectrum_b = mx.fft.rfft(whitened_b, axis=-1)
    correlation = mx.fft.irfft(mx.conj(spectrum_a) * spectrum_b, n=n, axis=-1)
    correlations = mx.roll(correlation, n // 2, axis=-1).astype(mx.float32)
    return {
        "whitened_a": whitened_a,
        "whitened_b": whitened_b,
        "correlations": correlations,
        "stack": mx.mean(correlations, axis=0).astype(mx.float32),
    }
