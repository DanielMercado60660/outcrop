"""Periodic f-k filtering and single-component parabolic projection.

NumPy constructs only fixed frequency grids, phase scales, and the f-k mask.
All operations on gather amplitudes and Radon offsets use MLX and retain their
autodiff graph.
"""

import mlx.core as mx
import numpy as np


def fk_filter(gather, dt, dx, velocity_cutoff):
    """Filter a (space, time) gather using frequencies in Hz and cycles/m.

    Sampling intervals are seconds and meters; the cutoff is meters/second.
    Return the real inverse DFT in the original shape and amplitude units.
    """
    values = mx.array(gather, dtype=mx.float32)
    nx, nt = values.shape
    # Build the fixed mask in float64 so boundary decisions use the supplied
    # sampling intervals without an additional float32 rounding step.
    f = np.fft.fftfreq(nt, d=dt)[None, :]
    k = np.fft.fftfreq(nx, d=dx)[:, None]
    # Strictly greater keeps neither the exact cutoff nor DC (0 > 0 is false).
    keep = mx.array(np.abs(f) > velocity_cutoff * np.abs(k))
    spectrum = mx.fft.fft2(values, axes=(0, 1))
    filtered = mx.real(mx.fft.ifft2(spectrum * keep, axes=(0, 1)))
    return {"filtered": filtered.astype(mx.float32)}


def radon_demultiple(gather, offsets, dt, slowness):
    """Subtract a single known-slowness component at every time frequency.

    Offsets are meters from the midpoint and slowness is seconds/meter**2.
    The steering delay is slowness * offsets**2, with no recentering or shift.
    """
    values = mx.array(gather, dtype=mx.float32)
    x = mx.array(offsets, dtype=mx.float32)
    nx, nt = values.shape
    frequencies = np.fft.fftfreq(nt, d=dt)
    f = mx.array(frequencies.astype(np.float32))
    # Every float32 magnitude below 2**64 has a finite float32 square.
    # Preserve the existing arithmetic there, masking before the square so
    # the unused branch cannot introduce infinities into offset gradients.
    large = mx.abs(x) >= 2.0**64
    ordinary_x = mx.where(large, 0.0, x)
    phase = (-2.0 * np.pi * slowness) * ordinary_x[:, None] ** 2 * f[None, :]

    # For large offsets, factor |2*pi*slowness*f| into four equal factors:
    # (x * scale * scale)**2. Taking each scalar's fourth root separately
    # avoids underflow of tiny slowness or overflow of the scalar product.
    # These fixed scales use host float64; offset arithmetic stays in MLX.
    scale = mx.array(
        (2.0 * np.pi) ** 0.25
        * abs(float(slowness)) ** 0.25
        * np.abs(frequencies) ** 0.25,
        dtype=mx.float32,
    )
    scaled_x = mx.where(large, x, 0.0)[:, None] * scale[None, :]
    scaled_x = scaled_x * scale[None, :]
    direction = mx.array(
        -np.sign(float(slowness)) * np.sign(frequencies), dtype=mx.float32
    )
    phase = mx.where(large[:, None], direction[None, :] * scaled_x**2, phase)
    steering = mx.cos(phase) + 1j * mx.sin(phase)
    spectrum = mx.fft.fft(values, axis=1)
    # Each steering entry has unit magnitude, hence a^H a = nx. Keeping the
    # full signed frequency grid also handles DC and the even-length Nyquist
    # bin according to the API before taking the inverse transform's real part.
    # At DC, steering is all ones, so this also removes the spatial mean of
    # the DC coefficients while retaining their spatial variation.
    coefficient = mx.sum(mx.conj(steering) * spectrum, axis=0, keepdims=True) / nx
    residual = spectrum - steering * coefficient
    filtered = mx.real(mx.fft.ifft(residual, axis=1))
    return {"filtered": filtered.astype(mx.float32)}
