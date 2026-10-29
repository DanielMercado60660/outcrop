"""Deterministic image measurements, with no modeling or reference libraries."""

import numpy as np

WIDTH_DEFINITION = {
    "name": "full-column analytic-signal envelope FWHM",
    "axis": "z (axis 0), separately for each saved x column",
    "transform_length": "nz, the full physical column length (69 for layered)",
    "padding": "none; periodic DFT extension; no taper, detrending or smoothing",
    "analytic_signal": "ifft(fft(column)*h): h[0]=1; positive frequencies=2; negative frequencies=0; even-N Nyquist=1",
    "selection": "largest envelope sample in each saved half-open [lo,hi) depth window; first maximum on ties",
    "crossings": "first contiguous half-peak crossings inside the same window, linearly interpolated between samples",
    "rejection": "nonfinite or zero column/window, or either crossing outside the window",
    "units": "grid cells; multiply by dz for metres; ratios independent of dz",
}


def column_envelopes(image, columns):
    """Full-depth analytic-signal magnitudes, float64 (nz,len(columns)).

    The transform uses the signed image, before taking any absolute value or
    extracting depth windows. For N samples, double bins 1..(N-1)//2, retain
    DC and the even-N Nyquist bin, and zero negative-frequency bins. Use N-point
    FFT/IFFT without padding. Thus the Hilbert convention is completely fixed.
    """
    profiles = np.asarray(image, np.float64)[:, np.asarray(columns, dtype=int)]
    assert np.all(np.isfinite(profiles))
    n = profiles.shape[0]
    h = np.zeros(n, np.float64)
    h[0] = 1.0
    h[1:(n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    return np.abs(np.fft.ifft(np.fft.fft(profiles, axis=0) * h[:, None], axis=0))


def reflector_metrics(image, columns, windows):
    """Return (peak depths, FWHMs) in cells, shape (reflector, column).

    Compute the full-column analytic-signal envelope with column_envelopes;
    only then select each half-open depth window. Follow the largest envelope
    peak's contiguous half-height lobe to the first crossings; interpolate
    crossings linearly. Reject truncated lobes and empty images. Never clip
    widths, upsample, smooth, or search for a narrower neighboring side lobe.

    This measures the reflected wave packet rather than one oscillation of
    its carrier: side-lobe ringing broadens RTM's packet even when its sampled
    raw-amplitude central lobe is near the one-cell FWHM floor.
    """
    envelopes = column_envelopes(image, columns)
    depths, widths = [], []
    for lo, hi in np.asarray(windows, dtype=int):
        ds, ws = [], []
        for column in range(envelopes.shape[1]):
            profile = envelopes[lo:hi, column]
            p = int(np.argmax(profile))
            assert profile[p] > 0.0, "empty reflector window"
            half = profile[p] / 2.0
            left = right = p
            while left > 0 and profile[left] >= half:
                left -= 1
            while right < len(profile) - 1 and profile[right] >= half:
                right += 1
            assert profile[left] < half and profile[right] < half, "truncated FWHM"
            lcross = left + (half - profile[left]) / (profile[left + 1] - profile[left])
            rcross = right - 1 + (half - profile[right - 1]) / (profile[right] - profile[right - 1])
            ds.append(lo + p)
            ws.append(rcross - lcross)
        depths.append(ds)
        widths.append(ws)
    return np.asarray(depths), np.asarray(widths)
