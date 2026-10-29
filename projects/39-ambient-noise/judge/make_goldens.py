"""Generate float64 FFT references and an analytic 1-D Green function."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'golden'
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(39001)


def reference(a, b, dt, low, high, eps):
    n = a.shape[1]
    f = np.fft.rfftfreq(n, dt)
    mask = (f >= low) & (f <= high)

    def whiten(x):
        z = np.fft.rfft(x.astype(np.float64), axis=-1)
        m = np.abs(z)
        w = np.zeros_like(z)
        np.divide(z, m, out=w, where=(m > eps) & mask)
        return np.fft.irfft(w, n=n, axis=-1)

    wa, wb = whiten(a), whiten(b)
    c = np.fft.fftshift(np.fft.irfft(np.conj(np.fft.rfft(wa, axis=-1)) * np.fft.rfft(wb, axis=-1), n=n, axis=-1), axes=-1)
    return wa, wb, c, c.mean(axis=0)


# Equal populations of waves propagating in each direction in a homogeneous
# 1-D medium yield a symmetric impulse pair at the inter-receiver delay.
n, count, delay = 256, 128, 17
a = rng.standard_normal((count, n)).astype(np.float32)
direction = np.r_[np.ones(count // 2, dtype=int), -np.ones(count // 2, dtype=int)]
rng.shuffle(direction)
b = np.stack([np.roll(x, int(d * delay)) for x, d in zip(a, direction)])
wa, wb, c, stack = reference(a, b, 0.01, 3.0, 25.0, 1e-10)
f = np.fft.rfftfreq(n, 0.01)
impulses = np.zeros(n)
impulses[delay] = impulses[-delay] = 0.5
green = np.fft.fftshift(np.fft.irfft(np.fft.rfft(impulses) * ((f >= 3.0) & (f <= 25.0)), n=n))
coef = np.corrcoef(stack, green)[0, 1]
assert coef > 0.999999
assert np.argmax(stack) in (n // 2 - delay, n // 2 + delay)
print('Green correlation:', coef, 'stack peak:', float(stack.max()), 'peak lag:', int(np.argmax(stack) - n // 2))
np.savez_compressed(OUT / 'diffuse.npz', a=a, b=b, whitened_a=wa, whitened_b=wb, correlations=c, stack=stack, green=green)

# Odd length, zero traces, a DC signal, and unequal random traces.
n = 63
a = rng.standard_normal((4, n)).astype(np.float32)
b = rng.standard_normal((4, n)).astype(np.float32)
a[0] = 0
b[1] = 0
a[2] = 2
b[2] = np.cos(2 * np.pi * 5 * np.arange(n) / n)
wa, wb, c, stack = reference(a, b, 0.02, 0.0, 18.0, 0.1)
np.savez_compressed(OUT / 'edge.npz', a=a, b=b, whitened_a=wa, whitened_b=wb, correlations=c, stack=stack)

# Include the real Nyquist bin for an even length, as well as DC.
n = 64
a = rng.standard_normal((3, n)).astype(np.float32)
b = rng.standard_normal((3, n)).astype(np.float32)
a[0] = (-1.0) ** np.arange(n)
b[0] = -a[0]
wa, wb, c, stack = reference(a, b, 0.02, 0.0, 25.0, 0.1)
assert np.allclose(wa[0], a[0] / n)
assert np.isclose(c[0, n // 2], -1.0 / n)
np.savez_compressed(OUT / 'nyquist.npz', a=a, b=b, whitened_a=wa, whitened_b=wb, correlations=c, stack=stack)

# Report the nearest magnitudes to each cutoff, so a binary whitening
# decision cannot accidentally depend on float32 FFT roundoff.
case_margins = {}
for name, dt, low, high, eps in [('diffuse', 0.01, 3.0, 25.0, 1e-10),
                                  ('edge', 0.02, 0.0, 18.0, 0.1),
                                  ('nyquist', 0.02, 0.0, 25.0, 0.1)]:
    with np.load(OUT / f'{name}.npz') as data:
        n = data['a'].shape[-1]
        frequencies = np.fft.rfftfreq(n, dt)
        band = (frequencies >= low) & (frequencies <= high)
        magnitudes = np.concatenate([np.abs(np.fft.rfft(data[k].astype(np.float64), axis=-1))[:, band].ravel() for k in ('a', 'b')])
        rejected = magnitudes[magnitudes <= eps]
        retained = magnitudes[magnitudes > eps]
        case_margins[name] = {'largest_rejected': float(rejected.max()) if rejected.size else None,
                              'smallest_retained': float(retained.min())}
        print(name, 'cutoff margins:', case_margins[name])
        assert retained.min() > eps * 1.5
        if rejected.size:
            assert rejected.max() < eps * (0.001 if name == 'edge' else 0.5)

meta = {'seed': 39001, 'numpy': np.__version__, 'reference': 'NumPy float64 FFT and analytic 1D bidirectional impulse Green function', 'cases': {'diffuse': {'shape': [128, 256], 'dt_s': 0.01, 'band_hz': [3, 25], 'eps': 1e-10, 'delay_samples': 17, 'green_correlation': coef}, 'edge': {'shape': [4, 63], 'dt_s': 0.02, 'band_hz': [0, 18], 'eps': 0.1}, 'nyquist': {'shape': [3, 64], 'dt_s': 0.02, 'band_hz': [0, 25], 'eps': 0.1}}, 'cutoff_magnitudes': case_margins, 'tolerances': {'fft_relative': '1e-4 peak normalized for float32 FFT roundoff', 'cross_path_relative': '1e-4 peak normalized for float32 backend parity', 'green_correlation': 'at least 0.9, project criterion'}}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
