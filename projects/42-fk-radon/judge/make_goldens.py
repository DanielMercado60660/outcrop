"""Create deterministic float64 analytical Fourier/projection benchmarks."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / 'golden'
OUT.mkdir(exist_ok=True)
nx, nt, dt, dx = 32, 256, 0.004, 10.0
x = np.arange(nx, dtype=np.float64) * dx
t = np.arange(nt, dtype=np.float64) * dt
# Exactly periodic plane waves avoid spectral leakage at the f-k mask boundary.
phase = 2 * np.pi * (np.arange(nt)[None, :] / nt)
space = 2 * np.pi * np.arange(nx)[:, None] / nx
primary = np.cos(40 * phase - 2 * space)
noise = 1.5 * np.cos(8 * phase - 6 * space)
cutoff = 500.0
f = np.fft.fftfreq(nt, dt)
k = np.fft.fftfreq(nx, dx)
mask = np.abs(f)[None, :] > cutoff * np.abs(k)[:, None]
def fk(a):
    return np.fft.ifft2(np.fft.fft2(a) * mask).real
np.savez_compressed(OUT / 'fk.npz', gather=primary + noise, primary=primary,
                    unwanted=noise, expected=fk(primary + noise), dt=dt, dx=dx,
                    cutoff=cutoff)

# Broadband fixed-seed input probes the entire mask, including the nearest
# bins on each side. Remove boundary bins and their conjugates explicitly:
# integer indices obey 5*|n| == 8*|m| at the chosen cutoff.  This avoids
# making float32 cutoff rounding part of the oracle.
rng = np.random.default_rng(4202)
m = np.rint(np.fft.fftfreq(nx) * nx).astype(int)
n = np.rint(np.fft.fftfreq(nt) * nt).astype(int)
boundary = 5 * np.abs(n)[None, :] == 8 * np.abs(m)[:, None]
spectrum = np.fft.fft2(rng.standard_normal((nx, nt)))
spectrum[boundary] = 0
broadband = np.fft.ifft2(spectrum).real
np.savez_compressed(OUT / 'fk_broadband.npz', gather=broadband,
                    expected=fk(broadband), dt=dt, dx=dx, cutoff=cutoff)

# Two delayed parabolic Ricker events with different moveout.  The projection
# is exact at positive/negative frequencies; DC and Nyquist are zeroed so the
# real-valued gather and the DFT steering convention are exactly consistent.
offsets = (np.arange(nx) - (nx - 1) / 2) * dx
slow = 0.00008
freq = np.fft.fftfreq(nt, dt)
wave = (freq / 18) ** 2 * np.exp(-(freq / 18) ** 2)
wave[0] = wave[nt // 2] = 0
def event(p, delay, amplitude):
    spectrum = amplitude * wave[None, :] * np.exp(-2j * np.pi * freq[None, :] *
        (delay + p * offsets[:, None] ** 2))
    return np.fft.ifft(spectrum, axis=1).real
multiple = event(slow, 0.36, 2.0)
primary = event(-0.00004, 0.25, 1.0)
steer = np.exp(-2j * np.pi * offsets[:, None] ** 2 * slow * freq[None, :])
def radon(a):
    d = np.fft.fft(a, axis=1)
    return np.fft.ifft(d - steer * np.mean(steer.conj() * d, axis=0), axis=1).real
np.savez_compressed(OUT / 'radon.npz', gather=primary + multiple, primary=primary,
                    unwanted=multiple, expected=radon(primary + multiple),
                    offsets=offsets, dt=dt, slowness=slow)
# A spatially varying constant and a common constant exercise DC removal and
# preservation, respectively, alongside broadband frequencies and Nyquist.
radon_broadband = rng.standard_normal((nx, nt)) + 2 + offsets[:, None] / 100
np.savez_compressed(OUT / 'radon_broadband.npz', gather=radon_broadband,
                    expected=radon(radon_broadband), offsets=offsets, dt=dt,
                    slowness=slow)
def power_ratio(a, b):
    return float(np.sum(a*a) / np.sum(b*b))
metrics = {'fk_unwanted_power_ratio': power_ratio(fk(noise), noise),
           'fk_primary_power_ratio': power_ratio(fk(np.cos(40 * phase - 2 * space)), np.cos(40 * phase - 2 * space)),
           'radon_unwanted_power_ratio': power_ratio(radon(multiple), multiple),
           'radon_primary_power_ratio': power_ratio(radon(primary), primary)}
# Check that representative unit and cutoff mistakes are observable in the
# broadband fixture; use the same normalized max-error metric as the tests.
def mask_error(alternative):
    candidate = np.fft.ifft2(np.fft.fft2(broadband) * alternative).real
    return float(np.max(np.abs(candidate - fk(broadband))) /
                 np.max(np.abs(fk(broadband))))
metrics['fk_broadband_wrong_k_radians_error'] = mask_error(
    np.abs(f)[None, :] > cutoff * 2 * np.pi * np.abs(k)[:, None])
metrics['fk_broadband_double_cutoff_error'] = mask_error(
    np.abs(f)[None, :] > 2 * cutoff * np.abs(k)[:, None])
metrics['fk_broadband_triple_cutoff_error'] = mask_error(
    np.abs(f)[None, :] > 3 * cutoff * np.abs(k)[:, None])
assert min(metrics[key] for key in metrics if key.endswith('_error')) > 1e-4
assert metrics['fk_unwanted_power_ratio'] < .01 and metrics['radon_unwanted_power_ratio'] < .01
assert 10**(-.1) <= metrics['radon_primary_power_ratio'] <= 10**(.1)
meta = {'numpy': np.__version__, 'reference': 'NumPy float64 FFT and analytic orthogonal projection',
        'seed': 4202, 'inputs': {'nx': nx, 'nt': nt,
        'dt_seconds': dt, 'dx_meters': dx, 'fk_cutoff_m_per_s': cutoff,
        'radon_slowness_seconds_per_m2': slow,
        'broadband': 'seeded Gaussian; f-k exact-boundary DFT bins removed; Radon includes spatially varying DC'}, 'metrics': metrics,
        'tolerances': {'normalized_max_error': {'value': 1e-4, 'reason': 'float32 FFT and projection rounding'},
        'unwanted_power_ratio': {'value': 0.01, 'reason': '20 dB suppression'},
        'primary_power_ratio': {'value': [10**(-.1), 10**(.1)], 'reason': '1 dB preservation'}}}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(metrics, indent=2))
