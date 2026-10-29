"""Generate deterministic float64 Wiener/Levinson deconvolution references."""
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import solve_toeplitz

HERE = Path(__file__).resolve().parent / 'golden'
HERE.mkdir(parents=True, exist_ok=True)
wavelet = np.array([1.0, 0.5, -0.2], dtype=np.float32)
r = np.zeros(256, dtype=np.float32)
g = np.random.default_rng(4101)
indices = g.choice(np.arange(8, 220), size=35, replace=False)
r[indices] = g.uniform(-1, 1, len(indices)).astype(np.float32)
trace = np.convolve(r, wavelet).astype(np.float32)
length, eps = 32, 1e-6
w = wavelet.astype(np.float64)
a = np.zeros((len(w) + length - 1, length))
for j in range(length):
    a[j:j + len(w), j] = w
target = np.zeros(a.shape[0])
target[0] = 1
h = np.linalg.solve(a.T @ a + eps * np.eye(length), a.T @ target)
recovered = np.convolve(trace.astype(np.float64), h)[:len(trace)]
correlation = np.corrcoef(recovered[:len(r)], r)[0, 1]
np.savez(HERE / 'spiking.npz', trace=trace, wavelet=wavelet, reflectivity=r,
         inverse=h, recovered=recovered)
# A scaled wavelet makes absolute and relative diagonal loading distinct.
loaded_wavelet = (2 * wavelet).astype(np.float32)
loaded_trace = np.convolve(r, loaded_wavelet).astype(np.float32)
loaded_lambda = 0.2
loaded_a = 2 * a
gram = loaded_a.T @ loaded_a
rhs = loaded_a.T @ target
loaded_h = np.linalg.solve(gram + loaded_lambda * np.eye(length), rhs)
loaded_recovered = np.convolve(loaded_trace.astype(np.float64), loaded_h)[:len(loaded_trace)]
np.savez(HERE / 'spiking_loaded.npz', trace=loaded_trace, wavelet=loaded_wavelet,
         inverse=loaded_h, recovered=loaded_recovered)
unloaded_h = np.linalg.solve(gram, rhs)
relative_h = np.linalg.solve(gram + loaded_lambda * gram[0, 0] * np.eye(length), rhs)
def normalized_difference(x, y):
    return np.max(np.abs(x - y)) / np.max(np.abs(y))

print('loaded spiking inverse range', loaded_h.min(), loaded_h.max(),
      'unloaded difference', normalized_difference(unloaded_h, loaded_h),
      'relative loading difference', normalized_difference(relative_h, loaded_h))
assert normalized_difference(unloaded_h, loaded_h) > 1e-2
assert normalized_difference(relative_h, loaded_h) > 1e-2
for case, order, lag, seed in [('short', 8, 1, 4102), ('gap', 12, 3, 4103)]:
    noise = np.random.default_rng(seed).normal(size=192)
    x = np.convolve(noise, np.array([1.0, 0.65, -0.15, 0.08]))[:192].astype(np.float32)
    xd = x.astype(np.float64)
    ac = np.array([np.dot(xd[:len(xd)-k], xd[k:]) for k in range(lag + order)])
    col = ac[:order].copy()
    col[0] *= 1.01
    coeff = solve_toeplitz((col, col), ac[lag:lag + order])
    residual = xd.copy()
    for j, c in enumerate(coeff):
        residual[lag+j:] -= c * xd[:-(lag+j)]
    np.savez(HERE / f'predictive_{case}.npz', trace=x, coefficients=coeff, residual=residual)
    print(case, 'coeff range', coeff.min(), coeff.max(), 'energy ratio', np.dot(residual, residual)/np.dot(xd, xd))
print('spiking correlation', correlation, 'inverse range', h.min(), h.max())
assert correlation >= 0.9
meta = {'seed': {'spiking': 4101, 'short': 4102, 'gap': 4103},
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'inputs': {'spiking': {'length': 256, 'spikes': 35, 'wavelet': wavelet.tolist(), 'inverse_length': length, 'stabilization': eps},
                   'spiking_loaded': {'trace_length': len(loaded_trace), 'wavelet': loaded_wavelet.tolist(), 'inverse_length': length, 'stabilization': loaded_lambda, 'source': 'same reflectivity as spiking'},
                   'predictive': {'length': 192, 'short': {'order': 8, 'lag': 1}, 'gap': {'order': 12, 'lag': 3}, 'stabilization': 0.01}},
        'tolerances': {'relative_max_error': {'value': 1e-4, 'reason': 'float32 rounding against float64 reference'},
                       'minimum_correlation': {'value': 0.9, 'reason': 'required noise-free recovery threshold'}},
        'spiking_correlation': correlation}
(HERE / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
