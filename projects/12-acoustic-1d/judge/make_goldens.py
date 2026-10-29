"""Create deterministic d'Alembert and matched-stencil float64 references."""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import correlate1d

OUT = Path(__file__).parent / 'golden'
OUT.mkdir(exist_ok=True)

def gaussian(x, center, width):
    return np.exp(-((x - center) / width) ** 2)

def reference(u0, prev, c, dx, dt, steps):
    """SciPy correlation for the centered spatial stencil, float64 in time."""
    old, prev = u0.copy(), prev.copy()
    out = [old.copy()]
    r = c * dt / dx
    q = (r - 1) / (r + 1)
    for _ in range(steps):
        lap = correlate1d(old, np.array([1., -2., 1.]), mode='constant')
        new = 2 * old - prev + r * r * lap
        new[0] = old[1] + q * (new[1] - old[0])
        new[-1] = old[-2] + q * (new[-2] - old[-1])
        prev, old = old, new
        out.append(old.copy())
    return np.stack(out)

def save(name, x, u0, prev, c, dx, dt, steps, exact):
    u0, prev = u0.astype(np.float32), prev.astype(np.float32)
    wave = reference(u0.astype(np.float64), prev.astype(np.float64), c, dx, dt, steps)
    np.savez_compressed(OUT / (name + '.npz'), x=x, u0=u0, u_prev=prev,
                        c=c, dx=dx, dt=dt, steps=steps, expected=wave, exact=exact)
    return wave

c = 1.0
for n in (101, 201, 401):
    x = np.linspace(0., 1., n)
    dx = 1 / (n - 1)
    dt = 0.5 * dx
    steps = (n - 1) // 2
    f = lambda z: gaussian(z, 0.5, 0.06)
    u0 = f(x)
    prev = (f(x + c * dt) + f(x - c * dt)) / 2
    t = steps * dt
    exact = (f(x - c * t) + f(x + c * t)) / 2
    wave = save(f'convergence_{n}', x, u0, prev, c, dx, dt, steps, exact)
    print(n, 'relative L2 error', np.linalg.norm(wave[-1] - exact) / np.linalg.norm(exact))

for direction in (-1, 1):
    n = 401
    x = np.linspace(0., 1., n)
    dx = 1 / (n - 1)
    dt = 0.5 * dx
    steps = 960
    f = lambda z: gaussian(z, 0.5, 0.06)
    u0 = f(x)
    prev = f(x + direction * c * dt)
    exact = f(x - direction * c * steps * dt)
    wave = save(f'reflection_{direction:+d}', x, u0, prev, c, dx, dt, steps, exact)
    print('direction', direction, 'residual peak', np.max(np.abs(wave[-1])))

meta = {
    'reference': 'SciPy correlate1d centered Laplacian, float64 leapfrog and Mur boundary; d’Alembert Gaussian solutions',
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
    'seed': 'not applicable: deterministic analytic pulses',
    'inputs': 'Uniform x=0..1 m, c=1 m/s, Courant number 0.5, Gaussian width 0.06 m. Convergence: n=101,201,401, t=0.25 s; reflection: n=401, t=1.2 s, both directions. Initial arrays rounded to float32 before reference.',
    'tolerances': {'parity': '1e-4 normalized max error, project-wide float32 limit', 'cross_path': '1e-4 normalized max error, accumulated float32 rounding', 'order_deviation': '0.2, project requirement for order 2', 'reflection': '0.01 of incident peak, project requirement'},
}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
