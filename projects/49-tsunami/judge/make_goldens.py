"""Generate float64 deterministic shallow-water references and analytic solutions."""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import correlate1d
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent / 'golden'
G = 9.81


def advance(initial, dx, dt, steps, mode, depth):
    """Forward Euler, finite volume, Rusanov flux; periodic faces."""
    u = np.array(initial, dtype=np.float64, copy=True)
    for _ in range(steps):
        right = np.roll(u, -1, axis=0)
        if mode == 'linear':
            f = np.stack((u[:, 1], G * depth * u[:, 0]), axis=1)
            fr = np.stack((right[:, 1], G * depth * right[:, 0]), axis=1)
            a = np.sqrt(G * depth)
        else:
            f = np.stack((u[:, 1], u[:, 1] ** 2 / u[:, 0] + 0.5 * G * u[:, 0] ** 2), axis=1)
            fr = np.stack((right[:, 1], right[:, 1] ** 2 / right[:, 0] + 0.5 * G * right[:, 0] ** 2), axis=1)
            a = np.maximum(np.abs(u[:, 1] / u[:, 0]) + np.sqrt(G * u[:, 0]),
                           np.abs(right[:, 1] / right[:, 0]) + np.sqrt(G * right[:, 0]))[:, None]
        flux = 0.5 * (f + fr - a * (right - u))
        delta = correlate1d(flux, [-1., 1., 0.], axis=0, mode='wrap')
        u -= dt / dx * delta
    return u


def dam_exact(x, t, hl, hr):
    """Exact wet-bed Riemann solution: left rarefaction and right shock."""
    cl = np.sqrt(G * hl)
    def phi(h):
        return 2 * (np.sqrt(G * h) - cl) + (h - hr) * np.sqrt(0.5 * G * (1 / h + 1 / hr))
    hs = brentq(phi, hr, hl, xtol=1e-14)
    us = 2 * (cl - np.sqrt(G * hs))
    shock = hs * us / (hs - hr)
    xi = x / t
    h = np.full_like(x, hr)
    vel = np.zeros_like(x)
    star = xi < shock
    h[star], vel[star] = hs, us
    fan = xi < us - np.sqrt(G * hs)
    c = (2 * cl - xi[fan]) / 3
    h[fan] = c * c / G
    vel[fan] = 2 * (cl + xi[fan]) / 3
    left = xi < -cl
    h[left], vel[left] = hl, 0
    return np.stack((h, h * vel), axis=1), hs, shock


def advance_float32_perturbation(initial, dx, dt, steps, mode, depth):
    """Sanity-check float32 evolution with constant hydrostatic pressure removed."""
    base = np.float32(depth)
    u = np.asarray(initial, dtype=np.float32).copy()
    if mode == 'nonlinear':
        u[:, 0] -= base
    g = np.float32(G)
    factor = np.float32(dt / dx)
    for _ in range(steps):
        right = np.roll(u, -1, axis=0)
        if mode == 'linear':
            a = np.sqrt(g * base)
            f = np.stack((u[:, 1], g * base * u[:, 0]), axis=1)
            fr = np.stack((right[:, 1], g * base * right[:, 0]), axis=1)
        else:
            h, hr = base + u[:, 0], base + right[:, 0]
            f = np.stack((u[:, 1], u[:, 1] ** 2 / h + g * u[:, 0] * (base + 0.5 * u[:, 0])), axis=1)
            fr = np.stack((right[:, 1], right[:, 1] ** 2 / hr + g * right[:, 0] * (base + 0.5 * right[:, 0])), axis=1)
            a = np.maximum(np.abs(u[:, 1] / h) + np.sqrt(g * h),
                           np.abs(right[:, 1] / hr) + np.sqrt(g * hr))[:, None]
        flux = np.float32(0.5) * (f + fr - a * (right - u))
        u -= factor * (flux - np.roll(flux, 1, axis=0))
    if mode == 'nonlinear':
        u[:, 0] += base
    return u.astype(np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4901)
    cases = {}
    for n in (100, 200, 400):
        length, depth, steps = 1000., 10., n // 4
        dx = length / n
        c = np.sqrt(G * depth)
        dt = 0.4 * dx / c
        x = (np.arange(n) + 0.5) * dx
        eta = np.sin(2 * np.pi * x / length) * np.sinc(1 / n)
        initial = np.stack((eta, c * eta), axis=1)
        exact_eta = np.sin(2 * np.pi * (x - c * steps * dt) / length) * np.sinc(1 / n)
        exact = np.stack((exact_eta, c * exact_eta), axis=1)
        cases[f'linear_{n}'] = (initial, dx, dt, steps, 'linear', depth, exact)
    for n in (100, 200, 400):
        dx, steps = 200. / n, n // 4
        dt = 0.25 * dx / np.sqrt(G * 4.)
        x = -100 + (np.arange(n) + 0.5) * dx
        initial = np.stack((np.where(x < 0, 4., 1.), np.zeros(n)), axis=1)
        exact, hs, shock = dam_exact(x, dt * steps, 4., 1.)
        cases[f'dam_{n}'] = (initial, dx, dt, steps, 'nonlinear', 0., exact)
    for name, (initial, dx, dt, steps, mode, depth, exact) in cases.items():
        initial = initial.astype(np.float32).astype(np.float64)
        result = advance(initial, dx, dt, steps, mode, depth)
        np.savez_compressed(OUT / f'{name}.npz', initial=initial, reference=result, analytic=exact,
                            dx=dx, dt=dt, steps=steps, depth=depth)
        mask = np.abs(-100 + (np.arange(len(initial)) + 0.5) * dx) < 60 if mode == 'nonlinear' else np.ones(len(initial), bool)
        error = np.mean(np.abs(result[mask, 0] - exact[mask, 0]))
        print(name, 'range', result.min(axis=0), result.max(axis=0), 'L1 height error', error)
    print('dam star depth', hs, 'shock speed', shock, 'linear speed', np.sqrt(G * 10))
    for mode in ('linear', 'nonlinear'):
        n, dx, dt, steps, depth = 16, 2., 0.02, 4, 3.
        if mode == 'linear':
            initial = rng.normal(0, 0.3, (n, 2))
        else:
            initial = np.column_stack((3 + rng.uniform(-0.3, 0.3, n), rng.uniform(0.2, 0.5, n)))
        initial = initial.astype(np.float32).astype(np.float64)
        weights = rng.normal(0, 0.2, (n, 2)).astype(np.float32).astype(np.float64)
        def objective(v, dx=dx, dt=dt, steps=steps, mode=mode, depth=depth, weights=weights):
            return np.sum(advance(v, dx, dt, steps, mode, depth) * weights)
        eps = 1e-5
        grad = np.empty_like(initial)
        for i in range(n):
            for j in range(2):
                plus, minus = initial.copy(), initial.copy()
                plus[i, j] += eps
                minus[i, j] -= eps
                grad[i, j] = (objective(plus) - objective(minus)) / (2 * eps)
        np.savez_compressed(OUT / f'grad_{mode}.npz', initial=initial, weights=weights,
                            reference=advance(initial, dx, dt, steps, mode, depth),
                            objective=objective(initial), gradient=grad, dx=dx, dt=dt, steps=steps, depth=depth)
    # Small benchmark geometry, time horizon, and seeded perturbations.
    for depth in (1000., 4000.):
        n, dx, steps = 65536, 100., 80
        dt = 0.25 * dx / np.sqrt(G * (depth + 3.))
        x = np.arange(n, dtype=np.float64) / n
        deep_rng = np.random.default_rng(4901)
        eta = 2 * np.sin(2 * np.pi * x) + deep_rng.uniform(-0.05, 0.05, n)
        q = 10 * np.cos(2 * np.pi * x) + deep_rng.uniform(-0.05, 0.05, n)
        for mode in ('linear', 'nonlinear'):
            initial = np.column_stack((eta if mode == 'linear' else depth + eta, q))
            initial = initial.astype(np.float32).astype(np.float64)
            result = advance(initial, dx, dt, steps, mode, depth)
            check = advance_float32_perturbation(initial, dx, dt, steps, mode, depth)
            scales = np.max(np.abs(result), axis=0)
            scales[0] = np.max(np.abs(result[:, 0] - (depth if mode == 'nonlinear' else 0)))
            errors = np.max(np.abs(check - result), axis=0) / scales
            print('deep', mode, depth, 'dt', dt, 'ranges', result.min(axis=0),
                  result.max(axis=0), 'float32 scaled errors', errors)
            if np.any(errors > 1e-4):
                raise RuntimeError('float32 perturbation sanity check exceeds deep tolerance')
            np.savez_compressed(OUT / f'deep_{mode}_{int(depth)}.npz', initial=initial,
                                reference=result, dx=dx, dt=dt, steps=steps, depth=depth)
    meta = {'seed': 4901, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
            'scheme': 'periodic finite-volume Rusanov, forward Euler, positive wet depths; SciPy correlate1d flux divergence',
            'analytic': 'linear right-going sinusoid and exact wet-bed Riemann rarefaction/shock; g=9.81 m/s^2',
            'inputs': 'Each npz stores initial state, dx, dt, steps, depth; float32-quantized inputs, float64 references. Dam at x=0; linear domain [0,1000), dam domain [-100,100).',
            'tolerances': {'parity': '1e-4 max error normalized by max reference magnitude per component, with scale floor 1e-30; float32 rounding', 'gradient': '1e-3 normalized max error, float32 autodiff/adjoint vs centered float64 FD, step 1e-5', 'objective': '1e-4 max(1, abs(reference)), to handle cancellation', 'linear_order': '>=0.85, first-order upwind/Euler', 'dam_order': '>=0.45, shock and rarefaction numerical diffusion yield sub-first-order L1 convergence', 'wave_speed': '<=1%, project requirement', 'mass': '1e-5 relative to initial total, float32 conservation', 'cross_path': '1e-4 normalized max error, float32 accumulation'}}
    meta['deep_cases'] = {'seed': 4901, 'n': 65536, 'dx_m': 100., 'steps': 80,
                          'depths_m': [1000., 4000.], 'dt_s': '0.25*dx/sqrt(9.81*(depth+3))',
                          'initial': 'benchmark sinusoid plus uniform noise; linear eta or nonlinear total depth and identical q; inputs quantized to float32',
                          'tolerance': '1e-4 max error divided by peak reference surface perturbation or discharge; verified with float32 perturbation evolution, with output depth rounded once',
                          'cross_path_tolerance': '2e-4 of peak reference surface perturbation or discharge, triangle bound for two independently rounded float32 results',
                          'method': 'avoid hydrostatic pressure cancellation and repeated rounding of total depth, e.g. evolve eta=h-H and subtract constant g*H^2/2 from momentum flux'}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
