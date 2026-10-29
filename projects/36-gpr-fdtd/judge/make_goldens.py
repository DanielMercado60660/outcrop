"""Generate independent float64 Yee solutions using SciPy sparse operators."""
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse

C = 299792458.0
OUT = Path(__file__).parent / 'golden'


def operators(n, spacing):
    rows = np.arange(n)
    forward = sparse.coo_matrix((np.tile([-1., 1.], n),
        (np.repeat(rows, 2), np.column_stack((rows, (rows + 1) % n)).ravel())),
        shape=(n, n)).tocsr() / spacing
    return forward, -forward.T.tocsr()


def solve(eps, e, hx, hz, dx, dz, dt, steps):
    fx, bx = operators(e.shape[0], dx)
    fz, bz = operators(e.shape[1], dz)
    e, hx, hz = (np.asarray(a, dtype=np.float64).copy() for a in (e, hx, hz))
    for _ in range(steps):
        hx += C * dt * (fz @ e.T).T
        hz -= C * dt * (fx @ e)
        e += C * dt / eps * ((bz @ hx.T).T - bx @ hz)
    return e, hx, hz


def save(name, eps, e, hx, hz, dx, dz, dt, steps, **extra):
    # Inputs are quantized to the exact values presented to implementations.
    eps, e, hx, hz = [np.asarray(v, np.float32) for v in (eps, e, hx, hz)]
    result = solve(eps.astype(float), e, hx, hz, dx, dz, dt, steps)
    if name == 'reflection':
        # Translational symmetry makes Hz identically zero; sparse operations
        # can introduce float64 roundoff when evaluating identical rows.
        assert np.max(np.abs(result[2])) < 1e-12
        result[2].fill(0.0)
    np.savez_compressed(OUT / (name + '.npz'), epsilon_r=eps, electric=e,
        hx=hx, hz=hz, dx=dx, dz=dz, dt=dt, steps=steps,
        expected_electric=result[0], expected_hx=result[1], expected_hz=result[2], **extra)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3601)
    shape = (21, 19)
    eps = rng.uniform(1., 5., shape)
    fields = [rng.normal(0, .1, shape) for _ in range(3)]
    save('heterogeneous', eps, *fields, .08, .1, .3*.08/C, 45)
    # A diagonal plane wave has exact free-space dispersion omega=c|k|.
    for n in (24, 48, 96):
        dx = 1./n
        dt = .4*dx/C
        steps = n//2
        x, z = np.meshgrid(np.arange(n)*dx, np.arange(n)*dx, indexing='ij')
        k = 2*np.pi
        omega = C*k*np.sqrt(2.)
        e = np.sin(k*(x+z))
        hx = -np.sin(k*(x+z+dx/2)+omega*dt/2)/np.sqrt(2.)
        hz = np.sin(k*(x+z+dx/2)+omega*dt/2)/np.sqrt(2.)
        exact = np.sin(k*(x+z)-omega*steps*dt)
        save(f'wave_{n}', np.ones_like(e), e, hx, hz, dx, dx, dt, steps, analytic=exact)
    # Normal incidence on eps_r=4 at z=2 m; pulse is separated at t=2/c.
    dz = .005
    n = 800
    z = np.arange(n)*dz
    dt = .5*dz/C
    pulse = lambda q: np.exp(-((q-1.)/.15)**2)
    e = np.tile(pulse(z), (4, 1))
    hx = np.tile(-pulse(z+dz/2+C*dt/2), (4, 1))
    eps = np.tile(np.where(z >= 2., 4., 1.), (4, 1))
    save('reflection', eps, e, hx, np.zeros_like(e), dz, dz, dt, 800,
         reflection_coefficient=-1/3, reflection_start=150, reflection_stop=250)
    meta = {'seed': 3601, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'reference': 'float64 SciPy sparse Yee difference operators; analytic plane wave and Fresnel coefficient',
        'inputs': {'all': 'All input arrays and scalar numerical inputs are saved in each npz; SI units, periodic domain.',
            'heterogeneous': '21x19, random permittivity 1..5 and three random fields; 45 steps',
            'wave': '24, 48, 96 square grids over 1 m; diagonal one-cycle plane wave; CFL 0.4; n/2 steps',
            'reflection': '4x800 grid, 0.005 m spacing, 800 steps, Gaussian width 0.15 m at z=1 m, interface at z=2 m, relative permittivity 1 to 4'},
        'tolerances': {'parity': '1e-4 relative max norm, float32 rounding',
        'cross': '1e-4 relative max norm, float32 rounding',
        'zero_hz_absolute': '1e-6 V/m for reflection Hz parity and cross-check; exact symmetry gives zero, so relative error is undefined; allows float32 roundoff relative to a unit V/m pulse',
        'convergence': 'error ratio <= 0.30 per refinement, second order',
        'reflection': '2% relative amplitude, finite-grid interface error'}}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    for n in (24, 48, 96):
        with np.load(OUT / f'wave_{n}.npz') as d:
            print(n, 'analytic max error', np.max(np.abs(d['expected_electric']-d['analytic'])))
    with np.load(OUT / 'reflection.npz') as d:
        print('reflection amplitude', np.min(d['expected_electric'][0, 150:250]), 'analytic', -1/3)


if __name__ == '__main__':
    main()
