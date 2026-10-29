"""Generate deterministic Deepwave parity and Green-function cases."""
import json
import sys
from pathlib import Path

import deepwave
import numpy as np
import scipy
import torch
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'tests'))
import tolerances as tol

OUT = ROOT / 'golden'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)


def run(v, source, sl, rl, dx, dt, order, pml, freq, vmax):
    with torch.no_grad():
        out = deepwave.scalar(torch.from_numpy(v.astype(np.float64)), dx, dt,
            source_amplitudes=torch.from_numpy(source.astype(np.float64)),
            source_locations=torch.from_numpy(sl), receiver_locations=torch.from_numpy(rl),
            accuracy=order, pml_width=pml, pml_freq=freq, max_vel=vmax)
    return out.receiver_amplitudes.numpy()


def pulse(t):
    return np.exp(-0.5*((t - 0.15) / 0.025)**2)


def green(t, r, velocity):
    # For (c^-2 d_tt - Laplacian)u=-f delta(x), the substitution
    # tau=t-(r/c)cosh(s) removes the Green kernel's square-root singularity.
    a = r / velocity
    if t <= a:
        return 0.0
    hi = np.arccosh(t / a)
    return -quad(lambda s: pulse(t - a*np.cosh(s)), 0, hi,
                 epsabs=1e-13, epsrel=1e-12)[0] / (2*np.pi)


def main():
    for old in OUT.glob('conv_*.npz'):
        old.unlink()
    rng = np.random.default_rng(1301)
    z, x = np.indices((37, 43))
    v = (1550 + 120*np.sin(z/7)*np.cos(x/8) + 180*(z > 21)
         + rng.normal(0, 3, z.shape)).astype(np.float32)
    dt, dx, nt = 0.001, 10.0, 300
    t = np.arange(nt)*dt
    w = ((1-2*(np.pi*18*(t-0.05))**2)*np.exp(-(np.pi*18*(t-0.05))**2)).astype(np.float32)
    source = np.stack([w, -0.7*w])[:, None, :] * np.float32(1e-5)
    sl = np.array([[[10, 12]], [[26, 30]]], dtype=np.int64)
    rl = np.array([[[10, 12], [12, 25], [30, 36]],
                   [[26, 30], [20, 15], [5, 6]]], dtype=np.int64)
    pml = [9, 12, 10, 11]
    for order in (2, 4, 6, 8):
        trace = run(v, source, sl, rl, dx, dt, order, pml, 18.0, 1900.0)
        np.savez_compressed(OUT / f'parity_{order}.npz', velocity=v, source=source,
            source_locations=sl, receiver_locations=rl, expected=trace)
        print('parity', order, 'range', trace.min(), trace.max(), flush=True)
    for order in (2, 4, 6, 8):
        errors = []
        for h in (20, 10, 5):
            dt = .4*h/1500
            nt = round(.60/dt) + 1
            time = np.arange(nt)*dt
            analytic = np.array([green(t, 240., 1500.) for t in time])
            n = 1600//h + 1
            vel = np.full((n, n), 1500, dtype=np.float32)
            src = (np.array([pulse(t) for t in time], np.float32)/np.float32(h*h))[None, None, :]
            loc = np.array([[[800//h, 800//h]]], np.int64)
            rec = np.array([[[800//h, 1040//h]]], np.int64)
            result = run(vel, src, loc, rec, float(h), dt, order, [10]*4, 5., 1500.)
            np.savez_compressed(OUT / f'conv_{order}_{h}.npz', velocity=vel, source=src,
                source_locations=loc, receiver_locations=rec, expected=result, analytic=analytic)
            error = np.linalg.norm(result[0, 0] - analytic)/np.linalg.norm(analytic)
            errors.append(error)
        print('convergence', order, errors, 'rates', np.log2(np.array(errors[:-1])/errors[1:]), flush=True)
    info = {'seed': 1301, 'versions': {'numpy': np.__version__, 'scipy': scipy.__version__,
            'torch': torch.__version__, 'deepwave': deepwave.__version__},
        'parity': {'shape': [37,43], 'spacing_m': dx, 'dt_s': .001, 'nt': 300,
            'pml_width': pml, 'pml_freq_hz': 18., 'max_vel_m_s': 1900.,
            'description': 'heterogeneous two-shot model, random seed 1301, asymmetric PML'},
        'convergence': {'spacing_m': [20,10,5], 'dt_s': [0.4*h/1500 for h in (20,10,5)], 'velocity_m_s': 1500,
            'offset_m': 240, 'domain_width_m': 1600, 'pml_width': [10]*4,
            'pml_freq_hz': 5., 'description': 'Gaussian pulse centered at 0.15 s, sigma 0.025 s, point source scaled by inverse cell area; causal 2D Green convolution'},
        'tolerances': {name: {'value': value, 'reason': reason} for name, value, reason in tol.METADATA}}
    (OUT/'meta.json').write_text(json.dumps(info, indent=2)+'\n')


if __name__ == '__main__':
    main()
