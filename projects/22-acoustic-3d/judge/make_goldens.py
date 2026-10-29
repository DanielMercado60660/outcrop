"""Generate Deepwave float64 parity and analytic periodic-wave fixtures."""
import json
import platform
import runpy
import warnings
from pathlib import Path

import deepwave
import numpy as np
import scipy
import torch

OUT = Path(__file__).resolve().parent / 'golden'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
tol = runpy.run_path(str(OUT.parent.parent / 'tests' / 'tolerances.py'))

def propagate(v, u, prev, h, dt, steps):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = deepwave.scalar(torch.from_numpy(v), h, dt,
            wavefield_0=torch.from_numpy(u[None]), wavefield_m1=torch.from_numpy(prev[None]),
            accuracy=2, pml_width=0, boundary='periodic', nt=steps,
            max_vel=float(np.max(v)))
    return result[0].detach().numpy()[0], result[1].detach().numpy()[0]

def save(name, **data):
    np.savez_compressed(OUT / (name + '.npz'), **data)

rng = np.random.default_rng(22001)
shape = (13, 15, 17)
v = rng.uniform(0.8, 1.2, shape)
u = rng.normal(0, 0.15, shape)
prev = rng.normal(0, 0.15, shape)
h, dt, steps = 0.1, 0.02, 11
final, last = propagate(v, u, prev, h, dt, steps)
print('heterogeneous final range:', float(final.min()), float(final.max()))
save('heterogeneous', velocity=v, initial=u, previous=prev, final=final, last=last,
     spacing=h, dt=dt, steps=steps)

shape2 = (15, 17)
v2 = rng.uniform(0.8, 1.2, shape2)
u2 = rng.normal(0, 0.15, shape2)
p2 = rng.normal(0, 0.15, shape2)
f2, l2 = propagate(v2, u2, p2, h, dt, steps)
v3 = np.repeat(v2[None], 7, axis=0)
u3 = np.repeat(u2[None], 7, axis=0)
p3 = np.repeat(p2[None], 7, axis=0)
f3, l3 = propagate(v3, u3, p3, h, dt, steps)
assert np.max(np.abs(f3 - f2[None])) < 1e-12
assert np.max(np.abs(l3 - l2[None])) < 1e-12
save('invariant', velocity=v3, initial=u3, previous=p3, final_2d=f2, last_2d=l2,
     spacing=h, dt=dt, steps=steps)

errors = []
for n in (12, 24, 48):
    x = np.arange(n, dtype=np.float64) / n
    phase = 2*np.pi*(x[:, None, None] + x[None, :, None] + x[None, None, :])
    u = np.cos(phase)
    dt = 0.15/n
    omega = 2*np.pi*np.sqrt(3)
    prev = u*np.cos(omega*dt)
    exact = u*np.cos(omega*0.15)
    v = np.ones((n,n,n), dtype=np.float64)
    final, last = propagate(v, u, prev, 1/n, dt, n)
    err = np.max(np.abs(final-exact))
    errors.append(float(err))
    save(f'plane_{n}', velocity=v, initial=u, previous=prev, exact=exact,
         final=final, last=last, spacing=1/n, dt=dt, steps=n)
    print(f'plane n={n}: error={err:.9g}, range=[{final.min():.6g}, {final.max():.6g}]')
orders = np.log2(np.array(errors[:-1])/errors[1:])
print('observed orders:', orders)
print('2D max:', np.max(np.abs(f2)))
assert np.all((orders > tol['ORDER_MIN']) & (orders < tol['ORDER_MAX']))
meta = {
 'seed': 22001, 'libraries': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__, 'deepwave': deepwave.__version__},
 'equation': 'u_tt = v(x)^2 Laplacian(u); second order centered space and time; periodic faces',
 'inputs': {'heterogeneous': {'shape': shape, 'v_range': [0.8,1.2], 'spacing': h, 'dt': 0.02, 'steps': steps}, 'invariant': {'shape': list(v3.shape), 'independent_2d_shape': shape2}, 'plane': {'n': [12,24,48], 'domain_lengths': [1,1,1], 'velocity': 1, 'final_time': 0.15, 'dt_over_spacing': 0.15}},
 'tolerances': {'parity': {'value': tol['PARITY_RTOL'], 'reason': 'float32 accumulated error against float64 Deepwave, normalized by maximum reference amplitude with an initial-amplitude floor for plane waves near a zero crossing'}, 'cross_path': {'value': tol['CROSS_PATH_RTOL'], 'reason': 'float32 implementation parity, normalized by maximum reference amplitude with the same plane-wave initial-amplitude floor'}, 'order_min': {'value': tol['ORDER_MIN'], 'reason': 'second order central stencil and time update, allowing 0.2 pre-asymptotic deviation'}, 'order_max': {'value': tol['ORDER_MAX'], 'reason': 'expected second order within 0.2'}, 'analytic_error_max': {'value': tol['ANALYTIC_ERROR_MAX'], 'reason': 'bounds coarse-grid phase error and prevents convergence to a wrong solution'}},
 'analytic_max_errors': errors, 'analytic_observed_orders': orders.tolist(),
 'reference_notes': 'Deepwave scalar float64, accuracy=2, no PML, explicitly periodic on all faces, one shot. 2D fixture is independently propagated using Deepwave; project 13 is not yet available.'}
(OUT/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
