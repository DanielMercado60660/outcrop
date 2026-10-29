"""Analytic hyperbolic arrivals and independent float64 CMP reference."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'golden'
OUT.mkdir(exist_ok=True)
dt = 0.002
nt = 751
t = np.arange(nt, dtype=np.float64) * dt
x = np.linspace(0, 1500, 31, dtype=np.float64)
times = np.array([0.35, 0.72, 1.08])
truth = np.array([1800., 2400., 3000.])
gather = np.zeros((x.size, nt), dtype=np.float64)
for t0, v, amp in zip(times, truth, [1., -0.8, 0.7]):
    arrival = np.sqrt(t0*t0 + (x/v)**2)
    u = np.pi*25*(t[None, :] - arrival[:, None])
    gather += amp*(1-2*u*u)*np.exp(-u*u)
# Quantize inputs first, then calculate the reference from exactly those inputs.
gather = gather.astype(np.float32).astype(np.float64)
x = x.astype(np.float32).astype(np.float64)
velocities = np.arange(1500, 3301, 25, dtype=np.float64)


def correct(v):
    query = np.sqrt(t[None, :]**2 + (x[:, None]/v)**2)
    return np.stack([np.interp(q, t, trace, left=0., right=0.)
                     for q, trace in zip(query, gather)])


half = 5
s = []
energy_mask = []
for v in velocities:
    a = correct(v)
    numerator = np.sum(a, axis=0)**2
    denominator = x.size*np.sum(a*a, axis=0)
    num = np.array([numerator[max(0, j-half):min(nt, j+half+1)].sum() for j in range(nt)])
    den = np.array([denominator[max(0, j-half):min(nt, j+half+1)].sum() for j in range(nt)])
    s.append(np.divide(num, den, out=np.zeros_like(num), where=den != 0))
    energy_mask.append(den > 0.01)
s = np.asarray(s)
velocity = 2400.
c = correct(velocity)
np.savez_compressed(OUT/'cmp.npz', gather=gather, offsets=x, dt=np.array(dt), velocities=velocities,
                    half_window=np.array(half), velocity=np.array(velocity), corrected=c,
                    stack=c.mean(axis=0), semblance=s, energy_mask=np.asarray(energy_mask), event_times=times, truth=truth)
indices = np.rint(times/dt).astype(int)
picks = velocities[np.argmax(s[:, indices], axis=0)]
assert np.all(np.abs(picks/truth-1) <= .03), picks
assert np.max(np.abs(c)) <= 1.01 and np.all((s >= 0) & (s <= 1+1e-12))
meta = {'reference': 'Analytic hyperbolic arrivals and Ricker wavelets; NumPy float64 interpolation and explicit semblance sums',
        'numpy_version': np.__version__, 'seed': 'no randomness', 'dt_seconds': dt,
        'samples': nt, 'offsets_m': [float(x[0]), float(x[-1]), len(x)],
        'event_times_s': times.tolist(), 'true_velocities_m_per_s': truth.tolist(),
        'picked_velocities_m_per_s': picks.tolist(), 'trial_velocity_m_per_s': [1500, 3300, 25],
        'half_window_samples': half, 'ricker_frequency_hz': 25,
        'semblance_comparison_mask': 'Reference window denominator greater than 0.01 and time greater than 0.2 s; silent windows are ill-conditioned'}
print('picks m/s:', picks, 'semblance at picks:', s[np.argmax(s[:, indices], axis=0), indices])
print('gather range:', gather.min(), gather.max(), 'stack range:', c.mean(axis=0).min(), c.mean(axis=0).max())

# A nonzero final sample exercises zero fill and paired signed offsets.
edge_dt = 0.25
edge_t = np.arange(6, dtype=np.float64) * edge_dt
edge_x = np.array([-1., 0., 1., 3.], dtype=np.float64)
edge_trace = np.array([0.125, 0.25, -0.5, 0.75, -0.25, 1.], dtype=np.float64)
edge_gather = np.tile(edge_trace, (edge_x.size, 1))
edge_velocity = 2.0
edge_q = np.sqrt(edge_t[None, :]**2 + (edge_x[:, None]/edge_velocity)**2)
edge_corrected = np.stack([np.interp(q, edge_t, trace, left=0., right=0.)
                           for q, trace in zip(edge_q, edge_gather)])
assert edge_corrected[1, -1] == 1.0
assert edge_corrected[0, -1] == 0.0
assert np.array_equal(edge_corrected[0], edge_corrected[2])
np.savez_compressed(OUT/'edge.npz', gather=edge_gather, offsets=edge_x,
                    dt=np.array(edge_dt), velocity=np.array(edge_velocity),
                    corrected=edge_corrected, stack=edge_corrected.mean(axis=0))
np.savez_compressed(OUT/'zero.npz', gather=np.zeros_like(gather), offsets=x,
                    dt=np.array(dt), velocity=np.array(velocity), velocities=velocities,
                    half_window=np.array(half))
meta['boundary_fixture'] = {'dt_seconds': edge_dt, 'offsets_m': edge_x.tolist(),
                            'velocity_m_per_s': edge_velocity,
                            'trace': edge_trace.tolist(), 'last_sample_nonzero': True}
meta['tolerances'] = {
    'array_relative': [1e-4, 'float32 arithmetic versus float64'],
    'nmo_index_ulp_factor': [4, 'float32 sample-index ulps times maximum trace slope, allowing moveout and interpolation rounding'],
    'cross_index_ulp_factor': [8, 'sum of two independent four-ulp index error budgets'],
    'semblance_absolute': [1e-5, 'float32 reductions in energetic windows, validated against simulations'],
    'cross_semblance_absolute': [2e-5, 'sum of two independent semblance error budgets'],
    'velocity_relative': [0.03, 'project velocity criterion; velocity grid spacing 25 m/s']}
(OUT/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
print('boundary endpoint:', edge_corrected[1, -1], 'out of range:', edge_corrected[0, -1])
