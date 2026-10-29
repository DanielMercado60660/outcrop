"""Offline float64 Deepwave oracle. Run from this project directory."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import runpy
import time
from pathlib import Path

import deepwave
import numpy as np
import torch
from discrete_operator import predict, profiles

ROOT = Path(__file__).resolve().parents[1]
T = runpy.run_path(str(ROOT / "tests/tolerances.py"))
OUT = ROOT / "judge/golden"
SETTINGS = {'grid_spacing': 10.0, 'dt': 0.001, 'accuracy': 4, 'pml_width': 10,
            'pml_freq': 15.0, 'max_vel': 2600.0}


def evaluate(velocity, sources, sl, rl, observed, dtype=torch.float64):
    v = torch.tensor(velocity, dtype=dtype, requires_grad=True)
    result = deepwave.scalar(
        v, **SETTINGS, source_amplitudes=torch.tensor(sources, dtype=dtype),
        source_locations=torch.tensor(sl), receiver_locations=torch.tensor(rl),
        model_gradient_sampling_interval=1,
    )
    pred = result[-1]
    loss = 0.5 * (pred - torch.tensor(observed, dtype=dtype)).square().mean()
    loss.backward()
    return {'predicted': pred.detach().numpy(), 'objective': loss.detach().numpy(),
            'gradient': v.grad.numpy()}


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(1601)
    nz, nx, nt, shots = 24, 32, 360, 4
    z, x = np.mgrid[:nz, :nx]
    # Synthetic stand-in only: tilted layers, a fault and a smooth lens.
    background = 1800 + 7*z + 1.5*x
    true = background + 70*np.tanh((z-11-0.12*x-2*(x>18))/2)
    true += 75*np.exp(-((z-16)**2/24+(x-12)**2/50))
    initial = background - 75
    true, initial = true.astype(np.float32), initial.astype(np.float32)
    time_axis = np.arange(nt)*SETTINGS['dt']
    a = np.pi*15*(time_axis-0.08)
    wavelet = ((1-2*a*a)*np.exp(-a*a)).astype(np.float32)
    sources = np.broadcast_to(wavelet, (shots, 1, nt)).copy()
    sl = np.array([[[2, i]] for i in [4, 12, 20, 27]], dtype=np.int64)
    receivers = np.array([[j, i] for j in [2, nz-3] for i in range(2, nx-2, 2)])
    rl = np.broadcast_to(receivers, (shots, *receivers.shape)).copy().astype(np.int64)
    zeros = np.zeros((shots, len(receivers), nt))
    observed64 = evaluate(true, sources, sl, rl, zeros)['predicted']
    assert np.isfinite(observed64).all()
    assert np.all(observed64[..., 0] == 0), 'receiver timing convention changed'
    assert np.max(np.abs(observed64)) > 0
    assert np.all((true >= T['VELOCITY_MIN']) & (true <= T['VELOCITY_MAX']))
    # Save exact float32 API inputs; goldens are evaluated at those inputs in f64.
    observed = observed64.astype(np.float32)
    common = {'source_amplitudes': sources, 'source_locations': sl,
              'receiver_locations': rl, 'observed': observed, 'true_velocity': true}
    summaries = {}
    for name, v in [('initial', initial),
                    ('perturbed', (initial + 18*np.sin(x/4)*np.cos(z/5)).astype(np.float32))]:
        out = evaluate(v, sources, sl, rl, observed)
        # Independent evaluation of the WRITTEN equations, not a second call
        # through Deepwave's setup or Python/native stepping routines.
        written = predict(v, sources, sl, rl, SETTINGS)
        equation_error = float(np.max(np.abs(written-out['predicted'])) /
                               np.max(np.abs(out['predicted'])))
        assert equation_error <= T['EQUATION_AUDIT_REL'], equation_error
        f32 = evaluate(v, sources, sl, rl, observed, torch.float32)
        parity32 = {key:float(np.max(np.abs(f32[key].astype(float)-out[key])) /
                             np.max(np.abs(out[key]))) for key in out}
        assert max(parity32.values()) <= T['PARITY'], parity32
        directions = np.stack([np.ones_like(v), rng.normal(size=v.shape),
                               np.exp(-((z-12)**2+(x-16)**2)/30)])
        directions /= np.sqrt(np.mean(directions**2, axis=(1, 2)))[:, None, None]
        directions = directions.astype(np.float32)
        fd, actual_steps, fd32, equation_fd = [], [], [], []
        for d in directions:
            plus = (v + T['FD_STEP']*d).astype(np.float32)
            minus = (v - T['FD_STEP']*d).astype(np.float32)
            # Effective direction includes float32 input quantization.
            actual_steps.append((plus.astype(float)-minus)/(2*T['FD_STEP']))
            fp = evaluate(plus, sources, sl, rl, observed)['objective']
            fm = evaluate(minus, sources, sl, rl, observed)['objective']
            fd.append((fp-fm)/(2*T['FD_STEP']))
            pred_plus = predict(plus, sources, sl, rl, SETTINGS)
            pred_minus = predict(minus, sources, sl, rl, SETTINGS)
            equation_fd.append((0.5*np.mean((pred_plus-observed)**2)
                                - 0.5*np.mean((pred_minus-observed)**2)) /
                               (2*T['FD_STEP']))
            fp32 = evaluate(plus, sources, sl, rl, observed, torch.float32)['objective'].item()
            fm32 = evaluate(minus, sources, sl, rl, observed, torch.float32)['objective'].item()
            fd32.append((fp32-fm32)/(2*T['FD_STEP']))
        adj = np.einsum('ij,kij->k', out['gradient'], actual_steps)
        fd = np.array(fd)
        fd_error = np.abs(adj-fd)/np.maximum(np.abs(adj), np.abs(fd))
        assert np.max(fd_error) <= T['FD_REL'], fd_error
        equation_fd_error = np.abs(adj-equation_fd)/np.maximum(np.abs(adj), np.abs(equation_fd))
        assert np.max(equation_fd_error) <= T['FD_REL'], equation_fd_error
        adj32 = np.einsum('ij,kij->k', f32['gradient'].astype(float), actual_steps)
        fd32_error = np.abs(adj32-fd32)/np.maximum(np.abs(adj32), np.abs(fd32))
        assert np.max(fd32_error) <= T['FD_REL'], fd32_error
        np.savez_compressed(OUT / f'{name}.npz', velocity=v, **common, **out,
                            directions=directions, fd=fd, effective_directions=actual_steps)
        summaries[name] = {'objective': float(out['objective']),
            'gradient_norm': float(np.linalg.norm(out['gradient'])),
            'predicted_absmax': float(np.max(np.abs(out['predicted']))),
            'fd_relative_errors': fd_error.tolist(), 'fd': fd.tolist(),
            'float32_reference_parity': parity32, 'float32_fd_errors': fd32_error.tolist(),
            'written_equation_trace_relative_error': equation_error,
            'written_equation_fd': equation_fd,
            'written_equation_fd_relative_errors': equation_fd_error.tolist()}

    # Exactly the optimizer that tests replay; no truth-dependent updates.
    v = initial.copy()
    losses, errors = [], []
    initial_error = np.linalg.norm(v.astype(float)-true)/np.linalg.norm(true)
    for iteration in range(T['ITERATIONS']):
        out = evaluate(v, sources, sl, rl, observed)
        losses.append(float(out['objective']))
        errors.append(float(np.linalg.norm(v.astype(float)-true)/np.linalg.norm(true)))
        g = out['gradient']
        for k in range(T['BACKTRACKS']):
            candidate = np.clip(v-T['UPDATE_MAX']*2.0**(-k)*g/np.max(np.abs(g)),
                                T['VELOCITY_MIN'], T['VELOCITY_MAX']).astype(np.float32)
            trial = evaluate(candidate, sources, sl, rl, observed)
            if trial['objective'] < out['objective']:
                v = candidate
                break
        if iteration % 10 == 0:
            print(f'iteration {iteration}: loss={losses[-1]:.8g}, error={errors[-1]:.6g}', flush=True)
    final = evaluate(v, sources, sl, rl, observed)
    losses.append(float(final['objective']))
    errors.append(float(np.linalg.norm(v.astype(float)-true)/np.linalg.norm(true)))
    ratio = errors[-1]/initial_error
    assert ratio <= T['MODEL_ERROR_RATIO'], f'reference recovery failed: {ratio}'
    np.savez_compressed(OUT/'recovery.npz', final_velocity=v,
                        objectives=np.array(losses), model_errors=np.array(errors))
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'torch', 'deepwave', 'pytest']}
    meta = {'seed': 1601, 'data': 'SYNTHETIC STAND-IN; NOT Marmousi; no public model available',
        'reference': 'Deepwave scalar CPU float64, deterministic one thread', 'versions': versions,
        'settings': SETTINGS, 'shape': [nz,nx], 'shots': shots, 'receivers': len(receivers), 'nt': nt,
        'inputs': 'Float32 arrays saved in initial/perturbed.npz; locations int64; reference float64',
        'objective': '0.5 mean squared trace residual; gradient with respect to velocity in m/s',
        'fd_step_m_per_s': T['FD_STEP'], 'fd_directions': 'global, seeded random, smooth local lens; RMS-one',
        'tolerances': {k: {'value': T[k], 'reason': r} for k,r in T['REASONS'].items()},
        'optimizer': {k:T[k] for k in ['ITERATIONS','UPDATE_MAX','BACKTRACKS','VELOCITY_MIN','VELOCITY_MAX']},
        'golden_size_limit_bytes': T['MAX_GOLDEN_BYTES'],
        'summaries': summaries, 'velocity_range': [float(true.min()),float(true.max())],
        'recovery': {'initial_error': initial_error, 'final_error': errors[-1], 'ratio': ratio,
                     'initial_objective': losses[0], 'final_objective': losses[-1]},
        'runtime_seconds': time.monotonic()-start}
    meta['discrete_operator'] = {
        'version': 1, 'spec': 'tests/DISCRETE_OPERATOR.md',
        'spec_sha256': hashlib.sha256((ROOT/'tests/DISCRETE_OPERATOR.md').read_bytes()).hexdigest(),
        'audit': 'judge/discrete_operator.py: independent NumPy f64 recurrence and directional FD',
        'audit_sha256': hashlib.sha256((ROOT/'judge/discrete_operator.py').read_bytes()).hexdigest(),
        'target_reflection': 0.001, 'polynomial_power': 2, 'guard_cells': 2,
        'axis_total_lengths': [nz+2*(SETTINGS['pml_width']+2), nx+2*(SETTINGS['pml_width']+2)],
        'sigma0_per_s': float(-3*SETTINGS['max_vel']*np.log(0.001) /
                             (2*SETTINGS['pml_width']*SETTINGS['grid_spacing'])),
        'profile_sample_columns': ['a', 'b', 'db_per_m'],
        'axis_profile_samples': {label: [float(a[i]) for a in profiles(
            nz+2*(SETTINGS['pml_width']+2), SETTINGS['pml_width'], SETTINGS['grid_spacing'],
            SETTINGS['dt'], SETTINGS['pml_freq'], SETTINGS['max_vel'])]
            for label, i in [('outer_guard', 0), ('outer_active', 2), ('inner_pml', 11),
                             ('first_model_node', 12), ('second_model_node', 13)]}}
    meta['sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.npz'))}
    reference_root = Path(deepwave.__file__).parent
    meta['reference_sha256'] = {name:hashlib.sha256((reference_root/name).read_bytes()).hexdigest()
                                for name in ['scalar.py', '_common.py', '_regular_grid.py',
                                             'scalar_kernels.h', 'regular_grid.h', 'libdeepwave_C.dylib']}
    (OUT/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
