"""Locked float32 parity, discrete-adjoint and recovery tests (both paths)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, golden_meta, rel_err
from tolerances import (
    BACKTRACKS,
    EQUATION_AUDIT_REL,
    FD_REL,
    FD_STEP,
    GRADIENT_COSINE,
    ITERATIONS,
    MAX_GOLDEN_BYTES,
    MODEL_ERROR_RATIO,
    PARITY,
    UPDATE_MAX,
    VELOCITY_MAX,
    VELOCITY_MIN,
)

CASES = ['initial', 'perturbed']


def arguments(g):
    return {**{k:g[k] for k in ['velocity', 'source_amplitudes', 'source_locations',
                               'receiver_locations', 'observed']},
            **golden_meta()['settings']}


def checked(path, kwargs):
    out = call(path, 'misfit_and_grad', **kwargs)
    assert set(out) == {'predicted', 'objective', 'gradient'}
    for key, shape in [('predicted', kwargs['observed'].shape),
                       ('objective', ()), ('gradient', kwargs['velocity'].shape)]:
        assert out[key].shape == shape, key
        assert out[key].dtype == np.float32, key
        assert np.isfinite(out[key]).all(), key
    assert out['objective'] >= 0
    # Ensure the reported objective belongs to the returned traces.
    residual = out['predicted'].astype(np.float64)-kwargs['observed']
    assert rel_err(out['objective'], 0.5*np.mean(residual**2)) <= PARITY
    return out


def relative_scalar(a, b):
    return abs(a-b)/max(abs(a), abs(b))


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', CASES)
def test_deepwave_parity(path, case):
    g = golden(case)
    out = checked(path, arguments(g))
    for key in ['predicted', 'objective', 'gradient']:
        assert rel_err(out[key], g[key]) <= PARITY, key
    a, b = out['gradient'].astype(np.float64).ravel(), g['gradient'].ravel()
    cosine = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    assert cosine >= GRADIENT_COSINE


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('direction', range(3), ids=['global', 'random', 'lens'])
def test_centered_finite_difference(path, case, direction):
    g = golden(case)
    kwargs = arguments(g)
    base = checked(path, kwargs)
    d = g['directions'][direction]
    plus = (g['velocity']+FD_STEP*d).astype(np.float32)
    minus = (g['velocity']-FD_STEP*d).astype(np.float32)
    fp = checked(path, {**kwargs, 'velocity':plus})['objective'].item()
    fm = checked(path, {**kwargs, 'velocity':minus})['objective'].item()
    fd = (fp-fm)/(2*FD_STEP)
    effective = (plus.astype(np.float64)-minus)/(2*FD_STEP)
    adjoint = float(np.sum(base['gradient'].astype(np.float64)*effective))
    assert relative_scalar(adjoint, fd) <= FD_REL
    # An implementation cannot pass by returning an internally consistent,
    # physically different derivative. Compare with independent f64 oracle FD.
    assert relative_scalar(adjoint, float(g['fd'][direction])) <= FD_REL
    assert relative_scalar(fd, float(g['fd'][direction])) <= FD_REL


@pytest.mark.parametrize('case', CASES)
def test_mlx_metal_crosscheck(case):
    kwargs = arguments(golden(case))
    a, b = checked('mlx', kwargs), checked('metal', kwargs)
    for key in a:
        assert rel_err(a[key], b[key]) <= PARITY, key


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_fixed_iteration_recovery(path):
    g = golden('initial')
    kwargs = arguments(g)
    velocity = g['velocity'].copy()
    true = g['true_velocity'].astype(np.float64)
    initial_error = np.linalg.norm(velocity.astype(np.float64)-true)/np.linalg.norm(true)
    initial_loss = None
    for _ in range(ITERATIONS):
        out = checked(path, {**kwargs, 'velocity':velocity})
        if initial_loss is None:
            initial_loss = out['objective'].item()
        grad = out['gradient'].astype(np.float64)
        scale = np.max(np.abs(grad))
        assert scale > 0, 'stalled gradient before the fixed iteration count'
        for k in range(BACKTRACKS):
            candidate = np.clip(velocity-UPDATE_MAX*2.0**(-k)*grad/scale,
                                VELOCITY_MIN, VELOCITY_MAX).astype(np.float32)
            trial = checked(path, {**kwargs, 'velocity':candidate})
            if trial['objective'] < out['objective']:
                velocity = candidate
                break
    final = checked(path, {**kwargs, 'velocity':velocity})
    error = np.linalg.norm(velocity.astype(np.float64)-true)/np.linalg.norm(true)
    assert error/initial_error <= MODEL_ERROR_RATIO
    assert final['objective'] < initial_loss


def test_golden_integrity_and_reference_recovery():
    meta = golden_meta()
    folder = Path(__file__).resolve().parents[1]/'judge/golden'
    for name, digest in meta['sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    spec = folder.parents[1]/meta['discrete_operator']['spec']
    assert hashlib.sha256(spec.read_bytes()).hexdigest() == meta['discrete_operator']['spec_sha256']
    audit = folder.parents[1]/'judge/discrete_operator.py'
    assert hashlib.sha256(audit.read_bytes()).hexdigest() == meta['discrete_operator']['audit_sha256']
    for case in CASES:
        summary = meta['summaries'][case]
        assert summary['written_equation_trace_relative_error'] <= EQUATION_AUDIT_REL
        assert max(summary['written_equation_fd_relative_errors']) <= FD_REL
    r = golden('recovery')
    assert len(r['model_errors']) == ITERATIONS+1
    assert r['model_errors'][-1]/r['model_errors'][0] <= MODEL_ERROR_RATIO
    assert np.all(np.diff(r['objectives']) <= 0)
    assert sum(p.stat().st_size for p in folder.iterdir()) < MAX_GOLDEN_BYTES
