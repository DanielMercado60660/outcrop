"""Matched-stencil parity and analytic acoustic-wave checks."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import CROSS_PATH, ORDER_DEVIATION, PARITY, REFLECTION

CASES = ['convergence_101', 'convergence_201', 'convergence_401',
         'reflection_-1', 'reflection_+1']

def run(path, data):
    result = call(path, 'propagate', u0=data['u0'], u_prev=data['u_prev'],
                  c=float(data['c']), dx=float(data['dx']), dt=float(data['dt']),
                  steps=int(data['steps']))
    assert set(result) == {'wavefield'}
    wave = result['wavefield']
    assert wave.shape == data['expected'].shape
    assert wave.dtype == np.float32
    assert np.isfinite(wave).all()
    return wave

@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', CASES)
def test_matched_stencil(path, case):
    data = golden(case)
    assert rel_err(run(path, data), data['expected']) <= PARITY

@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_second_order_convergence(path):
    errors = []
    for n in (101, 201, 401):
        data = golden(f'convergence_{n}')
        final = run(path, data)[-1].astype(np.float64)
        errors.append(np.linalg.norm(final - data['exact']) / np.linalg.norm(data['exact']))
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all(np.abs(orders - 2.0) <= ORDER_DEVIATION), (errors, orders)

@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('direction', [-1, 1], ids=['left', 'right'])
def test_absorbing_boundary(path, direction):
    data = golden(f'reflection_{direction:+d}')
    wave = run(path, data)
    # The pulse has exited by the final time; normalize by the incident peak.
    residual = np.max(np.abs(wave[-1])) / np.max(np.abs(data['u0']))
    assert residual < REFLECTION, residual

@pytest.mark.parametrize('case', CASES)
def test_cross_path_mlx_metal(case):
    data = golden(case)
    a, b = run('mlx', data), run('metal', data)
    assert rel_err(a, b) <= CROSS_PATH
