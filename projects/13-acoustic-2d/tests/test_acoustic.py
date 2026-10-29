"""Parity against Deepwave and combined space-time convergence tests."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import (
    CONVERGENCE_COARSE_MIN,
    CONVERGENCE_FINE_MIN,
    CONVERGENCE_MAX,
    CROSS_PATH,
    PARITY,
)

ORDERS = (2, 4, 6, 8)


def args(data, order, spacing, dt, pml, freq, vmax):
    return {'velocity': data['velocity'], 'source': data['source'],
            'source_locations': data['source_locations'],
            'receiver_locations': data['receiver_locations'], 'spacing': spacing,
            'dt': dt, 'order': order, 'pml_width': pml, 'pml_freq': freq, 'max_vel': vmax}


def run(path, data, **kwargs):
    result = call(path, 'scalar', **args(data, **kwargs))
    assert set(result) == {'receiver_amplitudes'}
    trace = result['receiver_amplitudes']
    assert trace.shape == data['expected'].shape
    assert trace.dtype == np.float32
    assert np.isfinite(trace).all()
    return trace


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('order', ORDERS)
def test_deepwave_parity(path, order):
    data = golden(f'parity_{order}')
    actual = run(path, data, order=order, spacing=10., dt=.001,
                 pml=[9, 12, 10, 11], freq=18., vmax=1900.)
    assert rel_err(actual, data['expected']) <= PARITY


@pytest.mark.parametrize('order', ORDERS)
def test_cross_mlx_metal(order):
    data = golden(f'parity_{order}')
    kw = {'order': order, 'spacing': 10., 'dt': .001,
          'pml': [9, 12, 10, 11], 'freq': 18., 'vmax': 1900.}
    mlx = run('mlx', data, **kw)
    metal = run('metal', data, **kw)
    # A fixed reference peak keeps the cross-check symmetric.
    scale = np.max(np.abs(data['expected']))
    assert np.max(np.abs(mlx.astype(np.float64) - metal)) / scale <= CROSS_PATH


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('order', ORDERS)
def test_green_refinement(path, order):
    errors = []
    for spacing in (20, 10, 5):
        data = golden(f'conv_{order}_{spacing}')
        actual = run(path, data, order=order, spacing=float(spacing),
                     dt=.4*spacing/1500., pml=[10, 10, 10, 10],
                     freq=5., vmax=1500.)
        assert rel_err(actual, data['expected']) <= PARITY
        analytic = data['analytic']
        errors.append(np.linalg.norm(actual[0, 0].astype(np.float64) - analytic)
                      / np.linalg.norm(analytic))
    rates = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert CONVERGENCE_COARSE_MIN <= rates[0] <= CONVERGENCE_MAX, rates
    assert CONVERGENCE_FINE_MIN <= rates[1] <= CONVERGENCE_MAX, rates
