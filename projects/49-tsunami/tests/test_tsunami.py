"""Oracle parity, analytic convergence, conservation, speed and gradients."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import (
    CROSS_PATH,
    DAM_ORDER_MIN,
    DEEP_CROSS_PATH,
    DEEP_PARITY,
    GRADIENT,
    LINEAR_ORDER_MIN,
    MASS,
    OBJECTIVE,
    PARITY,
    SCALE_FLOOR,
    SPEED_RELATIVE,
)


def args(data, mode):
    return {'initial': data['initial'].astype(np.float32), 'dx': float(data['dx']),
            'dt': float(data['dt']), 'steps': int(data['steps']), 'mode': mode,
            'depth': float(data['depth'])}


def state(path, data, mode):
    result = call(path, 'propagate', **args(data, mode))
    assert set(result) == {'state'}
    value = result['state']
    assert value.shape == data['initial'].shape
    assert value.dtype == np.float32
    assert np.isfinite(value).all()
    return value.astype(np.float64)


def component_error(value, reference):
    return np.max(np.abs(value - reference), axis=0) / np.maximum(
        np.max(np.abs(reference), axis=0), SCALE_FLOOR)


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('name,mode', [(f'linear_{n}', 'linear') for n in (100, 200, 400)] +
                         [(f'dam_{n}', 'nonlinear') for n in (100, 200, 400)])
def test_same_grid_reference(path, name, mode):
    data = golden(name)
    value = state(path, data, mode)
    assert np.all(component_error(value, data['reference']) <= PARITY)
    if mode == 'nonlinear':
        assert np.min(value[:, 0]) > 0
        assert abs(value[:, 0].sum() - data['initial'][:, 0].sum()) / data['initial'][:, 0].sum() <= MASS


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('mode', ['linear', 'nonlinear'])
def test_analytic_convergence(path, mode):
    errors = []
    for n in (100, 200, 400):
        data = golden(f'{"linear" if mode == "linear" else "dam"}_{n}')
        value = state(path, data, mode)
        if mode == 'nonlinear':
            x = -100 + (np.arange(n) + 0.5) * float(data['dx'])
            mask = np.abs(x) < 60  # exclude the second, periodic dam at the seam
        else:
            mask = np.ones(n, dtype=bool)
        errors.append(np.mean(np.abs(value[mask, 0] - data['analytic'][mask, 0])))
    order = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(order >= (LINEAR_ORDER_MIN if mode == 'linear' else DAM_ORDER_MIN)), (errors, order)


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_linear_wave_speed(path):
    data = golden('linear_200')
    value = state(path, data, 'linear')
    initial_hat = np.fft.rfft(data['initial'][:, 0])[1]
    final_hat = np.fft.rfft(value[:, 0])[1]
    phase = np.angle(final_hat / initial_hat)
    length = len(value) * float(data['dx'])
    speed = -phase * length / (2 * np.pi * float(data['dt']) * int(data['steps']))
    expected = np.sqrt(9.81 * float(data['depth']))
    assert abs(speed / expected - 1) <= SPEED_RELATIVE


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('mode', ['linear', 'nonlinear'])
def test_centered_finite_difference_gradient(path, mode):
    data = golden(f'grad_{mode}')
    out = call(path, 'objective_and_grad', weights=data['weights'].astype(np.float32), **args(data, mode))
    assert set(out) == {'objective', 'gradient'}
    objective, grad = out['objective'], out['gradient']
    assert objective.shape == () and objective.dtype == np.float32
    assert grad.shape == data['initial'].shape and grad.dtype == np.float32
    assert np.isfinite(objective) and np.isfinite(grad).all()
    assert abs(float(objective) - float(data['objective'])) <= OBJECTIVE * max(1., abs(float(data['objective'])))
    # The golden uses centered float64 finite differences with step 1e-5.
    assert np.max(np.abs(grad - data['gradient'])) <= GRADIENT * np.max(np.abs(data['gradient']))


@pytest.mark.parametrize('name,mode', [(f'linear_{n}', 'linear') for n in (100, 200, 400)] +
                         [(f'dam_{n}', 'nonlinear') for n in (100, 200, 400)])
def test_cross_mlx_metal_state(name, mode):
    data = golden(name)
    a, b = state('mlx', data, mode), state('metal', data, mode)
    assert np.all(component_error(a, b) <= CROSS_PATH)


@pytest.mark.parametrize('mode', ['linear', 'nonlinear'])
def test_cross_mlx_metal_gradient(mode):
    data = golden(f'grad_{mode}')
    kwargs = dict(weights=data['weights'].astype(np.float32), **args(data, mode))
    a = call('mlx', 'objective_and_grad', **kwargs)
    b = call('metal', 'objective_and_grad', **kwargs)
    assert np.max(np.abs(a['gradient'] - b['gradient'])) <= CROSS_PATH * np.max(np.abs(data['gradient']))
    assert abs(float(a['objective']) - float(b['objective'])) <= OBJECTIVE * max(1., abs(float(data['objective'])))


def deep_error(value, reference, depth, mode):
    # The large resting depth must not hide errors in the tsunami signal.
    surface = reference[:, 0] - (depth if mode == 'nonlinear' else 0.)
    scales = np.array([np.max(np.abs(surface)), np.max(np.abs(reference[:, 1]))])
    return np.max(np.abs(value - reference), axis=0) / np.maximum(scales, SCALE_FLOOR)


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('depth', [1000, 4000])
@pytest.mark.parametrize('mode', ['linear', 'nonlinear'])
def test_deep_same_grid_reference(path, depth, mode):
    data = golden(f'deep_{mode}_{depth}')
    value = state(path, data, mode)
    assert np.all(deep_error(value, data['reference'], depth, mode) <= DEEP_PARITY)


@pytest.mark.parametrize('depth', [1000, 4000])
@pytest.mark.parametrize('mode', ['linear', 'nonlinear'])
def test_deep_cross_mlx_metal_state(depth, mode):
    data = golden(f'deep_{mode}_{depth}')
    a, b = state('mlx', data, mode), state('metal', data, mode)
    assert np.all(deep_error(a, b, depth, mode) <= DEEP_CROSS_PATH)
