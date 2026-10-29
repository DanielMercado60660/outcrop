import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import MIN_CORRELATION, RELATIVE_MAX_ERROR


def assert_array(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    assert rel_err(actual, expected) <= RELATIVE_MAX_ERROR


def spiking_args(data, case):
    loading = {'recovery': 1e-6, 'loaded': 0.2}[case]
    return {'trace': data['trace'], 'wavelet': data['wavelet'], 'inverse_length': 32, 'stabilization': loading}


def spiking_golden(case):
    return golden({'recovery': 'spiking', 'loaded': 'spiking_loaded'}[case])


def predictive_args(data, case):
    order, lag = {'short': (8, 1), 'gap': (12, 3)}[case]
    return {'trace': data['trace'], 'order': order, 'lag': lag, 'stabilization': 0.01}


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', ['recovery', 'loaded'])
def test_spiking(path, case):
    data = spiking_golden(case)
    out = call(path, 'spiking_deconvolution', **spiking_args(data, case))
    assert_array(out['inverse'], data['inverse'])
    assert_array(out['recovered'], data['recovered'])
    if case == 'recovery':
        assert np.corrcoef(out['recovered'][:len(data['reflectivity'])], data['reflectivity'])[0, 1] >= MIN_CORRELATION


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', ['short', 'gap'])
def test_predictive(path, case):
    data = golden('predictive_' + case)
    out = call(path, 'predictive_deconvolution', **predictive_args(data, case))
    assert_array(out['coefficients'], data['coefficients'])
    assert_array(out['residual'], data['residual'])


@pytest.mark.parametrize('case', ['recovery', 'loaded'])
def test_mlx_metal_spiking(case):
    data = spiking_golden(case)
    a = call('mlx', 'spiking_deconvolution', **spiking_args(data, case))
    b = call('metal', 'spiking_deconvolution', **spiking_args(data, case))
    for key in ('inverse', 'recovered'):
        assert_array(a[key], b[key])


@pytest.mark.parametrize('case', ['short', 'gap'])
def test_mlx_metal_predictive(case):
    data = golden('predictive_' + case)
    a = call('mlx', 'predictive_deconvolution', **predictive_args(data, case))
    b = call('metal', 'predictive_deconvolution', **predictive_args(data, case))
    for key in ('coefficients', 'residual'):
        assert_array(a[key], b[key])
