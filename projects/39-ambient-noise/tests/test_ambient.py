import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import CROSS_PATH_RELATIVE, FFT_RELATIVE, GREEN_CORRELATION_MIN

CASES = {'diffuse': (0.01, 3.0, 25.0, 1e-10), 'edge': (0.02, 0.0, 18.0, 0.1), 'nyquist': (0.02, 0.0, 25.0, 0.1)}
KEYS = ('whitened_a', 'whitened_b', 'correlations', 'stack')


def run(path, name):
    g = golden(name)
    dt, low, high, eps = CASES[name]
    return g, call(path, 'ambient_noise', a=g['a'], b=g['b'], dt=dt, low=low, high=high, eps=eps)


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('name', CASES)
def test_reference(path, name):
    g, out = run(path, name)
    assert set(out) == set(KEYS)
    for key in KEYS:
        assert out[key].shape == g[key].shape
        assert out[key].dtype == np.float32
        assert np.isfinite(out[key]).all()
        assert rel_err(out[key], g[key]) <= FFT_RELATIVE, key


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_green_recovery(path):
    g, out = run(path, 'diffuse')
    assert np.corrcoef(out['stack'], g['green'])[0, 1] >= GREEN_CORRELATION_MIN


@pytest.mark.parametrize('name', CASES, ids=lambda name: name + '-mlx-metal')
def test_crosscheck(name):
    _, a = run('mlx', name)
    _, b = run('metal', name)
    for key in KEYS:
        assert rel_err(a[key], b[key]) <= CROSS_PATH_RELATIVE, key
