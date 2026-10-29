"""Golden parity and independent signal-separation requirements."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err

from .tolerances import MAX_RESIDUAL_POWER, PARITY, PRIMARY_POWER_MAX, PRIMARY_POWER_MIN


def arguments(case, data, key):
    args = {'gather': data[key].astype(np.float32), 'dt': float(data['dt'])}
    if case == 'fk':
        args.update(dx=float(data['dx']), velocity_cutoff=float(data['cutoff']))
    else:
        args.update(offsets=data['offsets'].astype(np.float32),
                    slowness=float(data['slowness']))
    return args


def run(path, case, data, key):
    fn = 'fk_filter' if case == 'fk' else 'radon_demultiple'
    out = call(path, fn, **arguments(case, data, key))
    assert set(out) == {'filtered'}
    result = out['filtered']
    assert result.shape == data[key].shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    return result


def ratio(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.sum(a * a) / np.sum(b * b)


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', ['fk', 'radon'])
def test_golden(path, case):
    data = golden(case)
    assert rel_err(run(path, case, data, 'gather'), data['expected']) <= PARITY


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', ['fk', 'radon'])
def test_broadband_golden(path, case):
    data = golden(case + '_broadband')
    assert rel_err(run(path, case, data, 'gather'), data['expected']) <= PARITY


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', ['fk', 'radon'])
def test_separation(path, case):
    data = golden(case)
    unwanted = run(path, case, data, 'unwanted')
    primary = run(path, case, data, 'primary')
    assert ratio(unwanted, data['unwanted']) <= MAX_RESIDUAL_POWER
    assert PRIMARY_POWER_MIN <= ratio(primary, data['primary']) <= PRIMARY_POWER_MAX


@pytest.mark.parametrize('case', ['fk', 'radon'])
def test_cross_mlx_metal(case):
    data = golden(case)
    a = run('mlx', case, data, 'gather')
    b = run('metal', case, data, 'gather')
    assert rel_err(a, b) <= PARITY


@pytest.mark.parametrize('case', ['fk', 'radon'])
def test_broadband_cross_mlx_metal(case):
    data = golden(case + '_broadband')
    a = run('mlx', case, data, 'gather')
    b = run('metal', case, data, 'gather')
    assert rel_err(a, b) <= PARITY
