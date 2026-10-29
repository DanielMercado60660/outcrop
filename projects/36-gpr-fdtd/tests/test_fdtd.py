import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import (
    CONVERGENCE_RATIO_MAX, CROSS, PARITY, REFLECTION_RELATIVE, ZERO_HZ_ABSOLUTE,
)

CASES = ['heterogeneous', 'wave_24', 'wave_48', 'wave_96', 'reflection']
FIELDS = ('electric', 'hx', 'hz')


def run(path, data):
    args = {k: data[k] for k in ('epsilon_r', 'electric', 'hx', 'hz')}
    args.update({k: float(data[k]) for k in ('dx', 'dz', 'dt')})
    args['steps'] = int(data['steps'])
    result = call(path, 'propagate', **args)
    assert set(result) == set(FIELDS)
    for field in FIELDS:
        assert result[field].shape == data['electric'].shape
        assert result[field].dtype == np.float32
        assert np.isfinite(result[field]).all()
    return result


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', CASES)
def test_same_grid_reference(path, case):
    data = golden(case)
    result = run(path, data)
    for field in FIELDS:
        if case == 'reflection' and field == 'hz':
            assert np.max(np.abs(result[field] - data['expected_hz'])) <= ZERO_HZ_ABSOLUTE
            continue
        assert rel_err(result[field], data['expected_' + field]) <= PARITY, field


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_free_space_convergence(path):
    errors = []
    for n in (24, 48, 96):
        data = golden(f'wave_{n}')
        e = run(path, data)['electric'].astype(np.float64)
        errors.append(np.max(np.abs(e - data['analytic'])))
    assert errors[1] / errors[0] <= CONVERGENCE_RATIO_MAX
    assert errors[2] / errors[1] <= CONVERGENCE_RATIO_MAX


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_reflection_coefficient(path):
    data = golden('reflection')
    e = run(path, data)['electric']
    start, stop = int(data['reflection_start']), int(data['reflection_stop'])
    measured = float(np.min(e[0, start:stop]))
    expected = float(data['reflection_coefficient'])
    assert abs((measured - expected) / expected) <= REFLECTION_RELATIVE


@pytest.mark.parametrize('case', CASES)
def test_mlx_metal_cross_check(case):
    data = golden(case)
    mlx, metal = run('mlx', data), run('metal', data)
    for field in FIELDS:
        if case == 'reflection' and field == 'hz':
            assert np.max(np.abs(mlx[field] - metal[field])) <= ZERO_HZ_ABSOLUTE
            continue
        assert rel_err(mlx[field], metal[field]) <= CROSS, field
