"""Numerical contract tests; all inputs and reference results are saved fixtures."""
import numpy as np
import pytest
import tolerances as tol
from outcrop_harness import call, golden, rel_err

PATHS = ['metal']  # this project has no MLX path


def run(path, data):
    result = call(path, 'propagate', velocity=data['velocity'].astype(np.float32),
                  initial=data['initial'].astype(np.float32),
                  previous=data['previous'].astype(np.float32),
                  spacing=float(data['spacing']), dt=float(data['dt']),
                  steps=int(data['steps']))
    assert set(result) == {'final', 'last'}
    for a in result.values():
        assert a.shape == data['velocity'].shape
        assert a.dtype == np.float32
        assert np.isfinite(a).all()
    return result


def comparison_floor(fixture, data):
    # A plane wave has unit initial amplitude even when its final field is near zero.
    return float(np.max(np.abs(data['initial']))) if fixture.startswith('plane_') else 0.0


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('fixture', ['heterogeneous', 'plane_12', 'plane_24', 'plane_48'])
def test_deepwave_parity(path, fixture):
    data = golden(fixture)
    result = run(path, data)
    for name in ('final', 'last'):
        assert rel_err(result[name], data[name], floor=comparison_floor(fixture, data)) <= tol.PARITY_RTOL


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_invariant_axis_matches_2d_reference(path):
    data = golden('invariant')
    result = run(path, data)
    for name in ('final', 'last'):
        reference = np.broadcast_to(data[name + '_2d'], data['velocity'].shape)
        assert rel_err(result[name], reference) <= tol.PARITY_RTOL


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_analytic_homogeneous_convergence(path):
    errors = []
    for n in (12, 24, 48):
        data = golden(f'plane_{n}')
        result = run(path, data)
        errors.append(np.max(np.abs(result['final'].astype(np.float64) - data['exact'])))
    assert max(errors) < tol.ANALYTIC_ERROR_MAX
    assert all(error > 0 for error in errors)
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders >= tol.ORDER_MIN) & (orders <= tol.ORDER_MAX)), (errors, orders)


@pytest.mark.skip(reason='no MLX path for this project')
@pytest.mark.parametrize('fixture', ['heterogeneous', 'invariant', 'plane_12', 'plane_24', 'plane_48'])
def test_mlx_metal_cross_check(fixture):
    data = golden(fixture)
    mlx = run('mlx', data)
    metal = run('metal', data)
    for name in ('final', 'last'):
        assert rel_err(metal[name], mlx[name], floor=comparison_floor(fixture, data)) <= tol.CROSS_PATH_RTOL
