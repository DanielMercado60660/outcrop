"""Prediction parity with the fixed Harmonica inverse solutions."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err
from tolerances import CROSS_PATH_REL_ERR, GRID_REL_ERR

CASES = ('standard_weak', 'standard_damped', 'boosted_seed7', 'boosted_seed19')


def inputs(case):
    g = golden(case)
    kwargs = {name: g[name] for name in ('coordinates', 'data', 'points', 'evaluation',
                                         'weights', 'data_indices', 'data_offsets',
                                         'source_indices', 'source_offsets')}
    kwargs.update(damping=float(g['damping']), method=str(g['method']))
    return g, kwargs


def prediction(path, kwargs):
    out = call(path, 'equivalent_sources', **kwargs)
    assert set(out) == {'prediction'}
    result = out['prediction']
    assert result.shape == (kwargs['evaluation'].shape[0],)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    return result


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', CASES)
def test_prediction_matches_harmonica(path, case):
    g, kwargs = inputs(case)
    actual = prediction(path, kwargs)
    error = rel_err(actual, g['prediction'])
    assert error <= GRID_REL_ERR, f'{case} {path}: relative error {error:g}'


@pytest.mark.parametrize('case', CASES)
def test_mlx_vs_metal(case):
    _, kwargs = inputs(case)
    mlx = prediction('mlx', kwargs)
    metal = prediction('metal', kwargs)
    error = rel_err(mlx, metal)
    assert error <= CROSS_PATH_REL_ERR, f'{case}: cross-path relative error {error:g}'
