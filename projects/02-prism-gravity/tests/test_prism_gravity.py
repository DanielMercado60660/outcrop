"""Parity, invariance, distance dependence, and field/potential consistency."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err

from .tolerances import (
    CROSS_REL,
    DISTANCE_CROSS_REL,
    DISTANCE_REL,
    GRADIENT_REL,
    NEAR_REL,
    TRANSLATION_REL,
)

FIELDS = ('potential', 'g_e', 'g_n', 'g_z')


def evaluate(path, data, field, coordinates=None):
    xyz = data['coordinates'] if coordinates is None else coordinates
    out = call(path, 'prism_gravity', coordinates=xyz, prisms=data['prisms'],
               density=data['density'], field=field)
    assert set(out) == {field}
    values = out[field]
    assert values.shape == (len(xyz),)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    return values


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('case', ['single', 'mixed'])
@pytest.mark.parametrize('field', FIELDS)
def test_reference(path, case, field):
    data = golden(case)
    assert rel_err(evaluate(path, data, field), data[field]) <= NEAR_REL


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('field', FIELDS)
def test_translation(path, field):
    base, moved = golden('translation'), golden('translated')
    a, b = evaluate(path, base, field), evaluate(path, moved, field)
    assert rel_err(a, base[field]) <= NEAR_REL
    assert rel_err(b, moved[field]) <= NEAR_REL
    assert rel_err(b, a) <= TRANSLATION_REL


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('field', FIELDS)
def test_error_vs_distance(path, field):
    data = golden('distance')
    actual = evaluate(path, data, field).reshape(4, 3)
    expected = data[field].reshape(4, 3)
    errors = [rel_err(a, b) for a, b in zip(actual, expected)]
    print(f'{path} {field}: relative error at 2, 4, 8, 16 widths = {errors}')
    assert all(err <= limit for err, limit in zip(errors, DISTANCE_REL)), errors


@pytest.mark.parametrize('case', ['single', 'mixed', 'translation', 'translated', 'gradient'])
@pytest.mark.parametrize('field', FIELDS)
def test_cross_mlx_metal(case, field):
    data = golden(case)
    a, b = evaluate('mlx', data, field), evaluate('metal', data, field)
    # Normalize by the oracle to avoid making either implementation the reference.
    scale = np.max(np.abs(data[field]))
    assert np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))) <= CROSS_REL * scale


@pytest.mark.parametrize('field', FIELDS)
def test_distance_cross_mlx_metal(field):
    data = golden('distance')
    a = evaluate('mlx', data, field).reshape(4, 3).astype(np.float64)
    b = evaluate('metal', data, field).reshape(4, 3).astype(np.float64)
    ref = data[field].reshape(4, 3)
    errors = [np.max(np.abs(x - y)) / np.max(np.abs(z)) for x, y, z in zip(a, b, ref)]
    print(f'mlx-metal {field}: relative difference at 2, 4, 8, 16 widths = {errors}')
    assert all(err <= limit for err, limit in zip(errors, DISTANCE_CROSS_REL)), errors


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('field', ['g_e', 'g_n', 'g_z'])
def test_potential_gradient(path, field):
    data = golden('gradient')
    # The oracle differentiates float64 potential with a 2 m centered step.
    # g_z reverses the upward-coordinate derivative; the golden includes this.
    assert rel_err(evaluate(path, data, field), data[f'fd_{field}']) <= GRADIENT_REL
