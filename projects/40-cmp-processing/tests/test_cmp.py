import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import (
    ARRAY_REL,
    CROSS_INDEX_ULPS,
    CROSS_SEMBLANCE_ABS,
    NMO_INDEX_ULPS,
    SEMBLANCE_ABS,
    VELOCITY_REL,
)


def inputs(g):
    return {'gather': g['gather'].astype(np.float32),
            'offsets': g['offsets'].astype(np.float32), 'dt': float(g['dt'])}


def invoke(path, name, g):
    kw = inputs(g)
    if name == 'semblance':
        kw.update(velocities=g['velocities'].astype(np.float32), half_window=int(g['half_window']))
    else:
        kw['velocity'] = float(g['velocity'])
    return call(path, name, **kw)[{'nmo': 'corrected', 'stack': 'stack', 'semblance': 'semblance'}[name]]


def index_error_budget(g, ulps):
    # Resolution at the last in-range sample converted to amplitude.
    spacing = float(np.spacing(np.float32(g['gather'].shape[1] - 1)))
    slope = float(np.max(np.abs(np.diff(g['gather'], axis=1))))
    return ulps * spacing * slope


def check(actual, expected, atol, mask=None):
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    if mask is not None:
        actual, expected = actual[mask], expected[mask]
    np.testing.assert_allclose(actual, expected, rtol=ARRAY_REL, atol=atol)


def energetic_mask(g):
    times = np.arange(g['gather'].shape[1])[None, :] * float(g['dt'])
    return (times > 0.2) & g['energy_mask']


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('name,key', [('nmo', 'corrected'), ('stack', 'stack'), ('semblance', 'semblance')])
def test_reference(path, name, key):
    g = golden('cmp')
    mask = energetic_mask(g) if name == 'semblance' else None
    atol = SEMBLANCE_ABS if name == 'semblance' else index_error_budget(g, NMO_INDEX_ULPS)
    check(invoke(path, name, g), g[key], atol, mask)


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_velocity_picks(path):
    g = golden('cmp')
    s = invoke(path, 'semblance', g)
    assert s.shape == g['semblance'].shape and np.isfinite(s).all()
    indices = np.rint(g['event_times']/float(g['dt'])).astype(int)
    picks = g['velocities'][np.argmax(s[:, indices], axis=0)]
    assert np.all(np.abs(picks/g['truth'] - 1) <= VELOCITY_REL), picks


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
def test_zero_gather(path):
    g = golden('zero')
    for name, shape in [('nmo', g['gather'].shape), ('stack', (g['gather'].shape[1],)),
                        ('semblance', (len(g['velocities']), g['gather'].shape[1]))]:
        actual = invoke(path, name, g)
        assert actual.dtype == np.float32 and actual.shape == shape
        assert np.all(actual == 0)


@pytest.mark.parametrize('path', PATHS, ids=PATHS)
@pytest.mark.parametrize('name,key', [('nmo', 'corrected'), ('stack', 'stack')])
def test_boundary_and_signed_offsets(path, name, key):
    g = golden('edge')
    check(invoke(path, name, g), g[key], index_error_budget(g, NMO_INDEX_ULPS))


@pytest.mark.parametrize('name,key', [('nmo', 'corrected'), ('stack', 'stack'), ('semblance', 'semblance')])
def test_mlx_metal_crosscheck(name, key):
    g = golden('cmp')
    mask = energetic_mask(g) if name == 'semblance' else None
    atol = CROSS_SEMBLANCE_ABS if name == 'semblance' else index_error_budget(g, CROSS_INDEX_ULPS)
    check(invoke('metal', name, g), invoke('mlx', name, g), atol, mask)
