"""Generate deterministic float64 Harmonica reference values for project 02."""
from __future__ import annotations

import json
import platform
import runpy
from pathlib import Path

import choclo
import harmonica as hm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "judge" / "golden"
FIELDS = ("potential", "g_e", "g_n", "g_z")


def write_case(name, coordinates, prisms, density):
    # Reference inputs exactly match what the float32 implementations receive.
    xyz = np.asarray(coordinates, dtype=np.float32)
    prisms = np.asarray(prisms, dtype=np.float32)
    density = np.asarray(density, dtype=np.float32)
    result = {"coordinates": xyz, "prisms": prisms, "density": density}
    for field in FIELDS:
        result[field] = hm.prism_gravity(
            tuple(xyz.astype(np.float64).T), prisms.astype(np.float64),
            density.astype(np.float64), field=field, dtype=np.float64, parallel=False,
        )
        assert np.isfinite(result[field]).all(), (name, field)
    np.savez_compressed(OUT / f"{name}.npz", **result)
    return {field: [float(result[field].min()), float(result[field].max())] for field in FIELDS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    prism = np.array([[-100, 100, -80, 80, -120, 0]], dtype=np.float32)
    rho = np.array([2670], dtype=np.float32)
    # Center, faces, edges, and a vertex are all valid for these fields.
    special = np.array([[0, 0, 100], [100, 0, -60], [100, 80, -60],
                        [100, 80, 0], [0, 0, -60], [200, 0, -60],
                        [0, -180, -60], [0, 0, -220]], dtype=np.float32)
    random_points = rng.uniform([-250, -220, -200], [250, 220, 150], (24, 3)).astype(np.float32)
    mixed_prisms = np.array([[-180, -20, -70, 90, -210, -30],
                             [40, 170, -120, 60, -160, -10],
                             [-30, 50, 120, 200, -130, 20]], dtype=np.float32)
    mixed_density = np.array([750, -400, 0], dtype=np.float32)
    mixed_xyz = rng.uniform([-300, -260, 30], [300, 260, 250], (25, 3)).astype(np.float32)
    radii = np.array([2, 4, 8, 16], dtype=np.float32)
    directions = np.array([[0.6, 0.3, 0.74161985], [-0.4, 0.8, 0.4472136],
                           [0.2, -0.6, 0.7745967]], dtype=np.float32)
    center = np.array([0, 0, -60], dtype=np.float32)
    far_xyz = (center + radii[:, None, None] * np.float32(200) * directions[None, :, :]).reshape(-1, 3)
    shift = np.array([16384, -8192, 4096], dtype=np.float32)
    translation_xyz = np.array([[20, 30, 100], [-150, 45, 180], [250, -110, 30]], dtype=np.float32)
    shifted_prism = prism + np.repeat(shift, 2)[None, :]
    grad_xyz = np.array([[30, 40, 130], [-160, 70, 90], [180, -130, 70]], dtype=np.float32)
    ranges = {}
    ranges['single'] = write_case('single', np.concatenate([special, random_points]), prism, rho)
    ranges['mixed'] = write_case('mixed', mixed_xyz, mixed_prisms, mixed_density)
    ranges['distance'] = write_case('distance', far_xyz, prism, rho)
    ranges['translation'] = write_case('translation', translation_xyz, prism, rho)
    ranges['translated'] = write_case('translated', translation_xyz + shift, shifted_prism, rho)
    ranges['gradient'] = write_case('gradient', grad_xyz, prism, rho)

    # Validate the selected finite-difference step in float64 against the
    # reference accelerations. Save the float64 derivatives for implementation tests.
    with np.load(OUT / 'gradient.npz') as gradient:
        gradient_data = {key: gradient[key] for key in gradient.files}
        for axis, field, sign in ((0, 'g_e', 1), (1, 'g_n', 1), (2, 'g_z', -1)):
            plus = grad_xyz.astype(np.float64)
            minus = grad_xyz.astype(np.float64)
            plus[:, axis] += 2.0
            minus[:, axis] -= 2.0
            vp = hm.prism_gravity(tuple(plus.T), prism.astype(float), rho.astype(float), field='potential', parallel=False)
            vm = hm.prism_gravity(tuple(minus.T), prism.astype(float), rho.astype(float), field='potential', parallel=False)
            fd = sign * (vp - vm) / 4.0 * 1e5
            assert np.max(np.abs(fd - gradient[field])) / np.max(np.abs(gradient[field])) < 1e-4
            gradient_data[f'fd_{field}'] = fd
    np.savez_compressed(OUT / 'gradient.npz', **gradient_data)

    # Physical sanity: symmetry, attraction, linearity, translation, point mass.
    with np.load(OUT / 'single.npz') as z:
        assert z['potential'][0] > 0 and z['g_z'][0] > 0
        assert abs(z['g_e'][0]) < 1e-12 and abs(z['g_n'][0]) < 1e-12
        assert z['g_e'][5] < 0 and z['g_n'][6] > 0 and z['g_z'][7] < 0
    with np.load(OUT / 'mixed.npz') as m:
        coords = tuple(m['coordinates'].astype(float).T)
        a = hm.prism_gravity(coords, mixed_prisms[:1].astype(float), mixed_density[:1].astype(float), field='g_z', parallel=False)
        b = hm.prism_gravity(coords, mixed_prisms[1:].astype(float), mixed_density[1:].astype(float), field='g_z', parallel=False)
        assert np.allclose(m['g_z'], a + b, rtol=1e-12, atol=1e-15)
    with np.load(OUT / 'distance.npz') as d:
        mass = 200 * 160 * 120 * 2670
        point_potential = 6.67430e-11 * mass / (16 * 200)
        assert abs(d['potential'][-3] / point_potential - 1) < 0.002
    with np.load(OUT / 'translation.npz') as a, np.load(OUT / 'translated.npz') as b:
        for field in FIELDS:
            assert np.allclose(a[field], b[field], rtol=1e-12, atol=1e-15)

    tol = runpy.run_path(str(ROOT / 'tests' / 'tolerances.py'))
    meta = {
        'reference': 'Harmonica prism_gravity (Nagy et al. 2000; Fukushima 2020)',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'harmonica': hm.__version__, 'choclo': choclo.__version__},
        'seed': 20261028,
        'inputs': {'single': 'one 200 x 160 x 120 m prism; 8 special points and 24 fixed-seed points',
                   'mixed': 'three prisms, positive, negative and zero density; 25 fixed-seed points',
                   'distance': 'single prism, three directions at 2, 4, 8, 16 times 200 m from center',
                   'translation': 'same geometry at origin and translated by (16384, -8192, 4096) m',
                   'gradient': 'three exterior points; central finite difference step 2 m'},
        'input_storage': 'float32, promoted exactly to float64 for reference',
        'output_storage': 'float64', 'ranges': ranges,
        'distance_float32_measurement': {
            'provenance': 'Independent review: float32 Nagy/choclo port with safe log/atan and source-local corner-minus-observation coordinates; NumPy float32 and MLX GPU float32. These are maxima reported in review, not measurements made by this script.',
            'widths': [2, 4, 8, 16],
            'max_rel_by_field_over_backends': {
                'potential': [1.9e-5, 1.1e-4, 9.8e-4, 8.3e-3],
                'g_e': [2.6e-5, 3.6e-4, 2.4e-3, 1.6e-2],
                'g_n': [2.1e-5, 2.3e-4, 1.6e-3, 1.0e-2],
                'g_z': [1.4e-5, 2.2e-4, 9.6e-4, 1.7e-2],
            },
            'normalization': 'maximum absolute float64 reference in each field and distance group',
        },
        'tolerances': {'near_rel': [tol['NEAR_REL'], 'float32 near-field parity'],
                       'translation_rel': [tol['TRANSLATION_REL'], 'float32 translation invariance'],
                       'cross_rel': [tol['CROSS_REL'], 'two independent rounding errors'],
                       'distance_rel': [list(tol['DISTANCE_REL']), 'growing corner-term cancellation; at least 1.9x independently measured worst float32 error at each distance'],
                       'distance_cross_rel': [list(tol['DISTANCE_CROSS_REL']), 'two independent distance-dependent errors'],
                       'gradient_rel': [tol['GRADIENT_REL'], 'float64 centered-difference truncation and float32 field rounding']},
    }
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(ranges, indent=2))
    print('Physical checks passed: symmetry, attraction, superposition, translation, far-field point mass.')


if __name__ == '__main__':
    main()
