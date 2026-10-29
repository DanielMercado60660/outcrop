"""Generate fixed float64 Harmonica reference predictions from float32 inputs."""
from __future__ import annotations

import json
import runpy
from importlib.metadata import version
from pathlib import Path

import harmonica as hm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'judge' / 'golden'
TOL = runpy.run_path(str(ROOT / 'tests' / 'tolerances.py'))


def xyz(x, y, z):
    return np.column_stack((x, y, z)).astype(np.float32)


def pack_windows(windows):
    """Encode ordered, possibly overlapping memberships as CSR-style arrays."""
    offsets = np.zeros(len(windows) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(window) for window in windows], dtype=np.int32)
    indices = np.concatenate(windows).astype(np.int32) if windows else np.empty(0, np.int32)
    return indices, offsets


def verify_windows(observations, sources, data, weights, evaluation, damping,
                   source_windows, data_windows, expected):
    """Independently replay the supplied windows to validate their association and order."""
    observations, sources, evaluation = (a.astype(np.float64) for a in
                                         (observations, sources, evaluation))
    data, weights = data.astype(np.float64), weights.astype(np.float64)
    jac = 1 / np.linalg.norm(observations[:, None] - sources[None, :], axis=2)
    residual = data.copy()
    coefficients = np.zeros(len(sources))
    for source_window, data_window in zip(source_windows, data_windows):
        sub = jac[np.ix_(data_window, source_window)]
        scale = np.std(sub, axis=0)
        scale[scale == 0] = 1
        root_weight = np.sqrt(weights[data_window])
        design = sub / scale * root_weight[:, None]
        design = np.vstack((design, np.sqrt(damping) * np.eye(len(source_window))))
        target = np.concatenate((residual[data_window] * root_weight,
                                 np.zeros(len(source_window))))
        update = np.linalg.lstsq(design, target, rcond=None)[0] / scale
        coefficients[source_window] += update
        residual -= jac[:, source_window] @ update
    eval_jac = 1 / np.linalg.norm(evaluation[:, None] - sources[None, :], axis=2)
    assert np.allclose(eval_jac @ coefficients, expected, rtol=1e-10, atol=1e-10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(93421)
    x, y = np.meshgrid(np.linspace(0, 1000, 9), np.linspace(0, 800, 8))
    x = x.ravel() + rng.uniform(-16, 16, x.size)
    y = y.ravel() + rng.uniform(-16, 16, y.size)
    observations = xyz(x, y, 80 + rng.uniform(-15, 15, x.size))
    sx, sy = np.meshgrid(np.linspace(0, 1000, 6), np.linspace(0, 800, 5))
    sources = xyz(sx.ravel(), sy.ravel(), np.full(sx.size, -240))
    gx, gy = np.meshgrid(np.linspace(30, 970, 11), np.linspace(20, 780, 9))
    grid = xyz(gx.ravel(), gy.ravel(), np.full(gx.size, 120))
    # Field of two buried point sources, with opposite signs, in arbitrary units.
    true_sources = np.array([[300, 280, -350], [760, 570, -420]], dtype=np.float64)
    amplitudes = np.array([1800., -1050.])
    distance = np.linalg.norm(observations.astype(np.float64)[:, None, :] - true_sources, axis=2)
    data = (1 / distance @ amplitudes).astype(np.float32)
    weights = rng.uniform(0.6, 1.8, x.size).astype(np.float32)
    cases = [('standard_weak', 'standard', 0.001, 7),
             ('standard_damped', 'standard', 0.2, 7),
             ('boosted_seed7', 'boosted', 0.03, 7),
             ('boosted_seed19', 'boosted', 0.03, 19)]
    summaries = {}
    for name, method, damping, seed in cases:
        args = {'points': tuple(sources.astype(np.float64).T), 'damping': damping,
                'parallel': False, 'dtype': 'float64'}
        if method == 'boosted':
            model = hm.EquivalentSourcesGB(**args, window_size=600., random_state=seed)
        else:
            model = hm.EquivalentSources(**args)
        model.fit(tuple(observations.astype(np.float64).T), data.astype(np.float64),
                  weights=weights.astype(np.float64))
        prediction = model.predict(tuple(grid.astype(np.float64).T))
        train = model.predict(tuple(observations.astype(np.float64).T))
        if method == 'boosted':
            source_windows, data_windows = model._create_windows(
                tuple(observations.astype(np.float64).T))
            verify_windows(observations, sources, data, weights, grid, damping,
                           source_windows, data_windows, prediction)
        else:
            source_windows, data_windows = [], []
        source_indices, source_offsets = pack_windows(source_windows)
        data_indices, data_offsets = pack_windows(data_windows)
        assert np.isfinite(prediction).all()
        assert np.max(np.abs(prediction)) > 0.1
        assert np.sqrt(np.mean((train - data) ** 2)) < np.sqrt(np.mean(data ** 2))
        np.savez_compressed(OUT / f'{name}.npz', coordinates=observations, data=data,
                            points=sources, evaluation=grid, weights=weights,
                            prediction=prediction, damping=np.float64(damping),
                            window_size=np.float64(600.), seed=np.int64(seed),
                            method=np.array(method), data_indices=data_indices,
                            data_offsets=data_offsets, source_indices=source_indices,
                            source_offsets=source_offsets)
        summaries[name] = {'method': method, 'damping': damping, 'seed': seed,
                           'window_size_m': 600., 'prediction_min': float(prediction.min()),
                           'prediction_max': float(prediction.max()),
                           'train_rmse': float(np.sqrt(np.mean((train-data)**2))),
                           'windows': len(source_windows),
                           'window_data_counts': [len(w) for w in data_windows],
                           'window_source_counts': [len(w) for w in source_windows]}
        print(name, summaries[name])
    assert summaries['boosted_seed7']['windows'] > 1
    assert summaries['boosted_seed19']['windows'] > 1
    meta = {'reference': 'Harmonica EquivalentSources and EquivalentSourcesGB',
            'versions': {p: version(p) for p in ('harmonica', 'verde', 'numpy', 'scipy', 'scikit-learn', 'numba')},
            'seed_data': 93421, 'input_precision': 'float32, promoted exactly to float64 for fitting',
            'observation_count': len(observations), 'source_count': len(sources),
            'grid_shape': [9, 11], 'coordinates': 'metres, easting northing upward',
            'synthetic_field': 'two buried point-source 1/r fields, strengths 1800 and -1050',
            'sample_weights': 'uniform 0.6 to 1.8, fixed seed', 'cases': summaries,
            'window_inputs': 'ordered zero-based int32 membership indices and CSR-style offsets extracted from the fitted Harmonica model; seed and size are provenance only',
            'tolerances': {'grid_max_relative_error': {'value': TOL['GRID_REL_ERR'], 'reason': 'specified threshold; float32 regularized inverse and boosted sequential updates'},
                           'cross_path_max_relative_error': {'value': TOL['CROSS_PATH_REL_ERR'], 'reason': 'same specified prediction threshold for two float32 paths'}}}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
