"""Benchmark the joint inversion; run from the project directory."""
from __future__ import annotations

import importlib.util
import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'common' / 'py'))
os.environ['OUTCROP_PROJECT'] = str(ROOT)
from outcrop_harness import call

SIZES = {
    'small': {'shape': [4, 4, 3], 'observations_per_survey': 72, 'sweeps': 1},
    'medium': {'shape': [6, 6, 4], 'observations_per_survey': 180, 'sweeps': 8},
    'large': {'shape': [8, 8, 5], 'observations_per_survey': 320, 'sweeps': 60},
}


def load_judge():
    spec = importlib.util.spec_from_file_location('reference_11', ROOT / 'judge' / 'make_goldens.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(size):
    rng = np.random.default_rng(1101)
    shape = tuple(size['shape'])
    n = int(np.prod(shape))
    m = size['observations_per_survey']
    indices = np.arange(n).reshape(shape)
    d = np.zeros((3, n, n), dtype=np.float32)
    for axis in range(3):
        for cell in np.ndindex(shape):
            if cell[axis] + 1 < shape[axis]:
                neighbor = list(cell)
                neighbor[axis] += 1
                i, j = indices[cell], indices[tuple(neighbor)]
                d[axis, i, i], d[axis, i, j] = -1, 1
    ag = (rng.normal(size=(m, n)) / np.sqrt(m)).astype(np.float32)
    am = (rng.normal(size=(m, n)) / np.sqrt(m)).astype(np.float32)
    u, v = rng.normal(0.2, 0.3, (2, n)).astype(np.float32)
    return {'gravity_matrix': ag, 'magnetic_matrix': am, 'gravity_data': ag @ u,
            'magnetic_data': am @ v, 'derivatives': d, 'alpha': 0.05, 'smooth': 0.01,
            'coupling': 0.1, 'sweeps': size['sweeps']}


def main():
    judge = load_judge()
    results = {'judge': {}, 'mlx': {}}
    medium_outputs = {}
    for size_name, size in SIZES.items():
        args = inputs(size)
        ref_args = {k: v.astype(np.float64) if isinstance(v, np.ndarray) else v
                    for k, v in args.items()}

        def reference(ref_args=ref_args):
            u, v = judge.invert(ref_args['gravity_matrix'], ref_args['magnetic_matrix'],
                                ref_args['gravity_data'], ref_args['magnetic_data'],
                                ref_args['derivatives'], ref_args['alpha'],
                                ref_args['smooth'], ref_args['coupling'], ref_args['sweeps'])
            return {'density': u, 'susceptibility': v}

        for path, invoke in (('judge', reference), ('mlx', lambda args=args: call('mlx', 'invert', **args))):
            invoke()
            times, peaks, rss = [], [], []
            for _ in range(3):
                if path == 'judge':
                    tracemalloc.start()
                    tracemalloc.reset_peak()
                else:
                    mx.reset_peak_memory()
                start = time.perf_counter()
                output = invoke()
                times.append(time.perf_counter() - start)
                if path == 'judge':
                    peaks.append(tracemalloc.get_traced_memory()[1] / 2**20)
                    tracemalloc.stop()
                else:
                    peaks.append(mx.get_peak_memory() / 2**20)
                    rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
            record = {'wall_s': statistics.median(times), 'peak_mem_mb': max(peaks),
                      'mem_method': ('tracemalloc peak during call' if path == 'judge' else
                                     'MLX reset_peak_memory/get_peak_memory; process RSS high-water mark separately')}
            if rss:
                record['process_peak_rss_mb'] = max(rss)
            results[path][size_name] = record
            if size_name == 'medium':
                medium_outputs[path] = output
            print(f"{path} {size_name}: {record['wall_s']:.3f} s", flush=True)
    ref, actual = medium_outputs['judge'], medium_outputs['mlx']
    errors = []
    for key in ref:
        denominator = np.max(np.abs(ref[key]))
        errors.append(float(np.max(np.abs(actual[key] - ref[key])) / max(denominator, 1e-30)))
    report = {'project': '11-joint-grav-mag', 'sizes': SIZES, 'results': results,
              'medium_max_rel_diff': {'mlx': max(errors)},
              'notes': ('Inversion includes normal-matrix assembly and all alternating sweeps. '
                        'Inputs are deterministic synthetic normalized matrices; reference uses float64. '
                        'Timings include harness and output conversion. MLX inversion uses its CPU stream. '
                        'tracemalloc may omit native allocations; MLX peak is allocator memory, and macOS '
                        'process RSS is a lifetime high-water mark including earlier runs and inputs.')}
    (ROOT / 'bench' / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
