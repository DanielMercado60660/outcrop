"""Benchmark forward propagation and rendering."""
import importlib.util
import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / 'common' / 'py'))
os.environ['OUTCROP_PROJECT'] = str(ROOT)
from outcrop_harness import call

SIZES = {
    'small': {'ny': 160, 'nx': 160, 'nt': 140, 'nr': 64, 'pml_width': 12},
    'medium': {'ny': 320, 'nx': 320, 'nt': 240, 'nr': 128, 'pml_width': 12},
    'large': {'ny': 1000, 'nx': 1000, 'nt': 360, 'nr': 256, 'pml_width': 12},
}


def make_case(ny, nx, nt, nr, pml_width):
    rng = np.random.default_rng(50001316)
    y, x = np.mgrid[:ny, :nx]
    velocity = (1800 + 300 * (y >= ny * .55)
                + 120 * np.exp(-(((x / nx - .65) / .15) ** 2
                                 + ((y / ny - .65) / .12) ** 2))
                + rng.uniform(-5, 5, (ny, nx))).astype(np.float32)
    dt = .001
    a = (np.pi * 25 * (np.arange(nt) * dt - .05)) ** 2
    source = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)[None, :]
    receivers = np.column_stack((np.full(nr, ny // 4),
                                  np.linspace(5, nx - 6, nr, dtype=np.int64)))
    p = pml_width
    return {'velocity': velocity, 'state': np.zeros((6, ny + 2*p, nx + 2*p), np.float32),
            'source_amplitudes': source,
            'source_locations': np.array([[ny // 4, nx // 2]], np.int64),
            'receiver_locations': receivers, 'grid_spacing': [10., 10.], 'dt': dt,
            'accuracy': 4, 'pml_width': p, 'pml_freq': 25., 'max_velocity': 2500.,
            'boundary': 'pml', 'render_scale': 1.}


def load_reference():
    spec = importlib.util.spec_from_file_location('wave_judge', ROOT / 'judge' / 'make_goldens.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch.set_num_threads(2)
    return module.reference


def judge_call(reference, case):
    return reference({**case, 'grid_spacing': np.asarray(case['grid_spacing'])})


def benchmark(path, case, reference):
    invoke = (lambda: judge_call(reference, case)) if path == 'judge' else (
        lambda: call('metal', 'run_frames', **case))
    invoke()
    times, peaks = [], []
    for _ in range(3):
        if path == 'judge':
            tracemalloc.start()
        else:
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        start = time.perf_counter()
        result = invoke()
        times.append(time.perf_counter() - start)
        if path == 'judge':
            peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
            tracemalloc.stop()
        else:
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peaks.append(max(0, after - before) / 1024**2)
        del result
    method = ('tracemalloc peak; excludes native Torch allocations' if path == 'judge' else
              'RUSAGE_CHILDREN ru_maxrss high-water delta, macOS bytes; may be zero after warm-up')
    return {'wall_s': statistics.median(times), 'peak_mem_mb': max(peaks), 'mem_method': method}


def main():
    reference = load_reference()
    results = {'judge': {}, 'metal': {}}
    for name, size in SIZES.items():
        case = make_case(**size)
        for path, path_results in results.items():
            if path == 'judge' and name == 'large':
                path_results[name] = {'skipped': 'Large float64 CPU reference may exceed 120 s.'}
            else:
                print(f'Benchmarking {path} / {name}', flush=True)
                path_results[name] = benchmark(path, case, reference)
    case = make_case(**SIZES['medium'])
    expected = judge_call(reference, case)
    actual = call('metal', 'run_frames', **case)
    pairs = (('state', 'expected_state'), ('receiver_amplitudes', 'expected_receivers'),
             ('rgba', 'expected_rgba'), ('texture_samples', 'expected_texture_samples'))
    errors = []
    for actual_key, expected_key in pairs:
        a = np.asarray(actual[actual_key], dtype=np.float64)
        b = np.asarray(expected[expected_key], dtype=np.float64)
        peak = float(np.max(np.abs(b)))
        error = float(np.max(np.abs(a - b)))
        errors.append(error / peak if peak else error)
    output = {'project': '50-wave-sandbox', 'sizes': SIZES, 'results': results,
              'medium_max_rel_diff': {'metal': max(errors)},
              'notes': 'Forward run_frames: fourth-order CFS-PML, 10 m spacing, 1 ms step; '
                       'fixed seed. Timings include reference rendering or Metal startup, file '
                       'transport and every-frame rendering. Relative difference is the maximum '
                       'of per-output max absolute errors normalized by each reference peak. '
                       'Child ru_maxrss is a cumulative high-water mark, so its delta is not a '
                       'reliable per-call peak and may omit GPU allocations. CPU tracemalloc '
                       'excludes native allocations. Runtime targets are estimates.'}
    (ROOT / 'bench' / 'results.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
