"""Deterministic analytic fixed-ray oracle; run from the project folder.

No external data or network. NumPy float64 SVD least squares implements the
spec's reference. Geometry is analytic line/cell clipping, independently checked
by a segment-breakpoint construction and explicit hand values.
"""

from __future__ import annotations

import importlib.util
import json
import platform
from hashlib import sha256
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "judge" / "golden"
SEED = 27001
spec = importlib.util.spec_from_file_location("judge_tolerances", ROOT / "tests" / "tolerances.py")
tol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tol)


def f32(values):
    return np.ascontiguousarray(values, dtype=np.float32)


def ray_lengths(x_edges, z_edges, sources, receivers):
    """Float64 per-cell slab clipping; boundary ownership matches the API."""
    x, z, src, rec = [np.asarray(v, dtype=np.float64) for v in (x_edges, z_edges, sources, receivers)]
    lengths = np.zeros((len(src), (len(x) - 1) * (len(z) - 1)), dtype=np.float64)
    for r, (start, end) in enumerate(zip(src, rec, strict=True)):
        direction = end - start
        distance = np.linalg.norm(direction)
        for iz in range(len(z) - 1):
            for ix in range(len(x) - 1):
                lower, upper = 0.0, 1.0
                for axis, edges, cell in ((0, x, ix), (1, z, iz)):
                    if direction[axis] == 0:
                        owner = min(np.searchsorted(edges, start[axis], side="right") - 1, len(edges) - 2)
                        if owner != cell:
                            lower, upper = 1.0, 0.0
                            break
                    else:
                        a = (edges[cell] - start[axis]) / direction[axis]
                        b = (edges[cell + 1] - start[axis]) / direction[axis]
                        lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
                lengths[r, iz * (len(x) - 1) + ix] = max(0.0, upper - lower) * distance
    return lengths


def breakpoint_lengths(x_edges, z_edges, sources, receivers):
    """Independent integration: sort ray/grid intersections, assign midpoints."""
    x, z, src, rec = [np.asarray(v, dtype=np.float64) for v in (x_edges, z_edges, sources, receivers)]
    out = np.zeros((len(src), (len(x) - 1) * (len(z) - 1)))
    for r, (a, b) in enumerate(zip(src, rec, strict=True)):
        d = b - a
        cuts = [0.0, 1.0]
        for k, edges in enumerate((x, z)):
            if d[k] != 0:
                cuts.extend(t for t in (edges - a[k]) / d[k] if 0 < t < 1)
        cuts = np.unique(cuts)
        for lo, hi in pairwise(cuts):
            midpoint = a + (lo + hi) * 0.5 * d
            ix = min(np.searchsorted(x, midpoint[0], side="right") - 1, len(x) - 2)
            iz = min(np.searchsorted(z, midpoint[1], side="right") - 1, len(z) - 2)
            out[r, iz * (len(x) - 1) + ix] += (hi - lo) * np.linalg.norm(d)
    return out


def check_close(a, b):
    np.testing.assert_allclose(a, b, rtol=tol.ORACLE_RTOL, atol=tol.ORACLE_ATOL)


def objective(matrix, times, model, prior, damping):
    g, t, s, p = [np.asarray(v, np.float64) for v in (matrix, times, model, prior)]
    residual, update = g @ s - t, s - p
    value = 0.5 * (residual @ residual + damping**2 * (update @ update))
    gradient = g.T @ residual + damping**2 * update
    return np.asarray(value), gradient


def solution(matrix, times, prior, damping):
    g, t, p = [np.asarray(v, np.float64) for v in (matrix, times, prior)]
    a = np.vstack((g, damping * np.eye(g.shape[1])))
    rhs = np.concatenate((t - g @ p, np.zeros(g.shape[1])))
    update, _, rank, singular = np.linalg.lstsq(a, rhs, rcond=None)
    assert rank == g.shape[1], "Test fixture has no unique minimizer"
    s = p + update
    # Independently check stationarity against matrix/vector scales.
    _, grad = objective(g, t, s, p, damping)
    scale = np.linalg.norm(g) * np.linalg.norm(t) + damping**2 * np.linalg.norm(p)
    assert np.linalg.norm(grad) <= tol.ORACLE_RTOL * scale
    return s, g @ s, float(singular[0] / singular[-1])


def save(name, **arrays):
    np.savez_compressed(GOLDEN / f"{name}.npz", **arrays)


def geometry_fixture(name, x, z, src, rec, hand=None):
    x, z, src, rec = map(f32, (x, z, src, rec))
    lengths = ray_lengths(x, z, src, rec)
    check_close(lengths, breakpoint_lengths(x, z, src, rec))
    check_close(lengths.sum(axis=1), np.linalg.norm(rec.astype(np.float64) - src, axis=1))
    check_close(lengths, ray_lengths(x, z, rec, src))
    if hand is not None:
        check_close(lengths, hand)
    save(name, x_edges=x, z_edges=z, sources=src, receivers=rec, matrix=lengths,
         path_lengths=np.linalg.norm(rec.astype(np.float64) - src, axis=1))
    return lengths


def inverse_fixture(name, matrix, times, prior, damping, **extra):
    matrix, times, prior = map(f32, (matrix, times, prior))
    s, predicted, condition = solution(matrix, times, prior, damping)
    save(name, matrix=matrix, travel_times=times, reference_slowness=prior,
         damping=np.asarray(damping, np.float64), slowness=s, predicted_times=predicted, **extra)
    return s, predicted, condition


def main():
    GOLDEN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summaries = {}

    # Two 1 km square cells in each dimension: rows are hand-integrable rays.
    root2 = np.sqrt(2.0)
    geometry_fixture(
        "geometry_hand", [0, 1000, 2000], [0, 1000, 2000],
        [[0, 500], [0, 1500], [500, 0], [0, 0], [0, 2000],
         [0, 1000], [1000, 0], [2000, 0], [0, 2000], [750, 750]],
        [[2000, 500], [2000, 1500], [500, 2000], [2000, 2000], [2000, 0],
         [2000, 1000], [1000, 2000], [2000, 2000], [2000, 2000], [750, 750]],
        [[1000, 1000, 0, 0], [0, 0, 1000, 1000], [1000, 0, 1000, 0],
         [1000*root2, 0, 0, 1000*root2], [0, 1000*root2, 1000*root2, 0],
         [0, 0, 1000, 1000], [0, 1000, 0, 1000], [0, 1000, 0, 1000],
         [0, 0, 1000, 1000], [0, 0, 0, 0]],
    )
    x = f32([-250, -100, 80, 400, 900])
    z = f32([100, 250, 600, 1100])
    src = f32(rng.uniform([-250, 100], [900, 1100], (19, 2)))
    rec = f32(rng.uniform([-250, 100], [900, 1100], (19, 2)))
    geometry_fixture("geometry_irregular", x, z, src, rec)

    # G/1000 = [[1,0],[0,1],[1,1]], G'G/1e6 = [[2,1],[1,2]].
    # The noisy sum datum gives s = [(2*t0-t1+t2),(-t0+2*t1+t2)] / 3000.
    hand_g = f32([[1000, 0], [0, 1000], [1000, 1000]])
    hand_t = f32([0.5, 0.6, 1.12])
    hand_s = np.array([2*float(hand_t[0])-float(hand_t[1])+float(hand_t[2]),
                       -float(hand_t[0])+2*float(hand_t[1])+float(hand_t[2])]) / 3000
    s, _, condition = inverse_fixture("small_hand", hand_g, hand_t, [0.00045, 0.00055], 0.0,
                                      hand_slowness=hand_s)
    check_close(s, hand_s)
    summaries["small_hand"] = {"slowness_s_per_m": s.tolist(), "condition_augmented": condition,
                                "hand_derivation": "inv(G.T G)= [[2,-1],[-1,2]]/3e6; multiply G.T t."}

    # Underdetermined, nonuniform prior: catches damping vs damping**2 and
    # returning updates instead of absolute slowness. Duplicate columns remain.
    g = f32([[700, 0, 700, 0], [0, 900, 0, 900], [300, 400, 300, 400]])
    s, _, condition = inverse_fixture("small_damped", g, [0.69, 0.97, 0.72],
                                      [0.00040, 0.00048, 0.00061, 0.00055], 300.0)
    summaries["small_damped"] = {"slowness_s_per_m": s.tolist(), "condition_augmented": condition}

    # Same geometry, two noise levels. 12x10 cells, 2x2 cell checker squares.
    x, z = f32(np.arange(13) * 100), f32(np.arange(11) * 100)
    levels_z, levels_x = np.linspace(25, 975, 15), np.linspace(25, 1175, 15)
    src, rec = [], []
    for a in levels_z:
        for b in levels_z:
            src.append([0, a])
            rec.append([1200, b])
    for a in levels_x:
        for b in levels_x:
            src.append([a, 0])
            rec.append([b, 1000])
    src, rec = f32(src), f32(rec)
    geom = geometry_fixture("geometry_checkerboard", x, z, src, rec)
    # Inversion inputs are float32; all references solve those exact inputs.
    g = f32(geom)
    iz, ix = np.indices((10, 12))
    truth = f32(0.0005 + 0.000025 * (1 - 2 * ((iz // 2 + ix // 2) % 2))).ravel()
    prior = f32(np.full(120, 0.0005))
    directions = rec.astype(np.float64) - src
    horizontal = np.abs(directions[:, 0]) >= np.abs(directions[:, 1])
    hits = geom > 0
    coverage = hits.sum(axis=0).reshape(10, 12)
    horizontal_hits = hits[horizontal].sum(axis=0).reshape(10, 12)
    vertical_hits = hits[~horizontal].sum(axis=0).reshape(10, 12)
    mask = ((iz >= 1) & (iz < 9) & (ix >= 1) & (ix < 11)
            & (coverage >= 12) & (horizontal_hits >= 4) & (vertical_hits >= 4)).ravel()
    assert np.count_nonzero(mask) > 0 and np.std(truth[mask]) > 0
    for name, noise_std in (("checkerboard_clean", 0.0), ("checkerboard_noisy", 2.0e-5)):
        noise = rng.normal(0, noise_std, len(g))
        times = f32(g.astype(np.float64) @ truth.astype(np.float64) + noise)
        s, predicted, condition = inverse_fixture(
            name, g, times, prior, 20.0, true_slowness=truth, mask=mask,
            coverage=coverage, horizontal_hits=horizontal_hits, vertical_hits=vertical_hits,
            grid_shape=np.array([10, 12], np.int32), noise_seconds=noise,
        )
        corr = float(np.corrcoef(s[mask] - prior[mask], truth[mask] - prior[mask])[0, 1])
        assert corr >= tol.CHECKERBOARD_MIN_CORRELATION
        assert np.all(s > 0) and np.all(predicted > 0)
        summaries[name] = {
            "nrays": len(g), "ncells": g.shape[1], "mask_cells": int(mask.sum()),
            "noise_std_seconds": noise_std, "correlation": corr,
            "condition_augmented": condition, "slowness_range_s_per_m": [float(s.min()), float(s.max())],
            "velocity_range_m_per_s": [float(1/s.max()), float(1/s.min())],
            "travel_time_range_s": [float(times.min()), float(times.max())],
            "residual_l2_s": float(np.linalg.norm(predicted - times)),
        }

    # Gradient fixtures use sparse nonnegative lengths and off-solution models.
    # Store every centered probe: tests never synthesize their own test inputs.
    g = f32(rng.uniform(50, 900, (9, 6)))
    g[rng.uniform(size=g.shape) < 0.35] = 0
    # Avoid crossing a float32 binade within a probe, so both endpoints are
    # exactly symmetric about the saved evaluation point after quantization.
    model = f32([0.00046, 0.00052, 0.00057, 0.00042, 0.00061, 0.00054])
    prior = f32([0.00050, 0.00048, 0.00053, 0.00051, 0.00055, 0.00047])
    truth = f32([0.00051, 0.00045, 0.00049, 0.00056, 0.00050, 0.00053])
    times = f32(g.astype(np.float64) @ truth.astype(np.float64) + rng.normal(0, 0.0003, len(g)))
    plus, minus = np.tile(model, (len(model), 1)), np.tile(model, (len(model), 1))
    for j in range(len(model)):
        plus[j, j] = model[j] + tol.FD_STEP_S_PER_M
        minus[j, j] = model[j] - tol.FD_STEP_S_PER_M
    separations = (np.diag(plus).astype(np.float64) - np.diag(minus).astype(np.float64))
    check_close((np.diag(plus).astype(np.float64) + np.diag(minus).astype(np.float64)) / 2,
                model.astype(np.float64))
    for name, damping in (("gradient_plain", 0.0), ("gradient_damped", 350.0)):
        value, gradient = objective(g, times, model, prior, damping)
        plus_values = np.array([objective(g, times, m, prior, damping)[0] for m in plus])
        minus_values = np.array([objective(g, times, m, prior, damping)[0] for m in minus])
        fd = (plus_values - minus_values) / separations
        check_close(fd, gradient)
        assert np.all(gradient != 0)
        save(name, matrix=g, travel_times=times, slowness=model,
             reference_slowness=prior, damping=np.asarray(damping), objective=value,
             gradient=gradient, fd_plus=plus, fd_minus=minus, fd_separation=separations,
             fd_plus_objectives=plus_values, fd_minus_objectives=minus_values, fd_gradient=fd)
        summaries[name] = {"objective_s_squared": float(value), "gradient_range_s_m": [float(gradient.min()), float(gradient.max())],
                           "fd_max_abs_difference": float(np.max(np.abs(fd - gradient)))}

    meta = {
        "project": "27-traveltime-tomo", "seed": SEED,
        "reference": "Analytic straight-line/cell intersections; NumPy float64 SVD least squares (numpy.linalg.lstsq, rcond=None).",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pytest": version("pytest"), "ruff": version("ruff")},
        "precision": "All implementation inputs quantized to float32 before float64 reference evaluation; reference outputs float64.",
        "model": "Fixed straight rays; 2-D cellwise absolute slowness, no eikonal relinearization.",
        "objective": "0.5*||G@s-t||^2 + 0.5*damping^2*||s-prior||^2 (unweighted, no normalization)",
        "input_inventory": {},
        "golden_sha256": {},
        "locked_test_count": 35,
        "checkerboard": {"shape_z_x": [10, 12], "cell_size_m": 100, "block_size_cells": [2, 2],
                         "background_slowness_s_per_m": 0.0005,
                         "true_slowness_s_per_m": [0.000475, 0.000525],
                         "acquisition": "225 all-paired left/right rays + 225 all-paired top/bottom rays, endpoint levels 25 m inside each corner",
                         "mask_rule": "Interior iz=1..8, ix=1..10; >=12 nonzero ray lengths, >=4 abs(dx)>=abs(dz) rays, >=4 other rays. Computed only from geometry; frozen in each npz.",
                         "damping_m": 20.0},
        "tolerances": tol.TOLERANCE_METADATA,
        "sanity": summaries,
    }
    for path in sorted(GOLDEN.glob("*.npz")):
        meta["golden_sha256"][path.name] = sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            meta["input_inventory"][path.name] = {
                key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files
            }
    (GOLDEN / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"Wrote {len(meta['input_inventory'])} NPZ files, {sum(p.stat().st_size for p in GOLDEN.iterdir())} bytes total.")


if __name__ == "__main__":
    main()
