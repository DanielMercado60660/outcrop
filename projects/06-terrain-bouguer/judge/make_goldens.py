"""Generate float64 Harmonica prism-layer reference data."""
import json
import platform
import runpy
from pathlib import Path

import harmonica as hm
import numpy as np
import xarray

OUT = Path(__file__).parent / "golden"
G = 6.67430e-11


def main():
    OUT.mkdir(exist_ok=True)
    x = np.array([-400., -200., 0., 200., 400.])
    y = np.array([-300., -100., 100., 300.])
    surface = np.array([[80, 120, 180, 100, -40], [50, 200, 350, 220, 20],
                        [-30, 100, 280, 160, 60], [0, -80, 40, 90, 120]], dtype=float)
    reference = np.full_like(surface, 20.)
    density = np.array([[2400, 2500, 2600, 2700, 2800], [2500, 2600, 2700, 2800, 2500],
                        [2700, 2650, 2550, 2450, 2500], [2600, 2500, 2400, 2700, 2800]], dtype=float)
    coords = np.array([[-400, -300, 500], [0, -100, 500], [200, 100, 650],
                       [400, 300, 900], [0, 100, 3000], [0, 100, 8000]], dtype=float)
    layer = hm.prism_layer((x, y), surface, reference,
                           properties={"density": density * np.sign(surface-reference)})
    gz = layer.prism_layer.gravity(tuple(coords.T), field="g_z", parallel=False, dtype="float64")
    ix = np.abs(coords[:, 0, None] - x).argmin(axis=1)
    iy = np.abs(coords[:, 1, None] - y).argmin(axis=1)
    slab = 2*np.pi*G*density[iy, ix]*(surface-reference)[iy, ix]*1e5
    np.savez_compressed(OUT / "terrain.npz", easting=x, northing=y, surface=surface,
                        reference=reference, density=density, coordinates=coords,
                        layer_gz=gz, slab_gz=slab, terrain_gz=gz-slab, correction_gz=-gz)
    for name, relief in (("depression", -100.), ("flat", 0.)):
        s = np.full_like(surface, 20. + relief)
        d = np.full_like(surface, 2670.)
        ll = hm.prism_layer((x, y), s, reference,
                            properties={"density": d * np.sign(s-reference)})
        gg = ll.prism_layer.gravity(tuple(coords.T), field="g_z", parallel=False, dtype="float64")
        ss = np.full(coords.shape[0], 2*np.pi*G*2670*relief*1e5)
        np.savez_compressed(OUT / (name + ".npz"), easting=x, northing=y, surface=s,
                            reference=reference, density=d, coordinates=coords,
                            layer_gz=gg, slab_gz=ss, terrain_gz=gg-ss, correction_gz=-gg)
        assert np.all(gg < 0) if relief else np.all(gg == 0)
    assert np.all(np.isfinite(gz)) and np.all(gz > 0) and gz[-1] < gz[-2] < gz[1]
    # Reproduce the benchmark's float32 inputs before evaluating in float64.
    rng = np.random.default_rng(20261028)
    bx = np.arange(64, dtype=np.float32) * 200
    by = np.arange(64, dtype=np.float32) * 200
    xx, yy = np.meshgrid(bx, by)
    bs = (180 + 80 * np.sin(xx / 1100) * np.cos(yy / 1300)).astype(np.float32)
    br = np.zeros_like(bs)
    bd = (2670 + 100 * np.cos(xx / 1700)).astype(np.float32)
    horizontal = np.column_stack((rng.uniform(bx[0], bx[-1], 256),
                                  rng.uniform(by[0], by[-1], 256))).astype(np.float32)
    heights = np.array([700, 1400, 3000, 8000], dtype=np.float32)
    bc = np.concatenate([np.column_stack((horizontal, np.full(256, h, np.float32)))
                         for h in heights])
    bl = hm.prism_layer((bx.astype(np.float64), by.astype(np.float64)),
                        bs.astype(np.float64), br.astype(np.float64),
                        properties={"density": bd.astype(np.float64)})
    bg = bl.prism_layer.gravity(tuple(bc.astype(np.float64).T), field="g_z",
                               parallel=False, dtype="float64")
    bix = np.floor((bc[:, 0] - bx[0]) / 200 + 0.5).astype(int)
    biy = np.floor((bc[:, 1] - by[0]) / 200 + 0.5).astype(int)
    bslab = 2 * np.pi * G * bd[biy, bix].astype(np.float64) * bs[biy, bix].astype(np.float64) * 1e5
    np.savez_compressed(OUT / "benchmark_regime.npz", easting=bx, northing=by,
                        surface=bs, reference=br, density=bd, coordinates=bc,
                        layer_gz=bg, slab_gz=bslab, terrain_gz=bg-bslab,
                        correction_gz=-bg, heights=heights)
    # At the edge a point can initially see more mass as altitude rises;
    # the layer-averaged signal nevertheless decays over these heights.
    assert np.all(bg > 0) and np.all(np.diff(bg.reshape(4, 256).mean(axis=1)) < 0)
    assert np.all(bslab > 0)
    tolerances = runpy.run_path(str(Path(__file__).parents[1] / "tests" / "tolerances.py"))
    meta = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "harmonica": hm.__version__, "xarray": xarray.__version__},
            "inputs": "terrain.npz contains all inputs and float64 outputs; deterministic hand-built heterogeneous 4x5 DEM with positive and negative relief, 200 m spacing, six elevations 500-8000 m; no random seed needed",
            "conventions": "upward coordinates; downward g_z mGal; density sign follows surface-reference; infinite slab at observation cell; G=6.67430e-11",
            "tolerances": {"near_rtol": [tolerances["NEAR_RTOL"], "project-wide float32 target"], "far_rtol": [tolerances["FAR_RTOL"], "far-field prism corner cancellation"], "abs_mgal": [tolerances["ABS_MGAL"], "near-zero cancellation and float32 summation"], "slab_rtol": [tolerances["SLAB_RTOL"], "float32 slab arithmetic"], "algebra_atol": [tolerances["ALGEBRA_ATOL"], "float32 algebra"]},
            "layer_gz_mgal": gz.tolist(), "slab_gz_mgal": slab.tolist()}
    meta["inputs_benchmark_regime"] = {
        "seed": 20261028, "grid": [64, 64], "spacing_m": 200,
        "surface_m": "180 + 80 sin(x/1100) cos(y/1300)",
        "density_kg_m3": "2670 + 100 cos(x/1700)",
        "reference_m": 0, "observations_per_height": 256,
        "heights_m": heights.tolist(),
        "note": "First 256 coordinates reproduce bench/run_bench.py medium; remaining heights use the same horizontal points. Inputs are float32 before float64 reference evaluation.",
    }
    meta["tolerances"]["benchmark_normalized"] = {
        "700_m": tolerances["NEAR_RTOL"], "higher_altitudes": tolerances["FAR_RTOL"],
        "slab": tolerances["SLAB_RTOL"],
        "reason": "Maximum absolute error divided by maximum absolute reference per output and altitude; no absolute floor. At 700 m use the project-wide target; higher altitude permits the existing far-field cancellation allowance. Slab uses its existing float32 arithmetic bound.",
    }
    meta["benchmark_layer_ranges_mgal"] = [
        [float(v.min()), float(v.max())] for v in bg.reshape(4, 256)]
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("layer g_z (mGal):", gz, "slab (mGal):", slab)
    print("benchmark layer ranges by height (mGal):", meta["benchmark_layer_ranges_mgal"])
    print("benchmark slab range (mGal):", float(bslab.min()), float(bslab.max()))


if __name__ == "__main__":
    main()
