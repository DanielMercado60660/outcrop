"""Generate fixed float64 Harmonica reference grids; run from project root."""
import json
from importlib.metadata import version
from pathlib import Path

import harmonica as hm
import numpy as np
import xarray as xr

OUT = Path(__file__).resolve().parent / "golden"
SEED = 7142026


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    cases = {}
    for name, ny, nx, dy, dx in [("periodic", 32, 40, 80., 120.),
                                  ("localized", 33, 39, 90., 140.),
                                  ("rough", 24, 32, 125., 75.)]:
        y, x = np.indices((ny, nx), dtype=np.float64)
        if name == "periodic":
            grid = 4 + 2*np.cos(2*np.pi*(3*x/nx + 2*y/ny)) + 0.7*np.sin(2*np.pi*(x/nx - 3*y/ny))
        elif name == "localized":
            grid = (3*np.exp(-((x-15)**2/45 + (y-17)**2/32))
                    - 1.4*np.exp(-((x-27)**2/22 + (y-11)**2/20)) + 0.015*x)
        else:
            grid = rng.normal(size=(ny, nx)) + 0.2*x/nx
        # Judge the exact float32 input, promoted before all reference work.
        grid = grid.astype(np.float32).astype(np.float64)
        da = xr.DataArray(grid, dims=("northing", "easting"),
                          coords={"northing": np.arange(ny)*dy, "easting": np.arange(nx)*dx})
        height = 175.0 if name != "rough" else 0.0
        inc, dec = (50., -23.) if name != "rough" else (90., 0.)
        mi, md = (35., 37.) if name == "localized" else (inc, dec)
        results = {
            "upward": hm.upward_continuation(da, height).values,
            "rtp": hm.reduction_to_pole(da, inc, dec, mi, md).values,
            "vertical": hm.derivative_upward(da).values,
            "tilt": hm.tilt_angle(da).values,
        }
        assert all(np.isfinite(v).all() for v in results.values())
        assert np.isclose(results["upward"].mean(), grid.mean(), atol=1e-12)
        assert abs(results["vertical"].mean()) < 1e-12
        assert abs(results["rtp"].mean()) < 1e-12
        assert np.max(np.abs(results["tilt"])) <= np.pi/2
        if name == "periodic":
            phase = 2*np.pi*(3*x/nx + 2*y/ny)
            phase2 = 2*np.pi*(x/nx - 3*y/ny)
            k1 = 2*np.pi*np.hypot(3/(nx*dx), 2/(ny*dy))
            k2 = 2*np.pi*np.hypot(1/(nx*dx), 3/(ny*dy))
            expected = 4 + 2*np.exp(-k1*height)*np.cos(phase) + 0.7*np.exp(-k2*height)*np.sin(phase2)
            expected_d = -2*k1*np.cos(phase) - 0.7*k2*np.sin(phase2)
            assert np.max(np.abs(results["upward"] - expected)) < 1e-6
            assert np.max(np.abs(results["vertical"] - expected_d)) < 1e-8
        if name == "rough":
            assert np.max(np.abs(results["upward"] - grid)) < 1e-12
            assert np.max(np.abs(results["rtp"] - (grid-grid.mean()))) < 1e-12
        np.savez_compressed(OUT / f"{name}.npz", grid=grid.astype(np.float32),
                            dx=dx, dy=dy, height=height, inclination=inc, declination=dec,
                            magnetization_inclination=mi, magnetization_declination=md, **results)
        cases[name] = {"shape": [ny, nx], "dx_m": dx, "dy_m": dy, "height_m": height,
                       "inclination_deg": inc, "declination_deg": dec,
                       "magnetization_inclination_deg": mi, "magnetization_declination_deg": md,
                       "input_range": [float(grid.min()), float(grid.max())],
                       "output_ranges": {k: [float(v.min()), float(v.max())] for k, v in results.items()}}
    meta = {"seed": SEED, "versions": {p: version(p) for p in ("numpy", "scipy", "harmonica", "xarray", "xrft")},
            "inputs": cases, "input_units": "arbitrary field units; magnetic field for RTP",
            "reference": "Harmonica high-level transformations, float64; float32-quantized input",
            "padding": "none; periodic FFT", "taper": "none", "edge_band_cells": 2,
            "relative_error": "maximum absolute error divided by maximum absolute reference in the interior",
            "tolerances": {"reference_relative": {"value": 1e-4, "reason": "project-wide float32 parity limit"}}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
