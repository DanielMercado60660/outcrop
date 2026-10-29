"""Generate Boule float64 WGS84 references from exactly representable inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import boule
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import tolerances

OUT = ROOT / "judge" / "golden"
SEED = 82026


def main():
    rng = np.random.default_rng(SEED)
    cases = {
        "surface": (
            np.array([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90], dtype=np.float32),
            np.zeros(13, dtype=np.float32),
        ),
        "near_earth": (
            np.array([[-90, -60, -45, -0.01, 0, 0.01],
                      [15, 30, 45, 60, 75, 90],
                      [0, 0, 45, 45, -45, 90]], dtype=np.float32),
            np.array([[0, 1, 10, 100, 1000, 10000],
                      [50000, 100000, 400000, 1000000, 2000000, 10000000],
                      [0, 1, 0, 1, 1000, 1000]], dtype=np.float32),
        ),
        "orbital": (
            np.concatenate((np.array([-90, -45, 0, 45, 90]), rng.uniform(-90, 90, 35))).astype(np.float32),
            np.concatenate((np.full(5, 20_000_000), rng.uniform(0, 20_000_000, 35))).astype(np.float32),
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ranges = {}
    for name, (latitude, height) in cases.items():
        # Round inputs to the implementation dtype before reference evaluation.
        gravity = np.asarray(boule.WGS84.normal_gravity(
            (None, latitude.astype(np.float64), height.astype(np.float64)),
            coordinate_system="geodetic", si_units=True), dtype=np.float64)
        if not np.all(np.isfinite(gravity)):
            raise ValueError(f"non-finite reference values in {name}")
        np.savez_compressed(OUT / f"{name}.npz", latitude=latitude, height=height, gravity=gravity)
        ranges[name] = {"shape": list(latitude.shape), "latitude_degrees": [float(latitude.min()), float(latitude.max())],
                        "height_m": [float(height.min()), float(height.max())],
                        "gravity_m_per_s2": [float(gravity.min()), float(gravity.max())]}
    surface = np.load(OUT / "surface.npz")
    assert abs(float(surface["gravity"][6]) - 9.7803253359) < tolerances.SURFACE_SANITY_ABSOLUTE
    assert abs(float(surface["gravity"][-1]) - 9.8321849379) < tolerances.SURFACE_SANITY_ABSOLUTE
    assert np.all(np.diff(surface["gravity"][6:]) > 0)
    near = np.load(OUT / "near_earth.npz")
    assert near["gravity"][2, 1] < near["gravity"][2, 0]
    meta = {
        "reference": "Boule WGS84.normal_gravity, geodetic, si_units=True",
        "versions": {"boule": boule.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
        "seed": SEED,
        "inputs": "Arrays are stored in each npz as float32, promoted to float64 before evaluating Boule; latitude degrees and nonnegative height meters. Gravity is float64 m/s^2.",
        "cases": ranges,
        "tolerances": {
            "gravity_relative": {"value": tolerances.GRAVITY_RELATIVE, "reason": "Project-wide float32 pointwise relative error limit against float64 reference."},
            "surface_sanity_absolute_m_per_s2": {"value": tolerances.SURFACE_SANITY_ABSOLUTE, "reason": "Sanity check against rounded WGS84 values; not an implementation tolerance."},
        },
        "reference_caveat": "Boule warns that negative heights are outside its validity. Its implementation computes a signed ellipsoidal-harmonic u component, despite documenting a magnitude; high altitudes may expose float64 cancellation in its auxiliary function.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(ranges, indent=2))


if __name__ == "__main__":
    main()
