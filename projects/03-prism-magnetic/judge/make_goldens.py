"""Generate deterministic float64 Harmonica prism-magnetic reference cases."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import harmonica as hm
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "golden"


def save(name, coordinates, prisms, magnetization, inducing_field):
    # Round inputs to exactly what both float32 implementations receive, then
    # evaluate the float64 oracle at those exact values.
    args = {k: np.asarray(v, dtype=np.float32) for k, v in locals().items() if k != "name"}
    xyz, p, m, direction = (args[k].astype(np.float64) for k in
                             ("coordinates", "prisms", "magnetization", "inducing_field"))
    b = np.stack(hm.prism_magnetic(xyz.T, p, m.T, field="b", parallel=False,
                                   dtype=np.float64), axis=-1)
    assert np.isfinite(b).all()
    np.savez(OUT / f"{name}.npz", **args, b_e=b[:, 0], b_n=b[:, 1],
             b_u=b[:, 2], total_field=b @ direction)
    return b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    prism = np.array([[-100, 100, -80, 80, -300, -100]])
    magnet = np.array([[1.25, -0.75, 2.0]])
    direction = np.array([0.36, 0.48, -0.8])
    xyz = np.column_stack((rng.uniform(-500, 500, 24), rng.uniform(-500, 500, 24),
                           rng.uniform(20, 250, 24)))
    b = save("single", xyz, prism, magnet, direction)
    assert 1 < np.max(np.abs(b)) < 1000
    # Integer offset exactly representable in float32. The same relative geometry
    # tests whether evaluation is truly source-local rather than origin-dependent.
    shift = np.array([100000, -200000, 50000])
    translated_prism = prism + np.repeat(shift, 2)
    bt = save("translated", xyz.astype(np.float32) + shift, translated_prism,
              magnet, direction)
    # Input quantization after translation changes locations by up to ~0.008 m.
    assert np.max(np.abs(bt - b)) / np.max(np.abs(b)) < 2e-4
    prisms = [[-250, -80, -160, 120, -450, -130],
              [40, 200, -100, 180, -320, -60],
              [-60, 80, 250, 420, -600, -210]]
    mags = [[1.3, -0.4, 0.7], [-0.8, 1.1, -1.6], [0.2, 0.9, 1.4]]
    save("multiple", xyz, prisms, mags, direction)
    # On the vertical axis of a centered prism with vertical magnetization,
    # horizontal components vanish and the upward field is positive.
    axis = [[0, 0, 0], [0, 0, 100], [0, 0, 300]]
    ba = save("axis", axis, prism, [[0, 0, 2]], [0, 0, 1])
    assert np.max(np.abs(ba[:, :2])) < 1e-10 and np.all(ba[:, 2] > 0)
    assert np.all(np.diff(ba[:, 2]) < 0)
    save("zero", xyz[:4], prism, [[0, 0, 0]], direction)
    # Distances are 8, 10 and 12 times the largest 200 m prism dimension,
    # from its center; use a vertical dipole axis for an independent check.
    far = np.array([[0, 0, -200 + 200 * d] for d in (8, 10, 12)] +
                   [[120 * d, 80 * d, -200 + 140 * d] for d in (8, 10, 12)])
    bf = save("far", far, prism, [[0, 0, 2]], [0, 0, 1])
    volume = 200 * 160 * 200
    r = far[:3, 2] + 200
    dipole_nt = 1e-7 * 2 * (2 * volume) / r**3 * 1e9
    assert np.max(np.abs(bf[:3, 2] / dipole_nt - 1)) < 0.01
    assert np.all(bf[:3, 2] > 0) and np.all(np.diff(bf[:3, 2]) < 0)
    meta = {
        "seed": 20261028,
        "versions": {k: version(k) for k in ("numpy", "harmonica", "choclo", "numba")},
        "python": platform.python_version(),
        "inputs": "All inputs are stored in each npz as float32; float64 Harmonica evaluates exactly those rounded values. ENU metres, A/m, unit inducing direction; outputs nT.",
        "cases": {"single": "24 seeded outside points, one asymmetric magnetized prism",
                  "translated": "same geometry translated by (100000,-200000,50000) m, with float32 quantization",
                  "multiple": "three unequal prisms, mixed magnetization, 24 points",
                  "axis": "three vertical-axis points, vertical magnetization",
                  "zero": "four points, zero magnetization",
                  "far": "axis and oblique points at approximately 8, 10, 12 maximum prism widths from center"},
        "tolerances": {"NEAR_REL": [1e-4, "project-wide float32 reference bound"],
                       "FAR_REL": [0.02, "proposed far-field float32 corner cancellation bound"],
                       "CROSS_NEAR_REL": [2e-4, "two near-field errors may have opposite signs"],
                       "CROSS_FAR_REL": [0.04, "two far-field errors may have opposite signs"],
                       "PROJECTION_REL": [2e-6, "float32 dot-product rounding"],
                       "ZERO_ABS_NT": [2e-5, "absolute floor for zero and cancellation"]},
        "dipole_axis_nT": dipole_nt.tolist(),
        "far_up_nT": bf[:, 2].tolist(),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("single maximum |B| nT:", np.max(np.abs(b)))
    print("axis up nT:", ba[:, 2])
    print("far up nT:", bf[:, 2], "dipole relative differences:", bf[:3, 2] / dipole_nt - 1)


if __name__ == "__main__":
    main()
