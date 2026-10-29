"""Generate deterministic analytic ray goldens independently of implementations."""
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent / "golden"


def layered(p, interfaces, velocities, depth):
    edges = np.r_[0.0, interfaces[interfaces < depth], depth]
    h = np.diff(edges)
    v = velocities[:len(h)]
    q = p * v
    return np.sum(h * q / np.sqrt(1 - q*q)), np.sum(h / (v * np.sqrt(1 - q*q)))


def smooth(p, v0, gradient, depth):
    # Snell invariant p=sin(theta)/v; ds=dz/cos(theta).
    def v(z):
        return v0 + gradient * z
    x = quad(lambda z: p*v(z)/math.sqrt(1-(p*v(z))**2), 0, depth,
             epsabs=1e-10, epsrel=1e-12)[0]
    t = quad(lambda z: 1/(v(z)*math.sqrt(1-(p*v(z))**2)), 0, depth,
             epsabs=1e-12, epsrel=1e-12)[0]
    return x, t


def build(name, kind, params, angles, offsets, depth):
    angles = np.asarray(angles, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    depth = float(np.float32(depth))
    if kind == "layered":
        interfaces = np.asarray(params[0], dtype=np.float32)
        velocities = np.asarray(params[1], dtype=np.float32)
        i64, v64 = interfaces.astype(np.float64), velocities.astype(np.float64)
        def forward(p):
            return layered(p, i64, v64, depth)
        v0 = float(v64[0])
        vmax = float(np.max(v64[:np.count_nonzero(i64 < depth)+1]))
        inputs = {"interfaces": interfaces, "velocities": velocities}
    else:
        v0, grad = (float(np.float32(x)) for x in params)
        def forward(p):
            return smooth(p, v0, grad, depth)
        vmax = max(v0, v0 + grad*depth)
        inputs = {"v0": np.array(v0), "gradient": np.array(grad)}
    shot = np.array([forward(math.sin(float(a))/v0) for a in angles])
    limit = 0.999/vmax
    ps = np.array([brentq(lambda p, target=float(x): forward(p)[0]-target, -limit, limit,
                          xtol=1e-16) for x in offsets])
    bent = np.array([forward(p) for p in ps])
    # Independently check the circular-ray geometry for nonzero gradients.
    if kind == "gradient" and grad != 0:
        for angle, (x, _) in zip(angles, shot):
            p = math.sin(float(angle))/v0
            if abs(p) > 1e-12:
                center_x = math.cos(float(angle))/(p*grad)
                center_z = -v0/grad
                radius2 = center_x**2 + center_z**2
                assert abs((x-center_x)**2 + (depth-center_z)**2 - radius2) < 1e-8*radius2
    assert np.all(shot[:, 1] >= depth / vmax)
    assert np.all(bent[:, 1] >= depth / vmax)
    assert np.max(np.abs(bent[:, 0]-offsets)) < 1e-6
    np.savez_compressed(
        OUT / (name + ".npz"), **inputs, depth=np.array(depth),
        angles=angles, offsets=offsets,
        shot_endpoints=np.column_stack((shot[:, 0], np.full(len(angles), depth))),
        shot_times=shot[:, 1],
        bent_endpoints=np.column_stack((bent[:, 0], np.full(len(offsets), depth))),
        bent_times=bent[:, 1], bent_angles=np.arcsin(ps*v0))
    print(f"{name}: shot x {shot[:, 0].min():.3f}..{shot[:, 0].max():.3f} m, "
          f"times {shot[:, 1].min():.6f}..{shot[:, 1].max():.6f} s; "
          f"bent times {bent[:, 1].min():.6f}..{bent[:, 1].max():.6f} s")


def main():
    OUT.mkdir(exist_ok=True)
    cases = [
        ("layered_multi", "layered", ([350, 850], [1800, 2400, 2000]), [-0.55, -0.2, 0, 0.2, 0.55], [-1300, -450, 0, 450, 1300], 1300),
        ("layered_homogeneous", "layered", ([], [2200]), [-0.7, 0, 0.7], [-1000, 0, 1000], 900),
        ("layered_partial", "layered", ([400, 1200], [1800, 2600, 2100]), [-0.45, 0, 0.45], [-700, 0, 700], 700),
        ("gradient_positive", "gradient", (1800, 0.6), [-0.65, -0.2, 0, 0.2, 0.65], [-1300, -400, 0, 400, 1300], 1200),
        ("gradient_negative", "gradient", (2400, -0.5), [-0.7, 0, 0.7], [-1100, 0, 1100], 1000),
        ("gradient_zero", "gradient", (2000, 0), [-0.6, 0, 0.6], [-900, 0, 900], 1000),
        ("gradient_shallow", "gradient", (1800, 0.6), [-0.65, 0, 0.65], [-500, 0, 500], 500),
    ]
    for case in cases:
        build(*case)
    meta = {
        "method": "Snell horizontal slowness; float64 depth integrals with scipy quad and brentq; circle identity checked independently",
        "seed": "none: explicit deterministic inputs",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "cases": [{"name": n, "kind": k, "params": p, "angles_rad": a,
                   "offsets_m": o, "depth_m": d} for n,k,p,a,o,d in cases],
        "tolerances": {
            "endpoint_relative": {"value": 1e-4, "reason": "project-wide float32 limit, Euclidean error divided by receiver distance"},
            "time_relative": {"value": 1e-4, "reason": "project-wide float32 limit per ray"},
            "angle_scaled": {"value": 1e-4, "reason": "one radian floor for vertical rays"},
            "cross_factor": {"value": 2.0, "reason": "sum of independent reference error bounds"},
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
