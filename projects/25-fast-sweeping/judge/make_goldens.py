"""Generate first-order scikit-fmm and analytic eikonal fixtures."""

import json
import platform
import runpy
from pathlib import Path

import numpy as np
import skfmm

HERE = Path(__file__).resolve().parent
OUT = HERE / "golden"


def make_case(dim, n, medium, shape=None):
    h = 2.0 / (n - 1)
    shape = shape or (n,) * dim
    axes = [-1.0 + h * np.arange(size, dtype=np.float64) for size in shape]
    coords = np.meshgrid(*axes, indexing="ij")
    source = [0.0] * dim
    if medium == "heterogeneous":
        source[0], source[1] = -0.4, 0.2
        if dim == 3:
            source[2] = -0.2
    radius = np.sqrt(sum((c - s) ** 2 for c, s in zip(coords, source, strict=True)))
    # A small sphere has its only negative sample at the central grid node.
    phi = radius - 0.01 * h
    if medium == "homogeneous":
        speed = np.full_like(radius, 2.0)
        analytic = radius / 2.0
    elif medium == "gradient":
        # v(x_0) = v0 + g*x_0. Exact point-source travel time.
        g, v0 = 0.6, 2.0
        speed = v0 + g * coords[0]
        analytic = np.arccosh(1 + g * g * radius**2 / (2 * v0 * speed)) / g
    elif medium == "heterogeneous":
        speed = 2.0 + 0.3 * np.sin(2.1 * coords[0]) * np.cos(1.7 * coords[1])
        if dim == 3:
            speed = speed + 0.2 * np.sin(1.4 * coords[2])
        analytic = np.zeros_like(speed)
    else:
        raise ValueError(medium)
    travel = np.asarray(skfmm.travel_time(phi, speed, dx=h, order=1))
    assert np.all(np.isfinite(travel)) and np.all(travel >= 0)
    assert speed.min() > 0 and travel.min() < h / speed.min()
    return {"phi": phi, "speed": speed, "travel": travel, "analytic": analytic,
            "radius": radius, "spacing": np.array(h, dtype=np.float64)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tol = runpy.run_path(str(HERE.parent / "tests" / "tolerances.py"))
    cases = {}
    fixtures = []
    for dim in (2, 3):
        for medium in ("homogeneous", "gradient", "heterogeneous"):
            sizes = (21, 41, 81) if medium != "heterogeneous" else (31,)
            for n in sizes:
                name = f"{medium}_{dim}d_{n}"
                fixtures.append((name, dim, n, medium, None))
    fixtures.extend([
        ("heterogeneous_2d_rect", 2, 21, "heterogeneous", (21, 27)),
        ("heterogeneous_3d_rect", 3, 21, "heterogeneous", (21, 25, 29)),
    ])
    for name, dim, n, medium, shape in fixtures:
        data = make_case(dim, n, medium, shape)
        np.savez_compressed(OUT / f"{name}.npz", **data)
        err = None
        if medium != "heterogeneous":
            mask = data["radius"] >= 0.25
            err = float(np.mean(np.abs(data["travel"][mask] - data["analytic"][mask])))
        cases[name] = {"shape": list(data["speed"].shape), "spacing_m": float(data["spacing"]),
                       "speed_range_m_per_s": [float(data["speed"].min()), float(data["speed"].max())],
                       "travel_range_s": [float(data["travel"].min()), float(data["travel"].max())],
                       "mean_analytic_error_s_radius_ge_0.25_m": err}
        print(name, cases[name])
    meta = {"generator": "judge/make_goldens.py", "seed": "none; deterministic analytic grids",
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-fmm": skfmm.__version__},
            "inputs": "Cartesian axes start at -1 m with scalar spacing; square and cubic grids span [-1,1] m, rectangular grids extend farther along longer axes. phi=distance from source minus 0.01 grid spacings. Analytic cases use the origin; heterogeneous cases use (-0.4, 0.2[, -0.2]) m. Homogeneous speed 2 m/s; gradient speed 2+0.6*x0 m/s; heterogeneous speed is documented in generator. skfmm.travel_time uses order=1 and scalar dx.",
            "analytic": "Homogeneous r/v; linear speed v0+g*x0: acosh(1+g^2*r^2/(2*v0*v))/g. Point-source comparison excludes r<0.25 m; sampled source sphere shrinks with h.",
            "tolerances": {"reference_relative_max": {"value": tol["REFERENCE_REL_MAX"], "reason": "Project-wide float32 same-discretization parity, normalized by maximum reference travel time."},
                           "min_observed_order": {"value": tol["MIN_OBSERVED_ORDER"], "reason": "Finite-grid allowance for point-source singularity and logarithmic corrections; the oracle's pairwise rates are approximately 0.67-0.72."},
                           "max_observed_order": {"value": tol["MAX_OBSERVED_ORDER"], "reason": "Finite-grid allowance around first order; guards accidental fixture or metric errors."}},
            "cases": cases}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
