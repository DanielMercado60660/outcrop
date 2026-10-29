"""Generate float64 scikit-fmm and analytic eikonal reference cases."""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from runpy import run_path

import numpy as np
import skfmm
from documented_rules import travel_time as documented_travel_time
from second_order import travel_time as second_order_travel_time

OUT = Path(__file__).resolve().parent / "golden"
TOL = run_path(str(OUT.parent.parent / "tests" / "tolerances.py"))
SELF_CHECKS = {}


def save(name, phi, speed, spacing, **extra):
    phi = np.asarray(phi, dtype=np.float32)
    speed = np.asarray(speed, dtype=np.float32)
    # Reference exactly the inputs implementations receive, evaluated in float64.
    phi64, speed64 = phi.astype(np.float64), speed.astype(np.float64)
    reference = {
        "time_order1": np.asarray(skfmm.travel_time(phi64, speed64, dx=spacing, order=1)),
        "time_order2": second_order_travel_time(phi64, speed64, spacing),
    }
    skfmm2 = np.asarray(skfmm.travel_time(phi64, speed64, dx=spacing, order=2))
    discrepancy = float(np.max(np.abs(reference["time_order2"] - skfmm2)) / np.max(skfmm2))
    print(name, "order-2 scikit-fmm normalized difference", discrepancy, flush=True)
    assert discrepancy <= TOL["ORDER2_SKFMM_SANITY_REL"]
    assert all(np.isfinite(t).all() and np.min(t) >= 0 for t in reference.values())
    # Independent transcription of the API, for BOTH orders on EVERY case.
    # Order 1 is checked against scikit-fmm, order 2 against second_order.py.
    checks = {}
    for order in (1, 2):
        documented = documented_travel_time(phi64, speed64, spacing, order)
        difference = float(np.max(np.abs(documented - reference[f"time_order{order}"])))
        checks[f"order{order}_max_abs_error"] = difference
        print(name, f"documented order-{order} max absolute error", difference, flush=True)
        assert difference <= TOL["DOCUMENTED_RULES_ATOL"], (name, order, difference)
        assert np.all(reference[f"time_order{order}"][phi == 0] == 0)
    SELF_CHECKS[name] = checks
    if "analytic" in extra and name.startswith("homogeneous_"):
        error = float(np.max(np.abs(reference["time_order1"] - extra["analytic"])))
        assert error / np.max(extra["analytic"]) <= TOL["PARITY_REL"]
    np.savez_compressed(OUT / f"{name}.npz", phi=phi, speed=speed,
                        spacing=np.asarray(spacing, dtype=np.float64),
                        time_skfmm_order2=skfmm2, **reference, **extra)
    return {k: [float(v.min()), float(v.max())] for k, v in reference.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(24001)
    cases = {}
    for dim, shape, spacing in [(2, (37, 31), [0.08, 0.11]),
                                 (3, (21, 19, 17), [0.09, 0.12, 0.15])]:
        axes = [np.arange(n) * dx for n, dx in zip(shape, spacing)]
        grid = np.meshgrid(*axes, indexing="ij")
        center = [axis[-1] * f for axis, f in zip(axes, [0.42, 0.55, 0.48])]
        phi = np.sqrt(sum((x - c)**2 for x, c in zip(grid, center))) - 0.34
        speed = 2.0 + sum(0.13 * np.sin((i + 1) * x) for i, x in enumerate(grid))
        speed += rng.uniform(-0.04, 0.04, size=shape)
        name = f"heterogeneous_{dim}d"
        cases[name] = {"shape": shape, "spacing": spacing, "speed_range":
                       [float(speed.min()), float(speed.max())],
                       "time_ranges": save(name, phi, speed, spacing)}

    # Planar source x=0, v(x)=2+0.8*x; ray paths are normal to the plane.
    # Nodes lie exactly on the interface and include both sides.
    for dim in (2, 3):
        for n in (20, 40, 80):
            h = 1.0 / n
            x = np.arange(-n // 4, n + 1, dtype=np.float64) * h
            shape = (len(x),) + (5,) * (dim - 1)
            phi = np.broadcast_to(x.reshape((-1,) + (1,) * (dim - 1)), shape)
            speed = 2.0 + 0.8 * phi
            analytic = np.abs(np.log(speed / 2.0) / 0.8)
            name = f"gradient_{dim}d_{n}"
            ranges = save(name, phi, speed, [h] * dim, analytic=analytic)
            cases[name] = {"shape": shape, "spacing": [h] * dim,
                           "time_ranges": ranges}

    # A planar source in homogeneous medium gives an exact distance solution,
    # including on each side of the interface, for both dimensions and orders.
    for dim in (2, 3):
        h = 0.1
        x = (np.arange(17, dtype=np.float64) - 6.35) * h
        shape = (17,) + (7,) * (dim - 1)
        phi = np.broadcast_to(x.reshape((-1,) + (1,) * (dim - 1)), shape)
        speed = np.full(shape, 2.5)
        name = f"homogeneous_{dim}d"
        ranges = save(name, phi, speed, [h] * dim, analytic=np.abs(phi) / 2.5)
        cases[name] = {"shape": shape, "spacing": [h] * dim,
                       "time_ranges": ranges}

    # A circular/spherical wavefront exercises coupled directional updates.
    # Compare outside the source sphere, excluding its immediate grid layer.
    for dim in (2, 3):
        for n in (20, 40, 80):
            h = 2.0 / n
            axis = np.linspace(-1.0, 1.0, n + 1)
            grid = np.meshgrid(*([axis] * dim), indexing="ij")
            radius = np.sqrt(sum(x * x for x in grid))
            phi = radius - 0.3
            speed = np.full(phi.shape, 2.5)
            analytic = np.maximum(phi, 0.0) / 2.5
            mask = radius >= 0.5
            name = f"radial_{dim}d_{n}"
            ranges = save(name, phi, speed, [h] * dim, analytic=analytic, mask=mask)
            cases[name] = {"shape": list(phi.shape), "spacing": [h] * dim,
                           "time_ranges": ranges}

    meta = {"seed": 24001, "numpy": version("numpy"),
            "scikit-fmm": version("scikit-fmm"), "cases": cases,
            "documented_rules_self_checks": SELF_CHECKS,
            "initialization": "Order 1 uses strict sign changes only; exact phi zeros freeze at time 0, but zero neighbors do not count as crossings. Order 2 retains edges to zero neighbors as crossings. Both use node-local speed and inverse-square combination across crossing axes.",
            "order1_update": "All axes with frozen immediate neighbors contribute to the larger quadratic root; no Godunov dimension dropping. See tests/api.py and independent judge/documented_rules.py.",
            "reference": "Order 1: skfmm.travel_time. Order 2: judge/second_order.py published upwind FMM variant; compared to skfmm order 2 as a loose sanity check. Float64 from float32 inputs; nonperiodic, unmasked.",
            "input_construction": "All arrays and spacing are saved in each npz. Heterogeneous: spherical signed distance with radius 0.34 and off-center origin; speed = 2 + sum_i 0.13 sin((i+1)*x_i) + uniform(-0.04,0.04), seed 24001. Gradient: x from -0.25 to 1, phi=x, speed=2+0.8*x. Homogeneous: phi=x with interface 0.035 past node 6, speed=2.5. Radial: axes span [-1,1], phi=r-0.3, speed=2.5; compare analytic outside r>=0.5.",
            "analytic": "planar source: |x|/v for constant v; |log((2+0.8*x)/2)/0.8| for v=2+0.8*x; radial: (r-0.3)/2.5 outside the sphere",
            "tolerances": {"documented_rules_atol": [TOL["DOCUMENTED_RULES_ATOL"], "judge-only absolute time error; independent float64 API transcription must reproduce every golden at both orders, allowing accumulated float64 roundoff"],
                           "order2_skfmm_sanity_rel": [TOL["ORDER2_SKFMM_SANITY_REL"], "different second-order interface, stencil and trial update rules; sanity check only"],
                           "parity_rel": [TOL["PARITY_REL"], "max absolute error / max reference time; project-wide float32 threshold"],
                           "cross_rel": [TOL["CROSS_REL"], "same scaling for float32 path agreement"],
                           "refinement_ratio": [TOL["REFINEMENT_RATIO"], "each grid halving must shrink first-order error, allowing pre-asymptotic effects"],
                           "min_coarse_error": [TOL["MIN_COARSE_ERROR"], "avoid degenerate refinement ratios"]}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, case in cases.items():
        print(name, case["time_ranges"])
    for kind in ("gradient", "radial"):
        for dim in (2, 3):
            errors = []
            for n in (20, 40, 80):
                with np.load(OUT / f"{kind}_{dim}d_{n}.npz") as data:
                    mask = data["mask"] if "mask" in data else np.ones(data["phi"].shape, dtype=bool)
                    errors.append(float(np.max(np.abs(data["time_order1"][mask] - data["analytic"][mask]))))
            ratios = [errors[1] / errors[0], errors[2] / errors[1]]
            print(kind, dim, "analytic max errors", errors, "ratios", ratios)
            assert errors[0] >= TOL["MIN_COARSE_ERROR"] and max(ratios) <= TOL["REFINEMENT_RATIO"]


if __name__ == "__main__":
    main()
