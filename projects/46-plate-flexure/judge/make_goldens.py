"""Generate analytic Euler-Bernoulli plate-on-Winkler-foundation references."""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "golden"


def solution(x, rigidity, foundation, load, plate):
    """Solve D w'''' + k w = P delta(x), or the free-edge counterpart.

    alpha=(4D/k)**(1/4). On an infinite plate symmetry and the shear
    jump D*(w'''(0+) - w'''(0-))=P give
    P alpha**3/(8D) exp(-u)(cos(u)+sin(u)). On a half-line,
    zero edge moment and D w'''(0)=P give P alpha**3/(2D)
    exp(-u) cos(u), with u=|x|/alpha or x/alpha respectively.
    """
    alpha = (4.0 * rigidity / foundation) ** 0.25
    u = np.abs(x) / alpha
    if plate == "continuous":
        return load * alpha**3 / (8.0 * rigidity) * np.exp(-u) * (np.cos(u) + np.sin(u))
    return load * alpha**3 / (2.0 * rigidity) * np.exp(-u) * np.cos(u)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Evaluate the float64 reference on inputs rounded to their wire values.
    configs = [
        ("continuous", 1.0e23, 32000.0, 1.0e12),
        ("broken", 1.0e23, 32000.0, 1.0e12),
        ("continuous", 7.3e21, 17500.0, -3.7e11),
        ("broken", 2.4e24, 51000.0, -8.2e11),
        ("continuous", 1.0e23, 32000.0, 0.0),
        ("broken", 1.0e23, 32000.0, 0.0),
    ]
    inputs = {}
    for i, (plate, d, k, p) in enumerate(configs):
        alpha = (4 * d / k) ** 0.25
        u = np.array([0, .05, .25, .5, 1, 1.5, np.pi/2, 2, 2.35,
                      3, 3*np.pi/2, 5, 7, 9], dtype=np.float64)
        if plate == "continuous":
            u = np.concatenate((-u[:0:-1], u))
        x = (u * alpha).astype(np.float32)
        d, k, p = (float(np.float32(v)) for v in (d, k, p))
        w = solution(x.astype(np.float64), d, k, p, plate)
        np.savez(OUT / f"case_{i}.npz", x=x, rigidity=np.array(d),
                 foundation=np.array(k), load=np.array(p), deflection=w)
        inputs[f"case_{i}"] = {"plate": plate, "rigidity_N_m": d,
                              "foundation_N_m3": k, "load_N_m": p,
                              "x_m": x.astype(float).tolist(),
                              "deflection_range_m": [float(w.min()), float(w.max())]}
        a = (4 * d / k) ** .25
        assert np.all(np.isfinite(w))
        if p:
            expected = p * a**3 / ((8 if plate == "continuous" else 2) * d)
            assert np.isclose(w[np.argmin(np.abs(x))], expected, rtol=1e-14)
            assert np.sign(expected) == np.sign(p)
        print(f"case_{i} {plate}: alpha={a:.6g} m, w=[{w.min():.6g}, {w.max():.6g}] m")
    meta = {
        "model": "Euler-Bernoulli elastic plate on a linear Winkler foundation; downward positive",
        "analytic_reference": "D w'''' + k w = P delta(x), with D*(w'''(0+) - w'''(0-))=P on the continuous plate. For the broken plate, x increases into the plate, w''(0)=0 and D w'''(0)=P, with the full P applied at the edge. Decay at infinity.",
        "seed": "none; deterministic fixed coordinates",
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "inputs": inputs,
        "tolerances": {
            "relative_max_norm": {"value": 1e-4, "reason": "project-wide float32 analytic parity threshold; normalized by peak reference magnitude to handle nodes"},
            "zero_absolute_m": {"value": 1e-7, "reason": "absolute tolerance for zero load, where relative error is undefined"},
            "cross_path_relative_max_norm": {"value": 1e-4, "reason": "project-wide float32 cross-path parity"},
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
