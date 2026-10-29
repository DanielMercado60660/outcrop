"""Generate deterministic homogeneous-medium double-difference examples."""
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent / "golden"


def make_case(seed, noise):
    rng = np.random.default_rng(seed)
    n = 12
    # Local Cartesian kilometres, depth positive. The true fault is y = 0.
    truth = np.column_stack((np.linspace(-3, 3, n), np.zeros(n),
                             5 + 0.25 * np.linspace(-3, 3, n)))
    origins = rng.uniform(-0.4, 0.4, n)
    stations = np.array([[-8, -7, 0], [-8, 6, 0], [0, -9, 0],
                         [1, 8, 0], [8, -6, 0], [9, 7, 0],
                         [-4, 2, 0], [5, -2, 0]], dtype=np.float64)
    velocity = 6.0
    initial = truth + rng.normal(0, [0.18, 0.8, 0.15], (n, 3))
    initial_origins = origins + rng.normal(0, 0.035, n)
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    arrivals = origins[:, None] + np.linalg.norm(truth[:, None] - stations[None], axis=2) / velocity
    # Independent differential-time noise (seconds), one measurement per pair/station.
    observed = arrivals[pairs[:, 0]] - arrivals[pairs[:, 1]]
    observed += rng.normal(0, noise, observed.shape)
    damping = 1e-4
    x0 = np.column_stack((initial, initial_origins)).ravel()

    def residual(x):
        events = x.reshape(n, 4)
        t = events[:, 3, None] + np.linalg.norm(
            events[:, None, :3] - stations[None], axis=2) / velocity
        dd = t[pairs[:, 0]] - t[pairs[:, 1]] - observed
        delta = (x - x0).reshape(n, 4).copy()
        delta[:, :3] /= velocity
        return np.concatenate((dd.ravel(), np.sqrt(damping) * delta.ravel()))

    result = least_squares(residual, x0, method="trf", ftol=1e-13,
                           xtol=1e-13, gtol=1e-13, max_nfev=1000)
    if not result.success or np.linalg.norm(result.grad, ord=np.inf) > 1e-8:
        raise RuntimeError(f"oracle did not converge: {result.message}")
    fit = result.x.reshape(n, 4)
    before, after = np.std(initial[:, 1]), np.std(fit[:, 1])
    assert after < 0.5 * before
    print(f"seed={seed}, noise={noise:g} s: spread {before:.6g} -> {after:.6g} km; "
          f"RMS DD {np.sqrt(np.mean(residual(result.x)[:observed.size]**2)):.6g} s; "
          f"optimality {np.linalg.norm(result.grad, ord=np.inf):.3g}")
    return {"stations": stations, "initial_locations": initial, "initial_origins": initial_origins,
            "pairs": pairs, "observed": observed, "velocity": np.array(velocity),
            "damping": np.array(damping), "truth": truth, "true_origins": origins,
            "locations": fit[:, :3], "origins": fit[:, 3],
            "residuals": residual(result.x)[:observed.size].reshape(observed.shape)}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, seed, noise in [("clean", 3101, 0.0), ("noisy", 3102, 0.002)]:
        np.savez_compressed(ROOT / f"{name}.npz", **make_case(seed, noise))
    meta = {
        "model": "straight-ray travel time origin + Euclidean distance / 6 km/s; Cartesian km, positive depth",
        "cases": {"clean": {"seed": 3101, "noise_std_s": 0.0},
                  "noisy": {"seed": 3102, "noise_std_s": 0.002}},
        "inputs": "12 events, 8 stations, every unordered event pair at every station, damping 0.0001",
        "objective": "sum((predicted DD - observed DD)^2) + damping * sum((delta_xyz / velocity)^2 + delta_origin^2)",
        "reference": "scipy.optimize.least_squares TRF, numerical Jacobian, float64",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "tolerances": {
            "solution_scaled_max_error": "1e-4 relative to max reference magnitude; project-wide float32 limit",
            "residual_absolute_s": "2e-5 s; float32 subtraction of arrival times and iterative solver error",
            "spread_ratio": "0.5 maximum, the project requirement",
            "cross_path": "not applicable: MLX-only project"}}
    (ROOT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
