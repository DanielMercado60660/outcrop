"""Deterministic homogeneous earthquake oracle; only judge imports SciPy.

Analytic Euclidean travel times, profiled weighted grid search, and independent
SciPy trust-region least_squares solutions provide float64 reference results.
Run from the project directory with the repository's pinned Python interpreter.
"""
from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "judge" / "golden"
SEED = 30012026
spec = importlib.util.spec_from_file_location("tolerances", ROOT / "tests/tolerances.py")
tol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tol)


def forward(stations, models, velocity):
    delta = models[:, None, :3] - stations[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    predicted = models[:, 3, None] + distance / velocity
    jac = np.concatenate((delta / (velocity * distance[:, :, None]),
                          np.ones((*distance.shape, 1))), axis=2)
    return predicted, jac


def evaluate(stations, arrivals, sigma, velocity, models):
    predicted, jac = forward(stations, models, velocity)
    residual = (predicted - arrivals) / sigma
    return {"predicted": predicted, "objective": 0.5 * np.sum(residual**2, axis=1),
            "gradient": np.einsum("bni,bn->bi", jac / sigma[None, :, None], residual)}


def solve(stations, arrivals, sigma, velocity, initial):
    models, cov = [], []
    for picks, start in zip(arrivals, initial):
        def fun(model, picks=picks):
            return (forward(stations, model[None], velocity)[0][0] - picks) / sigma

        def jac(model):
            return forward(stations, model[None], velocity)[1][0] / sigma[:, None]

        fit = least_squares(fun, start, jac=jac, x_scale=[5000, 5000, 5000, 1],
                            ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=300)
        assert fit.success, fit.message
        models.append(fit.x)
        cov.append(np.linalg.inv(jac(fit.x).T @ jac(fit.x)))
    models, cov = np.array(models), np.array(cov)
    return {"models": models, "covariance": cov,
            "objective": evaluate(stations, arrivals, sigma, velocity, models)["objective"],
            "ellipse": chi2.ppf(0.95, 2) * cov[:, :2, :2]}


def grid(stations, arrivals, sigma, velocity, axes):
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    travel = np.linalg.norm(nodes[:, None] - stations[None], axis=2) / velocity
    weight = 1 / sigma**2
    models, losses = [], []
    for picks in arrivals:
        origins = np.sum((picks - travel) * weight, axis=1) / weight.sum()
        losses_all = 0.5 * np.sum(((travel + origins[:, None] - picks) / sigma)**2, axis=1)
        index = np.argmin(losses_all)
        models.append(np.r_[nodes[index], origins[index]])
        losses.append(losses_all[index])
    return {"models": np.array(models), "objective": np.array(losses)}


def save(name, inputs, **outputs):
    np.savez_compressed(GOLDEN / f"{name}.npz", **inputs, **outputs)


def main():
    GOLDEN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # Three rings plus an asymmetric interior array resolve depth/time tradeoff.
    az = np.arange(16) * (2 * np.pi / 16)
    radius = np.array([9000, 16000, 23000, 16000] * 4)
    stations = np.column_stack((radius * np.cos(az), radius * np.sin(az),
                               -rng.uniform(0, 450, 16))).astype(np.float32)
    sigma = np.linspace(0.015, 0.035, 16).astype(np.float32)
    velocity = 6000.0
    base = {"stations": stations, "sigma": sigma, "velocity": np.array(velocity)}
    truth = np.array([[0, 0, 7000, 0.25], [3000, -2000, 9000, -0.6],
                      [-2470, 1650, 5320, 1.2], [4210, 3320, 8760, -1.5],
                      [-3800, -4200, 10250, 0.8], [1750, -3670, 6150, 2.1]])
    clean = forward(stations.astype(float), truth, velocity)[0].astype(np.float32)
    axes = [np.arange(-6000, 6001, 1000, dtype=np.float32)] * 2
    axes += [np.arange(3000, 12001, 1000, dtype=np.float32)]
    inputs = dict(base, arrivals=clean, x=axes[0], y=axes[1], z=axes[2])
    grid_out = grid(stations.astype(float), clean.astype(float), sigma.astype(float),
                    velocity, axes)
    save("grid", inputs, truth=truth, **grid_out)
    clean_inputs = dict(base, arrivals=clean, initial=grid_out["models"].astype(np.float32))
    clean_out = solve(stations.astype(float), clean.astype(float), sigma.astype(float),
                      velocity, clean_inputs["initial"].astype(float))
    save("clean", clean_inputs, truth=truth, **clean_out)
    noisy_truth = np.column_stack((rng.uniform(-4500, 4500, (tol.NOISY_TRIALS, 2)),
                                  rng.uniform(4500, 10500, tol.NOISY_TRIALS),
                                  rng.uniform(-2, 2, tol.NOISY_TRIALS)))
    noise = rng.normal(size=(tol.NOISY_TRIALS, len(stations))) * sigma
    noisy = (forward(stations.astype(float), noisy_truth, velocity)[0] + noise).astype(np.float32)
    initial = (noisy_truth + [1500, -1300, 900, 0.2]).astype(np.float32)
    noisy_inputs = dict(base, arrivals=noisy, initial=initial)
    noisy_out = solve(stations.astype(float), noisy.astype(float), sigma.astype(float),
                      velocity, initial.astype(float))
    save("noisy", noisy_inputs, truth=noisy_truth, **noisy_out)
    probes = (truth[:3] + [800, -600, 500, 0.12]).astype(np.float32)
    probe_inputs = dict(base, arrivals=clean[:3], models=probes)
    probe_out = evaluate(stations.astype(float), clean[:3].astype(float),
                         sigma.astype(float), velocity, probes.astype(float))
    fd = np.empty_like(probe_out["gradient"])
    for k, h in enumerate(tol.ORACLE_FD_STEPS):
        plus, minus = probes.astype(float), probes.astype(float)
        plus[:, k] += h
        minus[:, k] -= h
        fd[:, k] = (evaluate(stations, clean[:3], sigma, velocity, plus)["objective"]
                    - evaluate(stations, clean[:3], sigma, velocity, minus)["objective"]) / (2*h)
    fd_error = np.max(np.abs(fd - probe_out["gradient"]), axis=0) / np.max(np.abs(fd), axis=0)
    assert np.max(fd_error) <= tol.ORACLE_GRADIENT_RTOL
    # Check step selection using float32 forward/loss arithmetic as well.
    fd32 = np.empty_like(fd)
    for k, h in enumerate(tol.FD_STEPS):
        plus, minus = probes.copy(), probes.copy()
        plus[:, k] += h
        minus[:, k] -= h
        a = evaluate(stations, clean[:3], sigma, velocity, plus)["objective"].astype(float)
        b = evaluate(stations, clean[:3], sigma, velocity, minus)["objective"].astype(float)
        fd32[:, k] = (a - b) / (plus[:, k].astype(float) - minus[:, k].astype(float))
    fd32_error = np.max(np.abs(fd32 - probe_out["gradient"]), axis=0) / np.max(np.abs(fd32), axis=0)
    assert np.max(fd32_error) <= tol.GRADIENT_RTOL
    save("gradient", probe_inputs, **{f"expected_{k}": v for k, v in probe_out.items()})
    dxy = noisy_out["models"][:, :2] - noisy_truth[:, :2]
    mahal = np.einsum("bi,bij,bj->b", dxy, np.linalg.inv(noisy_out["ellipse"]), dxy)
    coverage = np.mean(mahal <= 1)
    grid_error = np.linalg.norm((grid_out["models"][:, :3] - truth[:, :3]) / 1000, axis=1)
    clean_error = np.linalg.norm(clean_out["models"][:, :3] - truth[:, :3], axis=1)
    assert np.max(grid_error) <= tol.GRID_MAX_CELLS
    assert np.max(clean_error) <= tol.GEIGER_MAX_POSITION_M
    assert np.max(np.abs(clean_out["models"][:, 3] - truth[:, 3])) <= tol.GEIGER_MAX_TIME_S
    assert tol.COVERAGE_MIN <= coverage <= tol.COVERAGE_MAX
    assert np.all(np.linalg.eigvalsh(noisy_out["covariance"]) > 0)
    assert abs(chi2.ppf(0.95, 2) - tol.ELLIPSE_CHI2_95) < tol.ORACLE_QUANTILE_ATOL
    semiaxes = np.sqrt(np.linalg.eigvalsh(noisy_out["ellipse"]))
    summary = {"grid_max_error_cells": float(grid_error.max()),
               "clean_max_error_m": float(clean_error.max()),
               "clean_max_time_error_s": float(np.max(np.abs(clean_out["models"][:, 3] - truth[:, 3]))),
               "coverage": float(coverage), "covered": int(np.sum(mahal <= 1)),
               "arrivals_range_s": [float(clean.min()), float(clean.max())],
               "noisy_loss_range": [float(noisy_out["objective"].min()), float(noisy_out["objective"].max())],
               "ellipse_semiaxis_range_m": [float(semiaxes.min()), float(semiaxes.max())],
               "oracle_gradient_relative_error": fd_error.tolist(),
               "float32_fd_step_check_relative_error": fd32_error.tolist()}
    metadata = {"seed": SEED, "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                    "reference": "Analytic t=t0+Euclidean distance/v; scipy.optimize.least_squares (TRF); scipy.stats.chi2",
                    "model": "Homogeneous 6000 m/s; local EN-down metres, relative seconds; known heteroscedastic independent pick sigma",
                    "precision": "Inputs quantized once to float32 and promoted to float64 for oracle computation; reference outputs float64",
                    "inputs": {"grid": "6 events, 16 stations, 13x13x10 grid, 1000 m cells",
                            "clean": "same 6 events; initial models from exhaustive grid",
                            "noisy": "200 independent Gaussian pick-noise trials, varied truth; fixed perturbed starts",
                            "gradient": "3 off-solution models; central difference steps recorded in tolerances.py"},
                    "ellipse": "Horizontal marginal covariance including depth and origin-time nuisance parameters; known sigma, no residual scaling",
                    "tolerances": {k: v for k, v in vars(tol).items() if k.isupper()},
                    "tolerance_reasons": (ROOT / "tests/tolerances.py").read_text(), "sanity": summary}
    (GOLDEN / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
