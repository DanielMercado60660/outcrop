"""Float64 synthetic MT oracle; run directly from the project directory.

Analytic quasistatic impedance recursion (e.g. Wait, Geo-Electromagnetism,
1982), with independent halfspace and two-layer reflection-form checks.
Occam principle: Constable, Parker & Constable (1987), Geophysics 52, 289-300,
doi:10.1190/1.1442303. SciPy SLSQP solves the explicitly defined constrained
problem; it is an oracle, not a prescribed implementation algorithm.
"""
from __future__ import annotations

import importlib.metadata
import json
import runpy
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[1]
T = runpy.run_path(str(ROOT / "tests" / "tolerances.py"))
MU0 = 4 * np.pi * 1e-7
SEED = 330101


def forward(model, thickness, frequencies):
    rho = 10.0 ** np.asarray(model, dtype=np.float64)
    omega = 2 * np.pi * np.asarray(frequencies, dtype=np.float64)
    intrinsic = np.sqrt(1j * omega[None, :] * MU0 * rho[:, None])
    z = intrinsic[-1].copy()
    for j in range(len(rho) - 2, -1, -1):
        u = np.tanh(np.sqrt(1j * omega * MU0 / rho[j]) * thickness[j])
        z = intrinsic[j] * (z + intrinsic[j] * u) / (intrinsic[j] + z * u)
    return np.array([np.log10(abs(z)**2 / (MU0 * omega)), np.angle(z)])


def jacobian(model, thickness, frequencies, step=None):
    step = T["JUDGE_FD_STEP"] if step is None else step
    eye = np.eye(len(model)) * step
    return np.stack([(forward(model + e, thickness, frequencies)
                      - forward(model - e, thickness, frequencies)) / (2 * step)
                     for e in eye], axis=-1)


def objective(model, thickness, frequencies, observed, sigma, alpha):
    residual = (forward(model, thickness, frequencies) - observed) / sigma
    diff = np.diff(model)
    value = 0.5 * np.mean(residual**2) + 0.5 * alpha * np.dot(diff, diff)
    gradient = np.einsum("ij,ijk->k", residual / sigma,
                         jacobian(model, thickness, frequencies)) / residual.size
    gradient[:-1] -= alpha * diff
    gradient[1:] += alpha * diff
    return value, gradient


def roughness(model):
    return 0.5 * np.sum(np.diff(model)**2)


def solve(initial, h, f, observed, sigma):
    def constraint(m):
        r = (forward(m, h, f) - observed) / sigma
        return T["TARGET_RMS"] - np.sqrt(np.mean(r**2))

    def constraint_jac(m):
        r = (forward(m, h, f) - observed) / sigma
        return -np.einsum("ij,ijk->k", r / sigma, jacobian(m, h, f)) / (
            r.size * np.sqrt(np.mean(r**2)))

    def rough_grad(m):
        g = np.zeros_like(m)
        g[:-1] -= np.diff(m)
        g[1:] += np.diff(m)
        return g

    feasible = least_squares(lambda m: ((forward(m, h, f)-observed)/sigma).ravel(),
                             initial, bounds=(T["MODEL_LOWER"], T["MODEL_UPPER"]),
                             xtol=T["ORACLE_LEAST_SQUARES_TOL"],
                             ftol=T["ORACLE_LEAST_SQUARES_TOL"],
                             gtol=T["ORACLE_LEAST_SQUARES_TOL"])
    assert constraint(feasible.x) > 0, "Synthetic target must be feasible"
    result = minimize(roughness, feasible.x, jac=rough_grad, method="SLSQP",
                      bounds=[(T["MODEL_LOWER"], T["MODEL_UPPER"])] * len(initial),
                      constraints={"type": "ineq", "fun": constraint,
                                   "jac": constraint_jac},
                      options={"ftol": T["ORACLE_CONSTRAINED_TOL"], "maxiter": 500})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, result.nit


def main():
    out = ROOT / "judge" / "golden"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    # Quantize all implementation inputs before the float64 reference evaluates
    # them: tests exercise the identical values even at high conductivity contrast.
    def quant(x):
        return np.asarray(x, np.float32).astype(np.float64)

    def save(name, **arrays):
        np.savez_compressed(out / f"{name}.npz", **arrays)

    summaries = {}
    cases = {
        "halfspace": ([2.0], [], [1e-4, 1., 1e4, .1, 1.]),
        "two_layer": ([1., 3.], [350.], np.logspace(-4, 4, 31)),
        "thin_conductor": ([3., -1., 2.5, 1.], [10., 2., 800.],
                           np.logspace(-4, 4, 31)[::-1]),
        "thick_resistor": ([0., 4., 1.], [20000., 50000.],
                           np.logspace(-4, 4, 31)),
    }
    for name, (model, h, f) in cases.items():
        model, h, f = map(quant, (model, h, f))
        response = forward(model, h, f)
        if name == "halfspace":
            np.testing.assert_allclose(response[0], model[0], rtol=T["ORACLE_RTOL"])
            np.testing.assert_allclose(response[1], np.pi / 4, rtol=T["ORACLE_RTOL"])
        if name == "two_layer":
            rho = 10**model
            omega = 2 * np.pi * f
            z1, z2 = [np.sqrt(1j * omega * MU0 * r) for r in rho]
            reflection = (z2-z1)/(z2+z1)
            decay = np.exp(-2 * np.sqrt(1j * omega * MU0/rho[0]) * h[0])
            z = z1 * (1 + reflection * decay) / (1 - reflection * decay)
            independent = [np.log10(abs(z)**2/(MU0*omega)), np.angle(z)]
            np.testing.assert_allclose(response, independent, rtol=T["ORACLE_RTOL"])
        assert np.all(np.isfinite(response))
        assert np.all((response[1] > 0) & (response[1] < np.pi/2))
        save(f"forward_{name}", model=model, thickness=h, frequencies=f, response=response)
        summaries[name] = {"rhoa_ohm_m": [float(x) for x in (10**response[0].min(),
                                                                 10**response[0].max())],
                           "phase_degrees": [float(x) for x in np.rad2deg(
                               [response[1].min(), response[1].max()])]}

    h = quant([40, 80, 160, 320, 640, 1280, 2560])
    f = quant(np.logspace(-3, 4, 33))
    models = {
        "conductor": [2.2, 2.1, 1.85, 1.6, 1.65, 1.9, 2.15, 2.3],
        "resistor": [1.7, 1.8, 2.05, 2.3, 2.25, 2., 1.75, 1.6],
        "constant": [2.] * 8,
    }
    for name, truth in models.items():
        truth = quant(truth)
        sigma = quant(np.tile([[.015], [.01]], (1, len(f))))
        clean = forward(truth, h, f)
        # Fixed RMS 0.8 noise guarantees feasibility at the truth; the quoted
        # sigma is conservative by 25%. This avoids an impossible target due
        # solely to the random chi-square realization on a small data vector.
        noise = rng.normal(size=clean.shape)
        noise *= 0.8 / np.sqrt(np.mean(noise**2))
        observed = quant(clean + noise * sigma)
        initial = quant(np.full(8, 2.65 if name != "resistor" else 1.2))
        recovered, iterations = solve(initial, h, f, observed, sigma)
        predicted = forward(recovered, h, f)
        rms = float(np.sqrt(np.mean(((predicted-observed)/sigma)**2)))
        # Fixed before judging: finite layers 0..6,
        # depths 0..5080 m. The halfspace is excluded from the recovery claim.
        resolved = np.arange(8) < 7
        error = float(np.max(abs(recovered[resolved]-truth[resolved])))
        assert rms <= T["TARGET_RMS"] + T["MISFIT_SLACK"]
        assert error <= T["MODEL_MAX_ERROR"]
        assert np.sqrt(np.mean(((forward(initial, h, f)-observed)/sigma)**2)) > 1
        save(f"inversion_{name}", truth=truth, initial_model=initial, thickness=h,
             frequencies=f, observed=observed, sigma=sigma, resolved=resolved,
             model=recovered, response=predicted, rms=rms, roughness=roughness(recovered))
        # Local linearized one-sigma resolution diagnostic, all 8 parameters
        # free (no regularization). This is evidence for the fixed depth mask,
        # not an adaptive mask chosen from the returned implementation model.
        weighted_j = (jacobian(truth, h, f) / sigma[..., None]).reshape(-1, len(truth))
        uncertainty = np.sqrt(np.diag(np.linalg.inv(weighted_j.T @ weighted_j)))
        summaries[f"inversion_{name}"] = {
            "rms": rms, "max_resolved_error": error,
            "roughness": float(roughness(recovered)), "oracle_iterations": iterations,
            "linearized_std_log10_rho": uncertainty.tolist(),
        }

    # Off-truth gradient case with unequal row/frequency weights, nonzero alpha,
    # and every component checked by saved centered perturbations.
    m = quant([2.1, 1.3, 2.5, 1.8])
    gh = quant([70, 250, 900])
    gf = quant([.003, .07, 2., 40., 1800.])
    obs = quant(forward(quant([2.3, 1.5, 2.2, 2.]), gh, gf))
    gs = quant([[.02, .025, .03, .04, .05], [.01, .012, .015, .02, .025]])
    for alpha in [0.0, 0.7]:
        val, grad = objective(m, gh, gf, obs, gs, alpha)
        pm = quant(m + np.eye(len(m)) * T["FD_STEP"])
        mm = quant(m - np.eye(len(m)) * T["FD_STEP"])
        plus = np.array([objective(x, gh, gf, obs, gs, alpha)[0] for x in pm])
        minus = np.array([objective(x, gh, gf, obs, gs, alpha)[0] for x in mm])
        fd = (plus-minus)/np.diag(pm-mm)
        assert np.linalg.norm(fd-grad)/np.linalg.norm(grad) <= T["GRADIENT_RTOL"]
        step = T["JUDGE_FD_STEP"] / 2
        eye = np.eye(len(m))*step
        finer = np.array([(objective(m+e, gh, gf, obs, gs, alpha)[0]
                           - objective(m-e, gh, gf, obs, gs, alpha)[0])/(2*step)
                          for e in eye])
        assert np.linalg.norm(finer-grad)/np.linalg.norm(grad) < T["JUDGE_FD_RTOL"]
        save(f"gradient_{str(alpha).replace('.', '_')}", model=m, thickness=gh,
             frequencies=gf, observed=obs, sigma=gs, alpha=alpha,
             objective=val, gradient=grad, plus_models=pm, minus_models=mm,
             plus_objectives=plus, minus_objectives=minus)

    # Exact-data zero residual/zero roughness case exercises stationary gradients.
    zm = quant([2., 2., 2.])
    zh, zf = quant([100., 400.]), quant([.01, 1., 100.])
    zo = quant(forward(zm, zh, zf))
    zs = quant(np.ones((2, 3)))
    zv, zg = objective(zm, zh, zf, zo, zs, .7)
    save("stationary", model=zm, thickness=zh, frequencies=zf, observed=zo,
         sigma=zs, alpha=.7, objective=zv, gradient=zg)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pytest")}
    meta = {
        "seed": SEED, "dtype": "float64; implementation inputs quantized to float32 first",
        "versions": versions, "reference": "analytic quasistatic impedance recursion; SciPy SLSQP",
        "conventions": "exp(+i omega t), quadrant I Z, top-down, mu0=4*pi*1e-7",
        "objective": "0.5*mean(((response-observed)/sigma)^2)+alpha*0.5*sum(diff(model)^2)",
        "noise": "seeded Gaussian normalized to weighted RMS=0.8; sigma conservative by 25%",
        "resolved_depth": "finite layers 0..6, 0..5080 m, all three inversion cases",
        "tolerance_source": "tests/tolerances.py",
        "tolerances": {k: v for k, v in T.items() if k.isupper()},
        "tolerance_reasons": (ROOT / "tests" / "tolerances.py").read_text(),
        "inputs": "All inputs, perturbations, uncertainties and truth are saved in each NPZ.",
        "sanity": summaries,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"versions": versions, "sanity": summaries}, indent=2))


if __name__ == "__main__":
    main()
