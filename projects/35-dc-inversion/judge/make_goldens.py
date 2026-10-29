"""Independent float64 SimPEG DC oracle. Run from the project with the repo venv.

No implementation is imported. Only this directory imports reference libraries.
All implementation-facing floating inputs are rounded to float32 then promoted
to float64 BEFORE computing goldens, so both sides solve identical inputs.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

import discretize
import numpy as np
import scipy.sparse as sp
import torch
from pymatsolver import SolverLU
from scipy.optimize import least_squares
from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "judge" / "golden"
SEED = 350101
spec = importlib.util.spec_from_file_location("criteria", ROOT / "tests/tolerances.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def rounded(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def geometry(small=False):
    if small:
        hx = rounded([20, 10, 10, 10, 10, 20])
        hz = rounded([20, 10, 10, 10, 10])
        origin = rounded([-40, -60])
        electrodes = rounded([[-15, 0], [-5, 0], [5, 0], [15, 0]])
        abmn = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2],
                         [1, 2, 0, 3], [1, 3, 0, 2], [2, 3, 0, 1]], np.int32)
    else:
        hx = rounded([80, 40, 20] + [10] * 16 + [20, 40, 80])
        hz = rounded([80, 40, 20] + [10] * 8)
        origin = rounded([-220, -220])
        electrodes = rounded([[x, 0] for x in range(-75, 76, 10)])
        rows = []
        # Both-sided dipole-dipole plus Wenner arrays at multiple separations.
        for a in range(15):
            for m in range(15):
                if 2 <= abs(m - a) <= 6:
                    rows.append((a, a + 1, m, m + 1))
        for spacing in range(1, 5):
            for a in range(16 - 3 * spacing):
                rows.append((a, a + 3 * spacing, a + spacing, a + 2 * spacing))
        abmn = np.array(sorted(set(rows)), np.int32)
    # Explicit trapezoidal inverse-cosine transform at y=0, including 1/pi.
    k = np.logspace(-5, np.log10(2), 25)
    dk = np.diff(k) / 2
    w = np.r_[dk, 0] + np.r_[0, dk]
    w[0] += k[0]  # constant extrapolation on [0,k[0]]
    return {"hx": hx, "hz": hz, "origin": origin,
            "quadrature": rounded(np.column_stack([k, w / np.pi])),
            "electrodes": electrodes, "abmn": abmn, "current": np.array(1.0)}


def simulator(g):
    mesh = discretize.TensorMesh([g["hx"], g["hz"]], origin=g["origin"])
    sources, order = [], []
    # SimPEG groups data by source; invert its permutation to preserve API order.
    for a, b in dict.fromkeys(map(tuple, g["abmn"][:, :2])):
        ix = np.flatnonzero(np.all(g["abmn"][:, :2] == [a, b], axis=1))
        m, n = g["abmn"][ix, 2:].T
        rx = dc.receivers.Dipole(g["electrodes"][m], g["electrodes"][n])
        sources.append(dc.sources.Dipole([rx], g["electrodes"][a],
                                        g["electrodes"][b], current=float(g["current"])))
        order.extend(ix)
    order = np.asarray(order)
    mapping = maps.ExpMap(nP=mesh.nC) * maps.LinearMap(-np.log(10) * sp.eye(mesh.nC))
    sim = dc.Simulation2DCellCentered(mesh, survey=dc.Survey(sources),
                                    sigmaMap=mapping, solver=SolverLU,
                                    bc_type="Robin", nky=25, do_trap=True)
    # Freeze transform nodes/weights; bypass SimPEG's mesh-dependent optimization.
    sim._quad_points = g["quadrature"][:, 0].copy()
    sim._quad_weights = g["quadrature"][:, 1].copy()
    return mesh, sim, order


def predict(sim, order, model):
    # SimPEG's model setter uses np.allclose to retain cached material matrices.
    # A 1e-5-dex FD step is "close" to a ~2-dex model, so force invalidation.
    # This is essential: otherwise the purported FD oracle is identically zero.
    sim.model = None
    f = sim.fields(model.ravel())
    predicted = np.empty(len(order))
    predicted[order] = sim.dpred(f=f)
    return predicted, f


def regularizer(shape):
    nz, nx = shape
    rows = [sp.eye(nz * nx, format="csr")]
    pairs = []
    for iz in range(nz):
        for ix in range(nx):
            i = iz * nx + ix
            if ix + 1 < nx:
                pairs.append((i, i + 1))
            if iz + 1 < nz:
                pairs.append((i, i + nx))
    r = np.repeat(np.arange(len(pairs)), 2)
    c = np.asarray(pairs).ravel()
    rows.append(sp.csr_matrix((np.tile([-1.0, 1.0], len(pairs)), (r, c)),
                              shape=(len(pairs), nz * nx)))
    return sp.vstack(rows, format="csr")


def objective(sim, order, model, observed, sigma, reference, beta):
    p, f = predict(sim, order, model)
    r = (p - observed) / sigma
    l = regularizer(model.shape)
    delta = (model - reference).ravel()
    value = r @ r / (2 * len(r)) + beta * np.linalg.norm(l @ delta)**2 / 2
    grad = sim.Jtvec(model.ravel(), ((r / sigma) / len(r))[order], f=f)
    grad += beta * (l.T @ (l @ delta))
    return value, grad.reshape(model.shape), p


def float32_reference(mesh, sim, order, g, observed, sigma, reference):
    """Replay the reference discrete objective with genuine float32 arithmetic.

    SimPEG/discretize provide fixed geometric, Robin and projection operators;
    these model-independent coefficients are rounded once to float32. Material
    mapping, face inner products, matrix assembly, dense LU solve, quadrature,
    residuals, regularization, reductions and autodiff all run in torch float32
    on CPU. No float64 objective or derivative enters this evaluator. It is a
    precision feasibility audit, not an implementation or a GPU pass claim.
    """
    def tensor(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32, device="cpu")

    div = tensor(sim.Div.toarray())
    # The isotropic face inner product is linear in cell resistivity. Obtain
    # its model-independent coefficient matrix from the reference library.
    face_map = mesh.get_face_inner_product_deriv(np.ones(mesh.nC))(np.ones(mesh.nF))
    face_map = tensor(face_map.toarray())
    gradients = []
    for k in g["quadrature"][:, 0]:
        sim.setBC(k)
        gradients.append(div.T - tensor(sim._MBC[k].toarray()))
    gradients = torch.stack(gradients)
    rhs = tensor(sim.getRHS(g["quadrature"][0, 0]))
    projections = []
    source_columns = []
    for i, source in enumerate(sim.survey.source_list):
        projection = source.receiver_list[0].getP(mesh, "CC").toarray()
        projections.append(projection)
        source_columns.extend([i] * len(projection))
    projection = tensor(np.concatenate(projections)[np.argsort(order)])
    source_columns = torch.tensor(np.asarray(source_columns)[np.argsort(order)])
    volumes = tensor(mesh.cell_volumes)
    k, w = tensor(g["quadrature"]).T
    observed, sigma, reference = map(tensor, (observed, sigma, reference))

    def evaluate(model, beta):
        m = tensor(model).requires_grad_(True)
        rho = torch.exp(m.flatten() * tensor(np.log(10)))
        resistance = face_map @ rho
        a = div @ (gradients / resistance[None, :, None])
        a = a + k[:, None, None]**2 * torch.diag(volumes / rho)
        u = torch.linalg.solve(a, rhs.expand(len(k), -1, -1))
        voltage_by_source = torch.sum(w[:, None, None] * u, dim=0)
        predicted = torch.sum(projection * voltage_by_source[:, source_columns].T, dim=1)
        delta = m - reference
        penalty = (torch.sum(delta**2) + torch.sum(torch.diff(delta, dim=0)**2)
                   + torch.sum(torch.diff(delta, dim=1)**2))
        value = torch.mean(((predicted - observed) / sigma)**2) / 2 + beta * penalty / 2
        gradient, = torch.autograd.grad(value, m)
        assert all(x.dtype == torch.float32 for x in
                   (rho, resistance, a, u, predicted, penalty, value, gradient))
        return tuple(x.detach().numpy() for x in (value, gradient, predicted))

    return evaluate


def recovery_metrics(model, truth, initial, active, anomaly, predicted, observed, sigma):
    contrast = truth[anomaly][0] - initial[anomaly][0]
    signed = np.sign(contrast) * (model - initial)
    support = active & (signed >= T.SUPPORT_CONTRAST_FRACTION * abs(contrast))
    assert np.any(support), "reference did not recover an anomaly"
    zz, xx = np.indices(model.shape)
    weights = np.where(support, signed, 0.0)
    recovered = np.array([(zz * weights).sum(), (xx * weights).sum()]) / weights.sum()
    true_center = np.array([zz[anomaly].mean(), xx[anomaly].mean()])
    return {"chi2": float(np.mean(((predicted - observed) / sigma)**2)),
            "location_cells": float(np.linalg.norm(recovered - true_center)),
            "anomaly_max_dex": float(np.max(abs(model[anomaly] - truth[anomaly]))),
            "background_rms_dex": float(np.sqrt(np.mean(
                (model[active & ~anomaly] - truth[active & ~anomaly])**2)))}


def check_recovery(metrics):
    assert metrics["chi2"] <= T.TARGET_CHI2, metrics
    assert metrics["location_cells"] <= T.LOCATION_CELLS, metrics
    assert metrics["anomaly_max_dex"] <= T.ANOMALY_LOG10_MAX, metrics
    assert metrics["background_rms_dex"] <= T.BACKGROUND_LOG10_RMS, metrics


def make_recovery(name, rng, center_x, anomaly_rho):
    g = geometry()
    mesh, sim, order = simulator(g)
    shape = (len(g["hz"]), len(g["hx"]))
    x, z = mesh.cell_centers.T.reshape(2, *shape)
    active = (abs(x) < 60) & (z > -40)
    anomaly = (abs(x - center_x) < 10) & (z > -30) & (z < -10)
    initial = rounded(np.full(shape, 2.0))
    truth = initial.copy()
    truth[anomaly] = rounded(np.log10(anomaly_rho))
    clean, _ = predict(sim, order, truth)
    sigma = rounded(0.005 * abs(clean) + 0.0001 * abs(clean).max())
    noise = rng.standard_normal(len(clean)) * sigma
    observed = rounded(clean + noise)
    beta, lower, upper, iterations = 0.02, 0.5, 3.5, 100
    l = regularizer(shape)
    free = np.flatnonzero(active.ravel())
    nd = len(clean)
    last = {}

    def evaluate(v):
        if "v" not in last or not np.array_equal(v, last["v"]):
            m = initial.copy()
            m.ravel()[free] = v
            p, f = predict(sim, order, m)
            last.update(v=v.copy(), m=m, p=p, f=f)
        return last

    def residual(v):
        a = evaluate(v)
        return np.r_[(a["p"] - observed) / sigma / np.sqrt(nd),
                     np.sqrt(beta) * (l @ (a["m"] - initial).ravel())]

    def jacobian(v):
        a = evaluate(v)
        j = sim.getJ(a["m"].ravel(), f=a["f"])[np.argsort(order)][:, free]
        return np.vstack([j / sigma[:, None] / np.sqrt(nd),
                          np.sqrt(beta) * l[:, free].toarray()])

    solution = least_squares(residual, initial.ravel()[free], jac=jacobian,
                             bounds=(lower, upper), max_nfev=iterations,
                             ftol=T.ORACLE_OPT_FTOL, xtol=T.ORACLE_OPT_XTOL,
                             gtol=T.ORACLE_OPT_GTOL)
    final = evaluate(solution.x)
    metrics = recovery_metrics(final["m"], truth, initial, active, anomaly,
                               final["p"], observed, sigma)
    print(name, json.dumps(metrics), "evaluations", solution.nfev, flush=True)
    check_recovery(metrics)
    start, _ = predict(sim, order, initial)
    a, b, m, n = (g["electrodes"][g["abmn"][:, j]] for j in range(4))
    geometric_factor = (1 / np.linalg.norm(a - m, axis=1) -
                        1 / np.linalg.norm(a - n, axis=1) -
                        1 / np.linalg.norm(b - m, axis=1) +
                        1 / np.linalg.norm(b - n, axis=1)) / (2 * np.pi)
    apparent = clean / geometric_factor
    homogeneous_apparent = start / geometric_factor
    # Physical sanity only: this coarse inversion mesh is not project 34's
    # mesh-convergence judge. Its homogeneous voltages should be of the right
    # sign, scale and units relative to the analytic 100 ohm m half-space.
    assert np.all(apparent > 0)
    assert np.all((homogeneous_apparent > T.SANITY_HALFSPACE_RHO_MIN) &
                  (homogeneous_apparent < T.SANITY_HALFSPACE_RHO_MAX))
    initial_chi2 = float(np.mean(((start - observed) / sigma)**2))
    assert initial_chi2 > T.MIN_INITIAL_CHI2_FACTOR * T.TARGET_CHI2
    assert np.max(abs(initial[anomaly] - truth[anomaly])) > T.ANOMALY_LOG10_MAX
    assert len(free) == 48 and anomaly.sum() == 4
    np.savez_compressed(OUT / f"{name}.npz", **g, initial=initial, truth=truth,
                        active=active.astype(np.int32), anomaly=anomaly.astype(np.int32),
                        clean=clean, observed=observed, standard_deviation=sigma,
                        initial_predicted=start, reference_model=final["m"],
                        reference_predicted=final["p"], beta=beta,
                        lower_bound=lower, upper_bound=upper, max_iterations=iterations)
    return {"metrics": metrics, "initial_chi2": initial_chi2, "n_data": nd,
            "shape": list(shape), "n_active": len(free), "n_anomaly": int(anomaly.sum()),
            "voltage_range": [float(clean.min()), float(clean.max())],
            "apparent_resistivity_range": [float(apparent.min()), float(apparent.max())],
            "homogeneous_apparent_range": [float(homogeneous_apparent.min()),
                                          float(homogeneous_apparent.max())],
            "rho_background": 100.0, "rho_anomaly": anomaly_rho,
            "reference_nfev": solution.nfev, "reference_status": int(solution.status),
            "reference_message": solution.message}


def make_gradient(rng):
    g = geometry(small=True)
    mesh, sim, order = simulator(g)
    shape = (len(g["hz"]), len(g["hx"]))
    reference = rounded(np.full(shape, 2.0))
    truth = reference.copy()
    truth[-2:, 2:4] = rounded(1.5)
    clean, _ = predict(sim, order, truth)
    sigma = rounded(0.02 * abs(clean) + 0.001 * abs(clean).max())
    observed = rounded(clean + rng.standard_normal(len(clean)) * sigma)
    model = rounded(reference + rng.normal(0, 0.12, shape))
    # Fixed dense and localized directions are saved, never sampled by tests.
    directions = rng.normal(size=(3, *shape))
    directions /= np.linalg.norm(directions.reshape(3, -1), axis=1)[:, None, None]
    localized = np.zeros((3, *shape))
    localized[0, -1, 1] = 1
    localized[1, -2, 2] = 1
    localized[2, 0, 0] = 1
    # A deep corner alone has a ~1 dex^-1 slope beside a ~1000 objective;
    # objective float32 rounding would swamp its centered difference. Mix in a
    # shallow cell, retaining sensitivity to both without relaxing FD tolerance.
    localized[2, -1, 1] = 0.5
    localized[2] /= np.linalg.norm(localized[2])
    directions = rounded(np.concatenate([directions, localized]))
    evaluate32 = float32_reference(mesh, sim, order, g, observed, sigma, reference)
    info = []
    for beta in (0.0, 20.0):
        value, grad, p = objective(sim, order, model, observed, sigma, reference, beta)
        value32, grad32, p32 = evaluate32(model, beta)
        # Audit the same parity requirements as the runtime tests before using
        # this evaluator to size the FD rounding margin.
        np.testing.assert_allclose(p32, p, rtol=T.PARITY_RTOL, atol=T.VOLTAGE_ATOL)
        np.testing.assert_allclose(value32, value, rtol=T.PARITY_RTOL, atol=T.OBJECTIVE_ATOL)
        np.testing.assert_allclose(grad32, grad, rtol=T.PARITY_RTOL, atol=T.GRADIENT_ATOL)
        fd, actual_directions, self_fd_errors32, saved_fd_errors32 = [], [], [], []
        objective_errors32, epsilon_budgets, truncation_errors = [], [], []
        for direction in directions:
            step = T.FD_STEP
            plus_model = rounded(model + step * direction)
            minus_model = rounded(model - step * direction)
            actual_direction = (plus_model - minus_model) / (2 * step)
            actual_directions.append(actual_direction)
            # The runtime contracts its gradient with this rounded-input chord,
            # not the nominal direction. Independently difference the float64
            # objective along that SAME chord at the smaller oracle step.
            step = T.ORACLE_FD_STEP
            plus = objective(sim, order, model + step * actual_direction, observed, sigma,
                             reference, beta)[0]
            minus = objective(sim, order, model - step * actual_direction, observed, sigma,
                              reference, beta)[0]
            fd.append((plus - minus) / (2 * step))
            step = T.FD_STEP
            plus32 = evaluate32(plus_model, beta)[0]
            minus32 = evaluate32(minus_model, beta)[0]
            # Match test_centered_finite_difference bit for bit: scalar
            # differences and gradient contractions are promoted by the test,
            # while all implementation/reference computations remain float32.
            adjoint32 = float(np.sum(grad32.astype(np.float64) * actual_direction))
            fd32 = (float(plus32) - float(minus32)) / (2 * step)
            self_fd_errors32.append(abs(fd32 - adjoint32) /
                                    max(abs(adjoint32), T.FD_SLOPE_FLOOR))
            saved_fd_errors32.append(abs(adjoint32 - fd[-1]) /
                                     max(abs(fd[-1]), T.FD_SLOPE_FLOOR))
            plus64 = objective(sim, order, plus_model, observed, sigma, reference, beta)[0]
            minus64 = objective(sim, order, minus_model, observed, sigma, reference, beta)[0]
            objective_errors32.extend([abs(float(plus32) - plus64),
                                       abs(float(minus32) - minus64)])
            slope64 = float(np.sum(grad * actual_direction))
            truncation_errors.append(abs((plus64 - minus64) / (2 * step) - slope64) /
                                     max(abs(slope64), T.FD_SLOPE_FLOOR))
            # Scale estimate, not a solver-error bound: one float32 epsilon
            # at each objective, divided by the centered difference signal.
            epsilon_budgets.append(np.finfo(np.float32).eps * max(abs(plus64), abs(minus64)) /
                                   (step * max(abs(slope64), T.FD_SLOPE_FLOOR)))
        fd = np.asarray(fd)
        slopes = np.einsum("kij,ij->k", actual_directions, grad)
        errors = abs(fd - slopes) / np.maximum(abs(slopes), T.FD_SLOPE_FLOOR)
        assert errors.max() <= T.ORACLE_FD_RTOL, errors
        # Replay the test's saved-slope assertion for EVERY direction, using
        # both the float64 golden gradient and a float32-rounded ideal output.
        saved_slope_errors = []
        for candidate in (grad, grad.astype(np.float32).astype(np.float64)):
            candidate_slopes = np.array([np.sum(candidate * v) for v in actual_directions])
            comparison = abs(candidate_slopes - fd) / np.maximum(abs(fd), T.FD_SLOPE_FLOOR)
            assert np.all(comparison <= T.FD_RTOL), comparison
            saved_slope_errors.append(comparison.tolist())
        print("gradient", beta, "actual-direction slopes", slopes.tolist(),
              "saved-slope errors (float64, float32)", saved_slope_errors,
              "full float32 self-FD errors", self_fd_errors32,
              "full float32 saved-slope errors", saved_fd_errors32, flush=True)
        assert max(self_fd_errors32) <= T.FD_RTOL, self_fd_errors32
        assert max(saved_fd_errors32) <= T.FD_RTOL, saved_fd_errors32
        suffix = "data" if beta == 0 else "regularized"
        np.savez_compressed(OUT / f"gradient_{suffix}.npz", **g,
                            log10_rho=model, observed=observed, standard_deviation=sigma,
                            reference=reference, beta=beta, objective=value,
                            gradient=grad, predicted=p, directions=directions,
                            fd_slopes=fd, row_selection=np.array(
                                [5, 1, 3, 0, 4, 2, 1], dtype=np.int32))
        info.append({"beta": beta, "objective": float(value),
                     "gradient_range": [float(grad.min()), float(grad.max())],
                     "oracle_fd_max_relative": float(errors.max()),
                     "saved_slope_float64_relative_by_direction": saved_slope_errors[0],
                     "saved_slope_float32_relative_by_direction": saved_slope_errors[1],
                     "fd_slopes_convention": "centered float64 differences at ORACLE_FD_STEP "
                         "along (float32(model+FD_STEP*v)-float32(model-FD_STEP*v))/(2*FD_STEP)",
                     "float32_audit": {
                         "arithmetic": "torch CPU float32 material mapping, assembly, LU solve, "
                                       "quadrature, objective and autodiff; fixed reference "
                                       "geometry/Robin/projection coefficients rounded to float32",
                         "step_dex": T.FD_STEP,
                         "objective": float(value32),
                         "objective_absolute_error": abs(float(value32) - value),
                         "perturbed_objective_max_absolute_error": max(objective_errors32),
                         "self_fd_relative_by_direction": self_fd_errors32,
                         "saved_slope_relative_by_direction": saved_fd_errors32,
                         "one_epsilon_scale_by_direction": epsilon_budgets,
                         "float64_runtime_step_truncation_by_direction": truncation_errors}})
    # Reciprocity and dipole signs exercise source/data permutations.
    base, _ = predict(sim, order, model)
    for label, transformed, factor in (
            ("reciprocal", g["abmn"][:, [2, 3, 0, 1]], 1),
            ("reverse_source", g["abmn"][:, [1, 0, 2, 3]], -1),
            ("reverse_receiver", g["abmn"][:, [0, 1, 3, 2]], -1)):
        changed = dict(g, abmn=transformed)
        _, s, o = simulator(changed)
        result, _ = predict(s, o, model)
        np.testing.assert_allclose(result, factor * base, rtol=T.ORACLE_LINEAR_RTOL,
                                   atol=T.VOLTAGE_ATOL)
        info.append({"sanity": label})
    return info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    recovery = {}
    for name, x, rho in [("conductor", -20, 25.0), ("resistor", 20, 400.0)]:
        recovery[name] = make_recovery(name, rng, x, rho)
    gradient = make_gradient(rng)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "simpeg", "discretize", "pymatsolver", "pytest", "mlx", "torch")}
    # Export each numerical value together with its authoritative comment.
    tolerance_text = (ROOT / "tests/tolerances.py").read_text()
    lines = tolerance_text.splitlines()
    criteria = {}
    for node in ast.parse(tolerance_text).body:
        if not isinstance(node, ast.Assign):
            continue
        key = node.targets[0].id
        comments = []
        i = node.lineno - 2
        while i >= 0 and lines[i].startswith("#"):
            comments.insert(0, lines[i][1:].strip())
            i -= 1
        assert comments, f"Missing reason for tolerance {key}"
        criteria[key] = {"value": getattr(T, key), "reason": " ".join(comments)}
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.glob("*.npz"))}
    inventory = {}
    for name in files:
        with np.load(OUT / name, allow_pickle=False) as data:
            inventory[name] = {key: {"shape": list(data[key].shape),
                                     "dtype": str(data[key].dtype)} for key in data.files}
    meta = {"project": "35-dc-inversion", "tag": "M", "paths": ["mlx"], "seed": SEED,
            "python": platform.python_version(), "versions": versions,
            "oracle": "SimPEG Simulation2DCellCentered + SciPy bounded least_squares",
            "precision": "float64, implementation inputs rounded through float32 first",
            "conventions": "See tests/api.py for mesh, Robin BCs, Fourier normalization, "
                           "ABMN order, log10 model and objective definitions.",
            "reference_optimizer": "bounded trust-region least_squares; fixed beta; "
                                   "analytic SimPEG Jacobian; max_nfev=100",
            "reference_cache": "sim.model=None before each solve forces material-cache "
                               "invalidation; np.allclose otherwise hides small FD steps",
            "tolerances": criteria,
            "tolerance_reasons_source": "tests/tolerances.py (comments are normative)",
            "noise": "independent seeded Gaussian: sigma=.005*abs(clean)+.0001*max(abs(clean)); "
                     "observations rounded to float32; sigma fixed after generation",
            "recovery": recovery, "gradient": gradient, "sha256": files, "inputs": inventory}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    total = sum(p.stat().st_size for p in OUT.iterdir())
    assert total < 20_000_000
    print(f"Wrote {len(files)} npz goldens, {total} bytes; versions={versions}")


if __name__ == "__main__":
    main()
