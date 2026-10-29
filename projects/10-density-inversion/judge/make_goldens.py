"""Deterministic float64 SimPEG oracle. Run from the project directory."""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import runpy
from pathlib import Path

import numpy as np
from discretize import TensorMesh
from packaging.requirements import Requirement
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from simpeg import data, data_misfit, maps, regularization, utils
from simpeg.potential_fields import gravity

OUT = Path(__file__).resolve().parent / "golden"
TOLERANCES = runpy.run_path(str(OUT.parent.parent / "tests" / "tolerances.py"))["TOLERANCES"]


def tol(name):
    return TOLERANCES[name]["value"]


def explicit_precision(shape, weights, alpha):
    """Independent edge-sum check on SimPEG's volume-normalized Hessian."""
    n = int(np.prod(shape))
    q = np.diag(weights**2)
    indices = np.arange(n).reshape(shape, order="F")
    for axis, scale in enumerate(alpha):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        for i, j in zip(indices[tuple(lo)].ravel(), indices[tuple(hi)].ravel()):
            v = scale * (weights[i] ** 2 + weights[j] ** 2) / 2
            q[i, i] += v
            q[j, j] += v
            q[i, j] -= v
            q[j, i] -= v
    return q


def build(name, seed, shape, h, offset, alpha, variant=0):
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    mesh = TensorMesh([np.full(k, h) for k in shape],
                      origin=[-nx*h/2, -ny*h/2, -nz*h])
    xx, yy = np.meshgrid(np.linspace(-nx*h/2, nx*h/2, 12),
                         np.linspace(-ny*h/2, ny*h/2, 12), indexing="ij")
    receivers = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 10.0)]
    survey = gravity.survey.Survey(gravity.sources.SourceField(
        receiver_list=[gravity.receivers.Point(receivers, components=["gz"])]))
    simulation = gravity.simulation.Simulation3DIntegral(
        mesh, survey=survey, rhoMap=maps.IdentityMap(nP=mesh.nC),
        store_sensitivities="ram", sensitivity_dtype=np.float64, n_processes=1)
    g = np.asarray(simulation.G, dtype=np.float64)
    assert g.dtype == np.float64 and np.all(g < 0) and np.all(np.isfinite(g))
    truth = np.zeros(shape)
    if nx >= 8:
        if variant == 0:
            truth[1:4, 1:4, -2:] = 0.5
            truth[5:7, 4:7, -2:] = -0.4
        else:
            truth[4:7, 1:4, -2:] = 0.45
            truth[1:3, 4:7, -2:] = -0.35
    else:
        truth[1:3, 1:3, -1:] = 0.5
    truth = truth.ravel(order="F")
    clean = simulation.dpred(truth)
    std = np.full(clean.size, 0.015 * np.max(np.abs(clean)))
    if name == "small":
        std *= np.linspace(0.7, 1.4, clean.size)
    observed = clean + std * rng.standard_normal(clean.size)
    weights = utils.depth_weighting(mesh, 0.0, exponent=2.0, threshold=offset)
    reg = regularization.WeightedLeastSquares(
        mesh, reference_model=np.zeros(mesh.nC), reference_model_in_smooth=True,
        alpha_s=1.0, alpha_x=h*h*alpha[0], alpha_y=h*h*alpha[1],
        alpha_z=h*h*alpha[2], alpha_xx=0.0, alpha_yy=0.0, alpha_zz=0.0,
        weights={"depth": weights**2})
    q = reg.deriv2(np.zeros(mesh.nC)).toarray() / (2*h**3)
    np.testing.assert_allclose(q, explicit_precision(shape, weights, alpha),
                               rtol=tol("oracle_regularization"), atol=tol("oracle_regularization"))
    dm = data_misfit.L2DataMisfit(data=data.Data(survey, dobs=observed,
                                              standard_deviation=std),
                                simulation=simulation)
    a = g / std[:, None]
    b = observed / std
    reference = np.zeros(mesh.nC)
    normal = a.T @ a
    rhs = a.T @ b

    def solve(log_beta):
        beta = np.exp(log_beta)
        model = cho_solve(cho_factor(normal + beta*q), rhs)
        return model, np.linalg.norm(a@model-b)**2

    log_beta = brentq(lambda x: solve(x)[1] - clean.size, -20, 25, xtol=tol("beta_root_xtol"))
    recovered, chi = solve(log_beta)
    np.testing.assert_allclose(chi, float(clean.size), rtol=tol("oracle_check"), atol=0)
    beta = np.exp(log_beta)
    corr = np.corrcoef(recovered, truth)[0, 1]
    error = np.linalg.norm(recovered-truth)
    if nx >= 8:
        assert corr >= tol("correlation"), (name, corr)
    # Nonzero reference and distinct beta probes exercise every objective term.
    probe = rng.normal(0, 0.2, mesh.nC)
    probe_reference = rng.normal(0, 0.05, mesh.nC)
    direction = rng.standard_normal(mesh.nC)
    direction /= np.linalg.norm(direction)
    dual = rng.standard_normal(clean.size)
    reg.reference_model = probe_reference
    probe_betas = np.array([0.0, beta, 7*beta])
    phi_d = dm(probe)
    phi_m = reg(probe) / h**3
    gradient_d = dm.deriv(probe)
    gradient_m = reg.deriv(probe) / h**3
    gradients = np.array([gradient_d + v*gradient_m for v in probe_betas])
    objectives = phi_d + probe_betas*phi_m
    # Float64 centered differences independently validate SimPEG derivatives.
    eps = tol("oracle_fd_step")
    fd = (dm(probe+eps*direction) + beta*reg(probe+eps*direction)/h**3
          - dm(probe-eps*direction) - beta*reg(probe-eps*direction)/h**3)/(2*eps)
    np.testing.assert_allclose(fd, gradients[1]@direction, rtol=tol("oracle_gradient"), atol=0)
    np.testing.assert_allclose(dm(recovered), chi, rtol=tol("oracle_check"), atol=0)
    # Check that the locked float32 FD step is numerically resolvable, including
    # the isolated regularization term. This is a CPU audit, not a path pass.
    fd_errors = []
    p32, r32, v32 = [x.astype(np.float32) for x in (probe, probe_reference, direction)]
    g32, d32, s32, q32 = [x.astype(np.float32) for x in (g, observed, std, q)]
    for bv, reg_only in [(v, False) for v in probe_betas] + [(1.0, True)]:
        def objective32(m, bv=bv, reg_only=reg_only):
            delta = m-r32
            pd = np.float32(0) if reg_only else np.sum(((g32@m-d32)/s32)**2)
            return pd + np.float32(bv)*(delta@q32@delta)
        step = tol("fd_step")
        f32 = (float(objective32(p32+step*v32))-float(objective32(p32-step*v32)))/(2*step)
        grad = bv*gradient_m if reg_only else gradient_d+bv*gradient_m
        exact = grad@direction
        fd_errors.append(float(abs(f32-exact)/max(abs(f32), abs(exact))))
    assert max(fd_errors) <= tol("gradient"), fd_errors
    payload = {"sensitivity": g, "receivers": receivers, "cell_centers": mesh.cell_centers,
               "shape": np.array(shape), "cell_size": np.array(h), "depth_offset": np.array(offset),
               "alpha_xyz": np.array(alpha), "true_model": truth, "clean": clean, "data": observed,
               "uncertainty": std, "reference": reference, "weights": weights, "precision": q,
               "beta": np.array(beta), "target": np.array(clean.size), "model": recovered,
               "predicted": g@recovered, "correlation": np.array(corr), "error": np.array(error),
               "probe": probe, "probe_reference": probe_reference, "direction": direction, "dual": dual,
               "probe_betas": probe_betas, "objectives": objectives, "gradients": gradients,
               "reg_gradient": gradient_m, "zero_sensitivity": np.zeros_like(g), "zero_data": np.zeros_like(observed),
               "phi_d": np.array(phi_d), "phi_m": np.array(phi_m), "forward": g@probe, "adjoint": g.T@dual}
    np.savez_compressed(OUT / f"{name}.npz", **payload)
    stats = {"seed": seed, "shape": list(shape), "cell_size": h, "depth_offset": offset,
             "alpha_xyz": list(alpha), "beta": beta, "n_data": clean.size,
             "chi_squared": chi, "correlation": corr, "model_error": error,
             "gz_range_mgal": [float(clean.min()), float(clean.max())],
             "noise_std_range_mgal": [float(std.min()), float(std.max())],
             "density_range_g_cc": [float(truth.min()), float(truth.max())],
             "recovered_range_g_cc": [float(recovered.min()), float(recovered.max())],
             "cpu_float32_fd_relative_errors": fd_errors,
             "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for key, value in payload.items()}}
    print(name, json.dumps({key: value for key, value in stats.items() if key != "arrays"}))
    return stats


def main():
    requirements = OUT.parent / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text().splitlines():
            if line and not line.startswith("#"):
                package, pinned = line.split("==")
                if md.version(package) != pinned:
                    raise RuntimeError(f"Reference environment changed: {package} must be {pinned}")
    OUT.mkdir(exist_ok=True)
    cases = {"small": build("small", 10010, (3, 4, 2), 40.0, 13.0, (0.3, 0.7, 1.2)),
             "blocks_a": build("blocks_a", 10011, (8, 8, 4), 50.0, 25.0, (0.4, 0.4, 0.4)),
             "blocks_b": build("blocks_b", 10012, (8, 8, 4), 50.0, 40.0, (0.3, 0.6, 0.9), 1)}
    pending = ["numpy", "scipy", "simpeg", "discretize", "geoana", "libdlf", "pymatsolver", "pytest", "ruff", "numba", "mlx"]
    versions = {}
    while pending:
        p = pending.pop()
        if p in versions:
            continue
        versions[p] = md.version(p)
        for spec in md.requires(p) or []:
            req = Requirement(spec)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                pending.append(req.name)
    versions = dict(sorted(versions.items()))
    meta = {"python": platform.python_version(), "versions": versions, "cases": cases,
            "tolerances": TOLERANCES,
            "reference": "SimPEG Simulation3DIntegral float64 geoana; L2DataMisfit; WeightedLeastSquares",
            "solver": "Exact SPD quadratic minimizer using scipy Cholesky of SimPEG Hessian; beta chosen by discrepancy principle chi^2=N; unconstrained, no positivity bounds.",
            "conventions": "ENU; z up; mesh top z=0; x fastest, z slowest; gz positive upward; density g/cm^3; gravity mGal; full squared norms, no 1/2.",
            "sources": ["Li & Oldenburg (1998), 3-D inversion of gravity data, Geophysics 63, 109-119, doi:10.1190/1.1444302",
                        "Installed SimPEG regularization/base.py and utils/model_utils.py (depth_weighting exponent=2)."],
            "input_policy": "All data, probes, directions, noise saved here; implementation receives float32 casts of these arrays."}
    meta["input_recipes"] = {
        "mesh": "Uniform cubes, origin=(-nx*h/2,-ny*h/2,-nz*h), top z=0; all cells active.",
        "survey": "12x12 endpoint-inclusive regular surface grid over mesh xy extent; z=10m; upward gz.",
        "noise": "default_rng(case.seed); independent standard normals times 0.015*max(abs(clean)); small has a 0.7..1.4 receiver-order uncertainty multiplier.",
        "probes": "After noise draw: normal(0,0.2) model, normal(0,0.05) reference, unit-normalized standard-normal direction, standard-normal data dual.",
        "truth": "Signed, piecewise-constant rectangular blocks, specified by source array slices and saved in each archive. Auxiliary small case is for derivative and stationarity tests only.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (OUT.parent / "requirements.txt").write_text("\n".join(f"{p}=={v}" for p, v in versions.items()) + "\n")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    assert size < 20_000_000
    print(f"Golden bytes: {size}")


if __name__ == "__main__":
    main()
