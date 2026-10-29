"""Finite rectangular Okada surface dislocations + independent SVD inverse.

The judge-only float64 kernel is asserted against all 81 finite-source values
in Okada (1985), Table 2, before any synthetic goldens are generated. Offline
published checks are supplied in data/okada/test_values.json, independently of
project 44.

SciPy's float64 augmented least squares is independent of implementation
solvers. All stored problem arrays are float32-representable values promoted
to float64 before reference calculations, eliminating input-rounding ambiguity.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import runpy
from decimal import Decimal
from pathlib import Path

import numpy as np
from okada_surface import displacement
from scipy.linalg import lstsq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "judge" / "golden"
CHECKLIST = ROOT.parents[1] / "data" / "okada" / "test_values.json"
TOL = runpy.run_path(str(ROOT / "tests" / "tolerances.py"))
SEED = 450101
CASES = ("joint_noiseless", "joint_noisy", "joint_alternate")


def validate_okada():
    """Check 27 displacements + 54 horizontal derivatives, including +/-90 dip."""
    published = json.loads(CHECKLIST.read_text())["okada1985_table2"]
    checks = {}
    for case in published["cases"]:
        inp = case["inputs"]
        if inp["source"] != "finite_rectangle":
            continue  # A surface finite-rectangle oracle has no point-source API.
        slip = [inp[k] for k in ["U1", "U2", "U3"]]

        def evaluate(x, y, inp=inp, slip=slip):
            return displacement(x, y, inp["d"], inp["dip_deg"], inp["L"], inp["W"], slip)

        x, y = inp["x"], inp["y"]
        h = TOL["OKADA_CHECK_FD_STEP"]
        u = evaluate(x, y)
        dx = (evaluate(x+h, y) - evaluate(x-h, y)) / (2*h)
        dy = (evaluate(x, y+h) - evaluate(x, y-h)) / (2*h)
        got = dict(zip(["ux", "uy", "uz", "duxdx", "duxdy", "duydx", "duydy", "duzdx", "duzdy"],
                       [*u, dx[0], dy[0], dx[1], dy[1], dx[2], dy[2]], strict=True))
        checks[case["name"]] = {}
        for key, printed in case["outputs"].items():
            want = float(printed)
            # Decimal preserves the last printed decimal place, including zeros.
            last_unit = 10.0 ** Decimal(printed).as_tuple().exponent if want else 0.0
            limit = TOL["OKADA_PRINTED_HALF_UNIT"] * last_unit + TOL["OKADA_CHECK_ATOL"]
            error = abs(float(got[key]) - want)
            assert error <= limit, (case["name"], key, got[key], want, error, limit)
            checks[case["name"]][key] = {"computed": float(got[key]), "printed": printed,
                                         "abs_error": error, "allowance": limit}
    assert len(checks) == 9
    return {"citation": published["citation"], "count": 81, "checks": checks,
            "source_sha256": hashlib.sha256(CHECKLIST.read_bytes()).hexdigest(),
            "scope": "All finite-source cases 2-4; no point-source or internal-depth API. Negative vertical dip uses geometric equivalence."}


def f32exact(value):
    return np.asarray(value, dtype=np.float32).astype(np.float64)


def objective(g, d, sigma, lmat, damping, slip):
    residual = (g @ slip - d) / sigma
    penalty = lmat @ slip
    value = 0.5 * (residual @ residual + damping**2 * (penalty @ penalty))
    gradient = g.T @ (residual / sigma) + damping**2 * lmat.T @ penalty
    return np.asarray(value), gradient


def build_case(index):
    rng = np.random.default_rng(SEED + index)
    # Six finite patches on a 12 km long, 5 km wide buried oblique-slip fault.
    # Ordering is shallow/deep dip band, then three along-strike patches.
    strike, dip, rake = ([35., 55., 65.] if index < 2 else [72., 42., 110.])
    theta, delta, rho = np.deg2rad([strike, dip, rake])
    # Columns are local along-strike, horizontal up-dip and up axes in ENU.
    rotation = np.array([[np.sin(theta), -np.cos(theta), 0],
                         [np.cos(theta), np.sin(theta), 0], [0, 0, 1]])
    length, width = 4000., 2500.
    patch_x = np.tile([-6000., -2000., 2000.], 2)
    band = np.repeat([0, 1], 3)
    patch_y = 2000. - (band+1) * width * np.cos(delta)
    bottom = 1000. + (band+1) * width * np.sin(delta)
    top = bottom - width * np.sin(delta)
    shift = np.array([900., -600., 0.]) if index == 2 else np.zeros(3)
    origins = np.column_stack([patch_x, patch_y, np.zeros(6)]) @ rotation.T + shift
    xx, yy = np.meshgrid(np.linspace(-14000., 14000., 7)+53.,
                         np.linspace(-11000., 11000., 7)+117.)
    insar_xyz = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    xx, yy = np.meshgrid(np.linspace(-13000., 13000., 5), np.linspace(-10000., 10000., 4))
    gps_xyz = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    gps_xyz[:, :2] += rng.uniform(-450., 450., (len(gps_xyz), 2))
    # Ascending/descending and a third heading/incidence: three independent
    # positive-up, toward-satellite LOS unit vectors at the same surface pixels.
    looks = np.array([[0.55, -0.15, np.sqrt(1-0.55**2-0.15**2)],
                      [-0.60, -0.12, np.sqrt(1-0.60**2-0.12**2)],
                      [0.12, 0.62, np.sqrt(1-0.12**2-0.62**2)]])
    obs_xyz = np.concatenate([np.tile(insar_xyz, (3, 1)), np.repeat(gps_xyz, 3, axis=0)])
    projections = np.concatenate([np.repeat(looks, len(insar_xyz), axis=0),
                                  np.tile(np.eye(3), (len(gps_xyz), 1))])
    enu_kernel = np.empty((len(obs_xyz), 6, 3), dtype=np.float64)
    for j in range(6):
        local = (obs_xyz - origins[j]) @ rotation
        local_u = displacement(local[:, 0], local[:, 1], bottom[j], dip,
                               length, width, [np.cos(rho), np.sin(rho), 0.])
        enu_kernel[:, j, :] = local_u @ rotation.T
    green = f32exact(np.einsum("npc,nc->np", enu_kernel, projections))
    # Reject the old degeneracy explicitly before writing any fixture.
    assert not np.any(np.all(green == 0, axis=1))
    assert np.linalg.matrix_rank(looks) == 3
    los_rank = np.linalg.matrix_rank(green[:3*len(insar_xyz)].reshape(3, -1))
    assert los_rank == 3
    true_slip = f32exact(0.25 + 1.6 * rng.uniform(size=6))
    sigma = f32exact(np.r_[np.full(3*len(insar_xyz), 0.002),
                          np.tile([0.001, 0.001, 0.003], len(gps_xyz))])
    clean = green @ true_slip
    noise = rng.normal(size=len(clean)) * sigma if index else np.zeros_like(clean)
    data = f32exact(clean + noise)
    # Horizontal/vertical nearest-neighbour differences on the supplied 2x3
    # patch grid; identity rows make the pure penalty objective full rank.
    differences = np.zeros((7, 6))
    for row, (a, b) in enumerate([(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]):
        differences[row, [a, b]] = [-1, 1]
    regularizer = f32exact(np.vstack([differences, 0.1 * np.eye(6)]))
    damping = float([0.0, 0.8, 1.5][index])
    areas = f32exact(np.full(6, length * width))
    modulus = 30e9
    augmented = np.vstack([green / sigma[:, None], damping * regularizer])
    rhs = np.r_[data / sigma, np.zeros(len(regularizer))]
    recovered, _, rank, singular = lstsq(augmented, rhs, lapack_driver="gelsd")
    assert rank == len(true_slip)
    predicted = green @ recovered
    moment_true = modulus * (areas @ abs(true_slip))
    moment = modulus * (areas @ abs(recovered))
    # A substantial penalty tests regularization inside the inverse itself,
    # independently of the pure-penalty objective/gradient fixtures below.
    strong_damping = 40.0
    strong_augmented = np.vstack([green / sigma[:, None], strong_damping * regularizer])
    strong_slip, _, strong_rank, _ = lstsq(strong_augmented, rhs, lapack_driver="gelsd")
    assert strong_rank == len(true_slip)
    unregularized, _, _, _ = lstsq(green / sigma[:, None], data / sigma, lapack_driver="gelsd")
    omitted_penalty_error = float(np.max(abs(unregularized-strong_slip)) / np.max(abs(strong_slip)))
    assert omitted_penalty_error > TOL["PARITY_RTOL"]
    recovered32, _, _, _ = lstsq(augmented.astype(np.float32), rhs.astype(np.float32), lapack_driver="gelsd")
    solve32_error = float(np.max(abs(recovered32-recovered)) / np.max(abs(recovered)))
    assert solve32_error <= TOL["PARITY_RTOL"]
    trial = f32exact(rng.uniform(-0.4, 1.3, 6))
    obj, grad = objective(green, data, sigma, regularizer, damping, trial)
    optimum, optimum_grad = objective(green, data, sigma, regularizer, damping, recovered)
    h = TOL["FD_STEP_M"]
    fd = np.empty_like(trial)
    for j in range(len(trial)):
        offset = np.zeros_like(trial)
        offset[j] = h
        plus = objective(green, data, sigma, regularizer, damping, trial + offset)[0]
        minus = objective(green, data, sigma, regularizer, damping, trial - offset)[0]
        fd[j] = (plus - minus) / (2 * h)
    fd_error = float(np.max(abs(fd - grad)) / np.max(abs(grad)))
    assert fd_error <= TOL["GRADIENT_RTOL"]
    corr = float(np.corrcoef(recovered, true_slip)[0, 1])
    moment_error = float(abs(moment / moment_true - 1))
    assert corr >= TOL["MIN_SLIP_CORRELATION"]
    assert moment_error <= TOL["MAX_MOMENT_REL_ERROR"]
    # Explicit zero-data and negative-data references prevent fixed-answer solvers.
    arrays = {"green": green, "data": data, "sigma": sigma, "regularizer": regularizer,
              "damping": np.asarray(damping), "areas": areas,
              "shear_modulus": np.asarray(modulus), "true_slip": true_slip,
              "clean_data": clean, "noise": data-clean, "slip": recovered,
              "predicted": predicted, "moment": np.asarray(moment),
              "true_moment": np.asarray(moment_true), "trial": trial,
              "objective": obj, "gradient": grad, "fd_gradient": fd,
              "optimum_objective": optimum, "optimum_gradient": optimum_grad,
              "zero_data": np.zeros_like(data), "zero_slip": np.zeros_like(trial),
              "zero_predicted": np.zeros_like(data), "zero_moment": np.asarray(0.0),
              "negative_data": -data, "negative_slip": -recovered,
              "negative_predicted": -predicted, "negative_moment": np.asarray(moment),
              "strong_damping": np.asarray(strong_damping), "strong_slip": strong_slip,
              "strong_predicted": green @ strong_slip,
              "strong_moment": np.asarray(modulus * (areas @ abs(strong_slip))),
              "patch_origin_enu": origins, "top_depth": top, "bottom_depth": bottom,
              "patch_length": np.asarray(length), "patch_width": np.asarray(width),
              "strike_dip_rake_deg": np.array([strike, dip, rake]),
              "poisson_ratio": np.asarray(0.25), "rotation_local_to_enu": rotation,
              "observation_xyz_enu": obs_xyz, "insar_xyz_enu": insar_xyz,
              "gps_xyz_enu": gps_xyz, "los_unit_vectors_enu": looks,
              "unit_slip_enu": enu_kernel, "projections_enu": projections}
    # A pure penalty fixture isolates damping, its square, and L^T L from the
    # much larger data term; a solver cannot pass by omitting regularization.
    penalty_green = np.zeros_like(green)
    penalty_data = np.zeros_like(data)
    penalty_obj, penalty_grad = objective(penalty_green, penalty_data, sigma,
                                         regularizer, damping, trial)
    arrays.update(penalty_green=penalty_green, penalty_data=penalty_data,
                  penalty_objective=penalty_obj, penalty_gradient=penalty_grad)
    # Verify that the prescribed FD step survives float32 objective rounding.
    # This is an oracle diagnostic, not an implementation or claimed GPU pass.
    fd32_errors = {}
    for label, gg, dd, expected_grad in [
        ("data_and_penalty", green, data, grad),
        ("penalty", penalty_green, penalty_data, penalty_grad),
    ]:
        trial32 = trial.astype(np.float32)
        fd32 = np.empty_like(trial)
        for j in range(len(trial)):
            plus, minus = trial32.copy(), trial32.copy()
            plus[j] += h
            minus[j] -= h
            values = [objective(gg.astype(np.float32), dd.astype(np.float32),
                                sigma.astype(np.float32), regularizer.astype(np.float32),
                                np.float32(damping), s)[0] for s in [plus, minus]]
            fd32[j] = (float(values[0])-float(values[1])) / (float(plus[j])-float(minus[j]))
        scale = float(np.max(abs(expected_grad)))
        error = float(np.max(abs(fd32-expected_grad)))
        assert error <= TOL["GRADIENT_RTOL"] * scale
        fd32_errors[label] = error / scale if scale else error
    # Full supplied permutations, generated here rather than in the tests.
    row_order = rng.permutation(len(data))
    col_order = rng.permutation(len(true_slip))
    arrays.update(permuted_green=green[row_order][:, col_order],
                  permuted_data=data[row_order], permuted_sigma=sigma[row_order],
                  permuted_regularizer=regularizer[:, col_order],
                  permuted_areas=areas[col_order], permuted_slip=recovered[col_order],
                  permuted_predicted=predicted[row_order], permuted_moment=np.asarray(moment))
    np.savez_compressed(OUT / f"{CASES[index]}.npz", **arrays)
    return {"seed": SEED+index, "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "damping": damping, "rank": int(rank), "condition": float(singular[0]/singular[-1]),
            "correlation": corr, "moment_rel_error": moment_error,
            "true_moment_Nm": float(moment_true), "recovered_moment_Nm": float(moment),
            "slip_range_m": [float(recovered.min()), float(recovered.max())],
            "displacement_range_m": [float(clean.min()), float(clean.max())],
            "zero_green_rows": int(np.sum(np.all(green == 0, axis=1))),
            "los_block_rank": int(los_rank),
            "gps_component_max_m": np.max(abs(clean[3*len(insar_xyz):].reshape(-1, 3)), axis=0).tolist(),
            "objective": float(optimum), "fd_relative_error": fd_error,
            "float32_lstsq_slip_relative_error": solve32_error,
            "strong_damping": strong_damping, "omitted_penalty_slip_relative_error": omitted_penalty_error,
            "float32_fd_diagnostic": fd32_errors}


def main():
    validation = validate_okada()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {name: build_case(i) for i, name in enumerate(CASES)}
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pytest"]}
    meta = {"status": "READY FOR ROUND-2 SCIENCE REVIEW; finite-fault oracle validated",
            "reference": "Okada (1985) rectangular surface solution + scipy.linalg.lstsq (gelsd), float64",
            "okada_validation": validation,
            "dependency": "Offline Table 2 values (data/okada/test_values.json); no project-44 code or goldens used",
            "python": platform.python_version(), "versions": versions, "cases": cases,
            "conventions": "ENU; strike clockwise from north; local y up-dip; bottom-edge depth; rake from +U1 toward +U2; toward-satellite LOS; look-major pixels then station-major GPS E,N,U; dip-band-major patch order",
            "inputs": "Each npz contains the inputs. Implementation array inputs G,d,sigma,L,areas,trial are float32-rounded then promoted to float64 for the oracle; scalar inputs, diagnostic geometry and reference outputs are float64.",
            "tolerances": {k: {"value": TOL[k], "reason": v} for k, v in TOL["REASONS"].items()},
            "judge_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [Path(__file__), ROOT / "judge" / "okada_surface.py",
                                              ROOT / "tests" / "tolerances.py"]},
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.npz"))}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"okada_checks_passed": validation["count"], "versions": versions, "cases": {
        name: {k: v for k, v in summary.items() if k != "shapes"}
        for name, summary in cases.items()}}, indent=2))


if __name__ == "__main__":
    main()
