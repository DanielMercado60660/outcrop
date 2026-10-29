"""Deterministic CPU float64 Deepwave elastic oracle; no implementation imports.

Run with the repository's pinned Python from the project directory. Every model, source,
observation, perturbation and analytic target consumed by pytest is saved here.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import runpy
import sys
import time
import warnings
from pathlib import Path

import deepwave
import numpy as np
import torch
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "judge" / "golden"
TOL = {k: v for k, v in runpy.run_path(str(ROOT / "tests/tolerances.py")).items()
       if k.isupper() and isinstance(v, (int, float))}
SEED = 150127
DTYPE = torch.float64
FORWARD_KEYS = ("vp", "vs", "rho", "source_y", "source_x", "source_locations",
                "receiver_locations", "spacing", "dt", "pml_width", "pml_freq",
                "max_vel")
REPORT = {}


def ricker(t, frequency, peak):
    q = (np.pi * frequency * (t - peak)) ** 2
    return (1 - 2 * q) * np.exp(-q)


def inputs(ny, nx, nt, h=10.0, dt=0.0004, shots=1, sources=1):
    return {"vp": np.full((ny, nx), 3000.), "vs": np.full((ny, nx), 1500.),
            "rho": np.full((ny, nx), 2000.),
            "source_y": np.zeros((shots, sources, nt)),
            "source_x": np.zeros((shots, sources, nt)),
            "source_locations": np.zeros((shots, sources, 2), np.int64),
            "receiver_locations": np.zeros((shots, 1, 2), np.int64),
            "spacing": np.array([h, h]), "dt": np.array(dt),
            "pml_width": np.array([12, 12, 12, 12], np.int64),
            "pml_freq": np.array(25.), "max_vel": np.array(5000.)}


def tensor(a, grad=False, dtype=DTYPE):
    return torch.tensor(a, dtype=dtype, requires_grad=grad)


def simulate(case, gradients=False, dtype=DTYPE):
    """Exact API mapping to the installed reference, with all defaults pinned."""
    differentiable = ("vp", "vs", "rho", "source_y", "source_x")
    a = {k: tensor(case[k], gradients, dtype) for k in differentiable}
    solid = a["rho"] > 0
    lamb = a["rho"] * (a["vp"] ** 2 - 2 * a["vs"] ** 2)
    mu = a["rho"] * a["vs"] ** 2
    safe_rho = torch.where(solid, a["rho"], torch.ones_like(a["rho"]))
    buoyancy = torch.where(solid, 1 / safe_rho, torch.zeros_like(safe_rho))
    # Bare top zero-width face is safe here because two vacuum rows separate
    # it from solid; explicit boundary="free" is rejected by this build.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pml_width.*")
        out = deepwave.elastic(
            lamb, mu, buoyancy, case["spacing"].tolist(), float(case["dt"]),
            source_amplitudes_y=a["source_y"], source_amplitudes_x=a["source_x"],
            source_locations_y=tensor(case["source_locations"], dtype=torch.int64),
            source_locations_x=tensor(case["source_locations"], dtype=torch.int64),
            receiver_locations_y=tensor(case["receiver_locations"], dtype=torch.int64),
            receiver_locations_x=tensor(case["receiver_locations"], dtype=torch.int64),
            receiver_locations_p=tensor(case["receiver_locations"], dtype=torch.int64),
            accuracy=2, pml_width=case["pml_width"].tolist(),
            pml_freq=float(case["pml_freq"]), max_vel=float(case["max_vel"]),
            mpml_fraction=0., surface_method="ivm", survey_pad=None,
            model_gradient_sampling_interval=1, freq_taper_frac=0.,
            time_pad_frac=0., time_taper=False, location_interpolation="none",
            backend="compiled", storage="device")
    assert out.info.step_ratio == 1, f"Unexpected substepping: {out.info}"
    traces = {"vy": out.receiver_amplitudes_y,
              "vx": out.receiver_amplitudes_x, "p": out.receiver_amplitudes_p}
    result = {k: v.detach().numpy().copy() for k, v in traces.items()}
    if gradients:
        n = traces["vy"].numel()
        scale = float(case["data_scale"])
        objective = sum(((traces[k] - tensor(case["observed_" + k[-1]],
                                             dtype=dtype)) ** 2).sum()
                        for k in ("vy", "vx")) / (2 * n * scale**2)
        objective.backward()
        result["objective"] = objective.detach().numpy().copy()
        for k in differentiable:
            result["grad_" + k] = a[k].grad.detach().numpy().copy()
    return result


def objective(case):
    out = simulate(case)
    return sum(np.sum((out["v" + c] - case["observed_" + c])**2)
               for c in ("y", "x")) / (2 * out["vy"].size * float(case["data_scale"])**2)


def save(name, case, out=None, **extra):
    if out is None:
        out = simulate(case)
    arrays = {**case, **{"expected_" + k: v for k, v in out.items()}, **extra}
    assert all(np.isfinite(v).all() for v in arrays.values()
               if np.asarray(v).dtype.kind in "fiu"), name
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    REPORT[name] = {"shape": list(case["vp"].shape),
                    "nt": case["source_y"].shape[-1],
                    "peaks": {k: float(np.max(np.abs(v))) for k, v in out.items()}}
    print(name, REPORT[name], flush=True)
    return out


def parity_cases(rng):
    for name, surface in (("heterogeneous", False), ("free_surface", True)):
        c = inputs(28, 34, 480, shots=2, sources=2)
        yy, xx = np.indices(c["vp"].shape)
        c["vp"] += 250 * (yy >= 14) + 130 * np.exp(-((xx-19)**2+(yy-13)**2)/30)
        c["vs"] += 180 * (yy >= 14) + 50 * np.sin(xx / 7)
        c["rho"] += 150 * (yy >= 14) + 30 * rng.uniform(-1, 1, yy.shape)
        c["spacing"] = np.array([9., 11.])
        c["pml_width"] = np.array([8, 10, 12, 9], np.int64)
        c["source_locations"] = np.array([[[5, 9], [9, 20]],
                                           [[6, 23], [12, 10]]], np.int64)
        c["receiver_locations"] = np.array([[[3, 5], [8, 16], [18, 25]],
                                             [[4, 18], [15, 12], [22, 27]]], np.int64)
        if surface:
            for k in ("vp", "vs", "rho"):
                c[k][:2] = 0
            c["pml_width"][0] = 0
            c["receiver_locations"][:, 0, 0] = 1
            c["source_locations"][0, 0, 0] = 1
        t = np.arange(480) * float(c["dt"])
        for shot in range(2):
            for src in range(2):
                c["source_y"][shot, src] = 1.e7 * (1-.3*shot) * ricker(t, 25, .045+.008*src)
                c["source_x"][shot, src] = -6.e6 * (1-.2*src) * ricker(t, 30, .052+.005*shot)
        out = save(name, c)
        assert all(np.max(np.abs(out[k])) > 0 for k in ("vy", "vx", "p"))
        single = {**c, **{k: c[k][:1].copy() for k in
                          ("source_y", "source_x", "source_locations", "receiver_locations")}}
        assert all(np.array_equal(simulate(single)[k], out[k][:1]) for k in out)
    z = inputs(9, 11, 16)
    z["source_locations"][:] = [4, 5]
    z["receiver_locations"][:] = [5, 6]
    save("zero", z)


def plane_cases():
    # Infinite uniform force sheet reduced to 1-D d'Alembert solution:
    # v(y,t) = F(t-|y-ys|/c)/(2*rho*c). The grid force density is F(t)/dy.
    # The sheet spans a domain wide enough that its ends cannot affect the
    # central receivers during the recorded window (even at the max FD speed).
    for mode, component, speed, freq, peak, receiver_y in (
            ("P", "y", 3000., 18., .07, 40),
            ("S", "x", 1500., 12., .115, 28)):
        errors, parity_l2_budgets = [], []
        for level in range(3):
            factor = 2**level
            h, dt = 7.5/factor, .0003/factor
            nt = 800 * factor
            c = inputs(81*factor+1, 221*factor+1, nt, h=h, dt=dt,
                       sources=221*factor-1)
            nsrc = c["source_y"].shape[1]
            c["source_locations"][0, :, 0] = 20*factor
            c["source_locations"][0, :, 1] = np.arange(1, nsrc+1)
            c["receiver_locations"] = np.array([[[receiver_y*factor, 110*factor]]], np.int64)
            times = np.arange(nt) * dt
            force = 1.e8 * ricker(times, freq, peak)
            c["source_" + component][0, :, :] = force[None, :] / h
            distance = (receiver_y-20)*7.5
            analytic = 1.e8 * ricker((np.arange(nt)-.5)*dt - distance/speed,
                                     freq, peak) / (2*2000*speed)
            out = simulate(c)
            e = np.linalg.norm(out["v"+component][0, 0] - analytic) / np.linalg.norm(analytic)
            errors.append(float(e))
            # Any path satisfying the per-sample max-norm parity criterion
            # has L2 perturbation <= sqrt(nt)*PARITY*reference_peak. Apply
            # the triangle inequality against the continuum analytic trace.
            parity_l2_budgets.append(float(TOL["PARITY"] * np.max(np.abs(out["v"+component]))
                                          * np.sqrt(nt) / np.linalg.norm(analytic)))
            save(f"plane_{mode}_{level}", c, out, analytic=analytic,
                 analytic_component=np.array(component), refinement=np.array(level))
        orders = np.log2(np.array(errors[:-1])/errors[1:])
        fine_bound = errors[-1] + parity_l2_budgets[-1]
        REPORT[f"convergence_{mode}"] = {
            "errors": errors, "orders": orders.tolist(),
            "parity_l2_budgets": parity_l2_budgets,
            "fine_error_with_full_parity_budget": fine_bound,
            "fine_threshold_headroom": TOL["PLANE_FINE_ERROR"] - fine_bound}
        print("convergence", mode, errors, orders, flush=True)
        assert np.all((orders >= TOL["ORDER_MIN"]) & (orders <= TOL["ORDER_MAX"]))
        assert fine_bound <= TOL["PLANE_FINE_ERROR"], REPORT[f"convergence_{mode}"]
        print("finest L2 bound including full parity budget", mode, fine_bound, flush=True)


def rayleigh_case():
    vp, vs = 3000., 1500.
    beta = (vs/vp)**2
    # q=(c_R/vs)^2; nonzero physical Rayleigh root in (0,1).
    polynomial = lambda q: q**3 - 8*q**2 + (24-16*beta)*q - 16*(1-beta)
    q = brentq(polynomial, .01, .999999, xtol=TOL["RAYLEIGH_ROOT_XTOL"])
    cr = vs * np.sqrt(q)
    assert abs(polynomial(q)) <= TOL["RAYLEIGH_ROOT_RESIDUAL"]
    assert 0 < cr < vs < vp
    h, dt, nt = 2.5, .0002, 4125
    c = inputs(103, 481, nt, h=h, dt=dt)
    for k in ("vp", "vs", "rho"):
        c[k][:2] = 0
    c["pml_width"] = np.array([0, 24, 24, 24], np.int64)
    c["pml_freq"] = np.array(18.)
    c["source_locations"][:] = [1, 40]
    offsets = np.array([400., 600., 800.])
    c["receiver_locations"] = np.array([[[1, 200], [1, 280], [1, 360]]], np.int64)
    t0 = .09
    c["source_y"][0, 0] = 1.e8 * ricker(np.arange(nt)*dt, 18., t0)
    out = simulate(c)
    arrival = t0 + offsets/cr
    # Lock windows before seeing the waveform: +/- half a Rayleigh period
    # about the analytic prediction; cubic-free three-point peak interpolation.
    windows = np.stack((arrival-0.5/18, arrival+0.5/18), axis=-1)
    measured = []
    for trace, window in zip(out["vy"][0], windows):
        indices = np.flatnonzero(((np.arange(nt)-.5)*dt >= window[0]) &
                                 ((np.arange(nt)-.5)*dt <= window[1]))
        idx = int(indices[np.argmax(np.abs(trace[indices]))])
        a, b, d = np.abs(trace[idx-1:idx+2])
        shift = .5*(a-d)/(a-2*b+d)
        measured.append((idx+shift-.5)*dt)
    err = np.abs(np.array(measured)-arrival)/arrival
    analytic_lags = np.diff(offsets)/cr
    lag_errors = np.abs(np.diff(measured)-analytic_lags)/analytic_lags
    save("lamb", c, out, analytic_arrival=arrival, arrival_windows=windows,
         offsets=offsets, rayleigh_speed=np.array(cr), source_peak=np.array(t0))
    REPORT["rayleigh"] = {"speed": float(cr), "analytic": arrival.tolist(),
                          "measured": measured, "relative_error": err.tolist(),
                          "lag_relative_error": lag_errors.tolist()}
    print("Rayleigh", REPORT["rayleigh"], flush=True)
    assert np.max(err) <= TOL["RAYLEIGH_TIME"]
    assert np.max(lag_errors) <= TOL["RAYLEIGH_TIME"]


def pml_cases():
    c = inputs(31, 37, 1200, shots=2)
    c["source_locations"][:] = [15, 18]
    c["receiver_locations"] = np.array([[[15, 18], [12, 22]],
                                         [[15, 18], [12, 22]]], np.int64)
    t = np.arange(1200)*float(c["dt"])
    c["source_y"][0, 0] = 1.e7*ricker(t, 30., .05)
    c["source_x"][1, 0] = 1.e7*ricker(t, 30., .05)
    c["pml_freq"] = np.array(30.)
    c["pml_width"][:] = 20
    # Domain edge >= 850 m from nearest receiver: reflected body waves cannot
    # return in 0.48 s (vp=3000 m/s). Same sheet/point source normalization.
    big = inputs(191, 197, 1200, shots=2)
    for k in ("source_y", "source_x", "pml_freq"):
        big[k] = c[k].copy()
    big["source_locations"] = c["source_locations"] + np.array([80, 80])
    big["receiver_locations"] = c["receiver_locations"] + np.array([80, 80])
    ref = simulate(big)
    incident = np.concatenate([ref[k].reshape(2, -1) for k in ("vy", "vx")], axis=1)
    norm = np.linalg.norm(incident, axis=1)
    start = 250  # 0.1 seconds, after the direct pulse, includes all echoes.
    residuals = []
    for name, width in (("pml", 20), ("reflective", 0)):
        c["pml_width"][:] = width
        out = simulate(c)
        residual = np.concatenate([(out[k]-ref[k])[..., start:].reshape(2, -1)
                                   for k in ("vy", "vx")], axis=1)
        e = np.linalg.norm(residual, axis=1) / norm
        residuals.append(e)
        save(name, c, out, open_vy=ref["vy"], open_vx=ref["vx"],
             late_start=np.array(start), incident_norm=np.array(norm))
    REPORT["absorption"] = {"pml": residuals[0].tolist(), "reflective": residuals[1].tolist(),
                            "improvement": (residuals[1]/residuals[0]).tolist()}
    print("absorption", REPORT["absorption"], flush=True)
    assert np.all(residuals[0] <= TOL["PML_REFLECTION"])
    assert np.all(residuals[1] >= TOL["REFLECTIVE_MIN"])
    assert np.all(residuals[1]/residuals[0] >= TOL["PML_IMPROVEMENT"])


def gradient_case(rng, surface=False):
    c = inputs(12, 14, 180, h=10., dt=.0005)
    yy, xx = np.indices(c["vp"].shape)
    c["vp"] += 80*np.sin(xx/3) + 100*(yy>6)
    c["vs"] += 40*np.cos(yy/3)
    c["rho"] += 50*np.cos((yy+xx)/4)
    c["pml_width"] = np.array([5, 6, 4, 7], np.int64)
    c["source_locations"][:] = [4, 5]
    c["receiver_locations"] = np.array([[[4, 7], [7, 6], [8, 9]]], np.int64)
    name = "gradient_surface" if surface else "gradient"
    if surface:
        for k in ("vp", "vs", "rho"):
            c[k][:2] = 0
        c["pml_width"][0] = 0
        c["receiver_locations"][0, 0, 0] = 1
    t = np.arange(180)*float(c["dt"])
    c["source_y"][0, 0] = 1.e7*ricker(t, 40., .026)
    c["source_x"][0, 0] = -7.e6*ricker(t, 35., .03)
    target = {**c, "vs": c["vs"]*1.03, "vp": c["vp"]*.98,
              "source_y": c["source_y"]*.9, "source_x": c["source_x"]*.9}
    observed = simulate(target)
    c["observed_y"], c["observed_x"] = observed["vy"], observed["vx"]
    c["data_scale"] = np.array(max(np.max(np.abs(observed[k])) for k in ("vy", "vx")))
    out = simulate(c, gradients=True)
    directions, fd_report = {}, {}
    for k in ("vp", "vs", "rho", "source_y", "source_x"):
        # Fixed smooth/global direction plus a fixed seeded local direction,
        # both with physical units; edge nodes are included to check padding.
        scale = np.max(np.abs(c[k]))
        ds = np.stack((c[k], scale*rng.normal(size=c[k].shape)))
        if surface and k in ("vp", "vs", "rho"):
            ds[:, :2] = 0  # Vacuum is a fixed region, not a variable material.
            assert np.all(out["grad_"+k][:2] == 0)
        for j, direction in enumerate(ds):
            step = TOL["ORACLE_FD_STEP"]
            plus, minus = {**c, k: c[k]+step*direction}, {**c, k: c[k]-step*direction}
            fd = (objective(plus)-objective(minus))/(2*step)
            adjoint = np.sum(out["grad_"+k]*direction)
            error = abs(fd-adjoint)/max(abs(fd), abs(adjoint))
            fd_report[f"{k}_{j}"] = {"fd": float(fd), "adjoint": float(adjoint), "error": float(error)}
            assert error <= TOL["ORACLE_GRAD_FD"], fd_report
        directions["direction_"+k] = ds
    save(name, c, out, **directions)
    REPORT[name+"_fd"] = fd_report
    print("oracle FD", fd_report, flush=True)


def float32_audit():
    """Check attainability on reference float32; never used as the oracle."""
    audit = {}
    fields = ("vp", "vs", "rho", "source_y", "source_x")
    for file in sorted(OUT.glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            g = {k: z[k] for k in z.files}
        name = file.stem
        c = {k: g[k] for k in FORWARD_KEYS}
        is_gradient = name.startswith("gradient")
        if is_gradient:
            c.update({k: g[k] for k in ("observed_y", "observed_x", "data_scale")})
        for k in fields:
            c[k] = c[k].astype(np.float32)
        out = simulate(c, gradients=is_gradient, dtype=torch.float32)
        errors = {}
        for k, value in out.items():
            reference = g["expected_"+k]
            scale = float(np.max(np.abs(reference)))
            absent = (name.startswith("plane_P") and k == "vx") or (
                name.startswith("plane_S") and k in ("vy", "p"))
            if absent:
                velocity = max(np.max(np.abs(g["expected_vy"])), np.max(np.abs(g["expected_vx"])))
                scale = velocity if k != "p" else velocity*np.max(g["rho"]*g["vp"])
            difference = float(np.max(np.abs(value.astype(np.float64)-reference)))
            errors[k] = difference/scale if scale > 0 else difference
            assert errors[k] <= (TOL["PARITY"] if scale > 0 else TOL["ZERO"]), (name, k, errors)
        if is_gradient:
            fd_errors = {}
            for k in fields:
                for j, d in enumerate(g["direction_"+k]):
                    step = TOL["FD_STEP"]
                    plus = {**c, k: (c[k].astype(np.float64)+step*d).astype(np.float32)}
                    minus = {**c, k: (c[k].astype(np.float64)-step*d).astype(np.float32)}
                    def loss(case, observations=c):
                        v = simulate(case, dtype=torch.float32)
                        return sum(np.sum((v["v"+axis].astype(np.float64)-
                                           observations["observed_"+axis].astype(np.float32).astype(np.float64))**2)
                                   for axis in ("y", "x"))/(2*v["vy"].size*float(observations["data_scale"])**2)
                    fd = (loss(plus)-loss(minus))/(2*step)
                    effective_d = (plus[k].astype(np.float64)-minus[k].astype(np.float64))/(2*step)
                    adjoint = np.sum(out["grad_"+k].astype(np.float64)*effective_d)
                    error = abs(fd-adjoint)/max(abs(fd), abs(adjoint))
                    fd_errors[f"{k}_{j}"] = float(error)
                    assert error <= TOL["GRAD_FD"], (name, k, j, fd, adjoint, error)
            errors["centered_fd"] = fd_errors
        audit[name] = errors
        print("float32 audit", name, errors, flush=True)
    REPORT["float32_reference_audit"] = audit


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    versions = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "deepwave", "pytest")}
    lock = json.loads((ROOT/"judge/reference_lock.json").read_text())
    dw = Path(deepwave.__file__).parent
    hashes = {name: hashlib.sha256((dw/name).read_bytes()).hexdigest()
              for name in lock["source_sha256"]}
    if versions != lock["versions"] or hashes != lock["source_sha256"]:
        raise RuntimeError("Reference environment differs from judge/reference_lock.json; "
                           "refusing to overwrite goldens. Restore the pinned environment.")
    print("versions", versions, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    parity_cases(rng)
    plane_cases()
    rayleigh_case()
    pml_cases()
    gradient_case(rng)
    gradient_case(rng, surface=True)
    float32_audit()
    sizes = {p.name: p.stat().st_size for p in OUT.glob("*.npz")}
    assert sum(sizes.values()) < TOL["GOLDEN_BYTES_LIMIT"]
    meta = {"project": "15-elastic-psv", "seed": SEED, "versions": versions,
                "python": sys.version, "reference": "Deepwave elastic, CPU C, float64",
                "source_sha256": hashes, "golden_bytes": sizes,
                "golden_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(OUT.glob("*.npz"))},
                "scheme": "2nd-order staggered space / leapfrog time, IVM free surface, CFS-C-PML kappa=1",
                "scheme_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                  for name in ("tests/SCHEME.md", "tests/api.py")},
                "inputs": {name: {k: (v.tolist() if v.ndim == 0 or v.size <= 16 else
                                   {"shape": list(v.shape), "dtype": str(v.dtype),
                                    "min": float(v.min()), "max": float(v.max())})
                               for k, v in np.load(OUT/f"{name}.npz").items()
                               if k in FORWARD_KEYS}
                        for name in REPORT if (OUT/f"{name}.npz").exists()},
                "tolerances": TOL,
                "tolerance_reasons": (ROOT/"tests/tolerances.py").read_text(),
                "checks": REPORT,
                "analytic": ["1-D d'Alembert P/S force-sheet Green function",
                          "Lamb half-space vertical surface load: Rayleigh secular polynomial q^3-8q^2+(24-16 beta)q-16(1-beta)=0, q=(cR/vs)^2, beta=(vs/vp)^2"],
                "papers": ["Komatitsch and Martin (2007), An unsplit convolutional perfectly matched layer improved at grazing incidence for the seismic wave equation, Geophysics 72(5), SM155-SM167, doi:10.1190/1.2757586",
                        "Lamb (1904), On the propagation of tremors over the surface of an elastic solid, Phil. Trans. R. Soc. A 203, 1-42, doi:10.1098/rsta.1904.0013",
                        "Zeng et al. (2012), Improved vacuum formulation for 2D finite-difference modeling of Rayleigh waves including surface topography and internal discontinuities, Geophysics 77(1), T1-T9, doi:10.1190/geo2011-0067.1"],
                "elapsed_seconds": time.monotonic()-start}
    (OUT/"meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    print("PASS oracle sanity", sum(sizes.values()), "bytes", meta["elapsed_seconds"], "seconds", flush=True)


if __name__ == "__main__":
    main()
