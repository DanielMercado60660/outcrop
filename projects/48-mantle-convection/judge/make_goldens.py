"""Float64 FD oracle and analytic/published checks for Blankenbach case 1a.

No network or implementation imports. SciPy's orthonormal DST-I inverts the
Dirichlet five-point Laplacian exactly up to double precision; two inversions
solve the stress-free biharmonic problem. The thermal scheme is defined in
tests/api.py, independently implemented here using NumPy float64 arithmetic.

Reference: Blankenbach et al. (1989), Geophys. J. Int. 98, 23-38,
doi:10.1111/j.1365-246X.1989.tb04425.x, constant-viscosity case 1a.
The published scalars are external targets, never used to tune the simulation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import runpy
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.fft import dstn, idstn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "judge" / "golden"
SEED = 481989
PUBLISHED_NU = 4.884409
PUBLISHED_VRMS = 42.864947
SPECS: dict[str, dict] = {}
EIGENVALUES: dict[tuple, np.ndarray] = {}


def mesh(n):
    a = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    return np.meshgrid(a, a)  # x, z, with arrays indexed [z,x]


def poisson(rhs):
    """Solve L_h v=rhs, v=0 at all walls, preserving the input precision."""
    n = rhs.shape[0] - 1
    key = (n, rhs.dtype.str)
    if key not in EIGENVALUES:
        lam = -4.0 * n**2 * np.sin(np.pi * np.arange(1, n) / (2 * n)) ** 2
        EIGENVALUES[key] = (lam[:, None] + lam[None, :]).astype(rhs.dtype)
    v = np.zeros_like(rhs)
    v[1:-1, 1:-1] = idstn(
        dstn(rhs[1:-1, 1:-1], type=1, norm="ortho") / EIGENVALUES[key],
        type=1, norm="ortho",
    )
    return v


def stokes(temperature, rayleigh, dtype=np.float64):
    t = np.asarray(temperature, dtype=dtype)
    n = t.shape[0] - 1
    rhs = np.zeros_like(t)
    rhs[1:-1, 1:-1] = rayleigh * (n / 2) * (t[1:-1, 2:] - t[1:-1, :-2])
    psi = poisson(poisson(rhs)) if rayleigh else np.zeros_like(t)
    u, w = np.zeros_like(t), np.zeros_like(t)
    u[1:-1] = (n / 2) * (psi[2:] - psi[:-2])
    u[0], u[-1] = n * psi[1], -n * psi[-2]
    w[:, 1:-1] = -(n / 2) * (psi[:, 2:] - psi[:, :-2])
    w[:, 0], w[:, -1] = -n * psi[:, 1], n * psi[:, -2]
    return {"psi": psi, "u": u, "w": w}


def thermal_terms(t, u, w):
    """Return RHS and diffusion, including reflected insulating-wall ghosts."""
    n = t.shape[0] - 1
    # Reflection creates the correct second derivative at the sidewall itself.
    padded = np.pad(t, ((0, 0), (1, 1)), mode="reflect")
    tx = (n / 2) * (padded[:, 2:] - padded[:, :-2])
    lap = n**2 * (padded[1:-1, 2:] + padded[1:-1, :-2]
                   + t[2:] + t[:-2] - 4 * t[1:-1])
    tz = (n / 2) * (t[2:] - t[:-2])
    rhs = np.zeros_like(t)
    rhs[1:-1] = lap - u[1:-1] * tx[1:-1] - w[1:-1] * tz
    return rhs, lap


def diagnostics(t, u, w):
    n = t.shape[0] - 1
    weights = np.ones(n + 1, dtype=np.float64) / n
    weights[[0, -1]] *= 0.5
    top = (n / 2) * (4 * t[-2] - t[-3] - 3 * t[-1])
    bottom = (n / 2) * (3 * t[0] - 4 * t[1] + t[2])
    return {
        "nu_top": np.array(weights @ top),
        "nu_bottom": np.array(weights @ bottom),
        "vrms": np.array(np.sqrt(weights @ (u*u + w*w) @ weights)),
    }


def convection(temperature, rayleigh, dt, steps, dtype=np.float64):
    t = np.array(temperature, dtype=dtype, copy=True)
    for _ in range(steps):
        v = stokes(t, rayleigh, dtype=dtype)
        rhs, _ = thermal_terms(t, v["u"], v["w"])
        midpoint = t + (dt / 2) * rhs
        v_mid = stokes(midpoint, rayleigh, dtype=dtype)
        rhs_mid, _ = thermal_terms(midpoint, v_mid["u"], v_mid["w"])
        t += dt * rhs_mid
    v = stokes(t, rayleigh, dtype=dtype)
    return {"temperature": t, "u": v["u"], "w": v["w"],
            **diagnostics(t, v["u"], v["w"])}


def as_input(t):
    """Quantize before solving: the oracle sees exactly the implementation input."""
    result = np.array(t, dtype=np.float32, order="C", copy=True)
    result[0], result[-1] = 1.0, 0.0  # Remove transcendental sin(pi) wall noise.
    return result


def save(name, info, **arrays):
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    SPECS[name] = {**info, "arrays": {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)}
        for k, v in arrays.items()
    }}


def save_run(name, temperature, ra, dt, steps, **extra):
    inp = as_input(temperature)
    out = convection(inp, ra, dt, steps)
    save(name, {"rayleigh": ra, "dt": dt, "steps": steps,
                "final_time": dt * steps}, input_temperature=inp,
         rayleigh=np.array(ra), dt=np.array(dt), steps=np.array(steps),
         **out, **extra)
    return out


def normalized_error(a, b):
    return float(np.max(np.abs(a-b)) / np.max(np.abs(b)))


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    tol = runpy.run_path(str(ROOT / "tests" / "tolerances.py"))
    rng = np.random.default_rng(SEED)
    sanity = {}

    # Multi-mode, asymmetric buoyancy tests on both even and odd interval counts.
    for n in (12, 17, 32):
        x, z = mesh(n)
        t = 1-z
        modes = []
        for k, ell in ((1, 1), (2, 1), (1, 3), (3, 2)):
            a = float(rng.uniform(-0.06, 0.06))
            t += a*np.cos(k*np.pi*x)*np.sin(ell*np.pi*z)
            modes.append([k, ell, a])
        inp = as_input(t)
        ra = 3750.0
        save(f"stokes_n{n}", {"rayleigh": ra, "modes": modes},
             input_temperature=inp, rayleigh=np.array(ra), **stokes(inp, ra))

    # Manufactured continuous Stokes solution: unequal wave numbers avoid
    # accidental superconvergence of some velocity components for k=ell.
    stokes_errors = []
    for n in (16, 32, 64):
        x, z = mesh(n)
        k, ell, a, ra = 1, 2, 0.1, 1.0e4
        t = 1-z + a*np.cos(k*np.pi*x)*np.sin(ell*np.pi*z)
        c = -ra*a*k*np.pi / (np.pi**4*(k*k+ell*ell)**2)
        exact = {
            "psi": c*np.sin(k*np.pi*x)*np.sin(ell*np.pi*z),
            "u": c*ell*np.pi*np.sin(k*np.pi*x)*np.cos(ell*np.pi*z),
            "w": -c*k*np.pi*np.cos(k*np.pi*x)*np.sin(ell*np.pi*z),
        }
        inp = as_input(t)
        out = stokes(inp, ra)
        errors = {key: normalized_error(out[key], exact[key]) for key in exact}
        stokes_errors.append(errors)
        save(f"stokes_refine_n{n}", {"rayleigh": ra, "mode": [k, ell, a],
                                    "analytic_errors": errors},
             input_temperature=inp, rayleigh=np.array(ra), **out,
             **{f"exact_{key}": v for key, v in exact.items()})
    sanity["stokes_errors"] = stokes_errors
    for key in ("psi", "u", "w"):
        orders = [np.log2(stokes_errors[i][key]/stokes_errors[i+1][key])
                  for i in range(2)]
        assert min(orders) >= tol["MIN_SPATIAL_ORDER"], (key, orders)
        assert stokes_errors[-1][key] <= tol["FINE_STOKES_ERROR"]

    # Pure diffusion is an exact coupled solution when Ra=0. Compare to the
    # continuum exponential, with a common small dt so spatial error dominates.
    diffusion_errors = []
    for n in (8, 16, 32):
        x, z = mesh(n)
        base = 1-z
        perturbation = 0.1*np.cos(np.pi*x)*np.sin(np.pi*z)
        exact = base + perturbation*np.exp(-2*np.pi**2*0.02)
        out = save_run(f"diffusion_n{n}", base+perturbation, 0.0, 0.00005, 400,
                       exact_temperature=exact, base=base)
        diffusion_errors.append(normalized_error(out["temperature"]-base,
                                                 exact-base))
    sanity["diffusion_errors"] = diffusion_errors
    assert min(np.log2(np.array(diffusion_errors[:-1]) /
                       diffusion_errors[1:])) >= tol["MIN_SPATIAL_ORDER"]
    assert diffusion_errors[-1] <= tol["FINE_DIFFUSION_ERROR"]

    # Temporal refinement vs exact SEMIDISCRETE heat solution, not the continuum
    # solution (which would retain a fixed spatial-error floor). All dt are stable
    # for every eigenmode on N=8, including roundoff-excited high frequencies.
    n = 8
    x, z = mesh(n)
    base = 1-z
    perturbation = 0.2*np.cos(2*np.pi*x)*np.sin(2*np.pi*z)
    eigenvalue = -8*n*n*np.sin(np.pi/n)**2
    exact = base+perturbation*np.exp(eigenvalue*0.032)
    temporal_errors = []
    for steps in (10, 20, 40):
        out = save_run(f"time_s{steps}", base+perturbation, 0.0, 0.032/steps,
                       steps, exact_temperature=exact, base=base)
        temporal_errors.append(normalized_error(out["temperature"]-base,
                                                exact-base))
    sanity["temporal_errors"] = temporal_errors
    assert min(np.log2(np.array(temporal_errors[:-1]) /
                       temporal_errors[1:])) >= tol["MIN_TEMPORAL_ORDER"]

    # An exact conductive invariant, including steps=0 diagnostic semantics.
    _, z = mesh(16)
    save_run("conduction_zero", 1-z, 1.0e4, 0.0001, 0)
    save_run("conduction_evolved", 1-z, 1.0e4, 0.0001, 20)

    # Off-equilibrium coupled parity catches advection, velocity sign, midpoint
    # Stokes recomputation, sidewall ghosts, and final-state diagnostics.
    for n, steps in ((12, 1), (24, 75)):
        x, z = mesh(n)
        t = 1-z + 0.08*np.cos(np.pi*x)*np.sin(np.pi*z)
        t += 0.025*np.cos(2*np.pi*x)*np.sin(3*np.pi*z)
        save_run(f"coupled_n{n}", t, 1.0e4, 0.0001, steps)

    # Independent local convergence of the nonlinear advection term as well.
    # For T0=1-z+A cos(pi*x)sin(pi*z), F(T0) and F'(T0)F(T0) are analytic.
    # Compare (T(dt)-T0)/dt to F0+dt*F'F0/2. This is a short-time Taylor target,
    # not a claimed exact finite-time convection solution. The O(dt^2) rate
    # remainder is much smaller than spatial errors on these three grids.
    coupled_errors = []
    for n in (8, 16, 32):
        x, z = mesh(n)
        a, ra, small_dt = 0.1, 1.0e4, 0.00005
        mode = np.cos(np.pi*x)*np.sin(np.pi*z)
        t = 1-z+a*mode
        c = -ra*a/(4*np.pi**3)
        b = -2*np.pi**2*a-c*np.pi
        d = c*a*np.pi**2/2
        f0 = b*mode+d*np.sin(2*np.pi*z)
        jff0 = (b*b/a*mode-4*np.pi**2*d*np.sin(2*np.pi*z)
                 + c*b*np.pi**2*np.sin(2*np.pi*z)
                 + 2*c*np.pi**2*d*mode*np.cos(2*np.pi*z))
        rate = f0+small_dt*jff0/2
        rate[[0, -1]] = 0.0
        out_local = save_run(f"coupled_refine_n{n}", t, ra, small_dt, 1,
                             analytic_rate=rate)
        actual_rate = (out_local["temperature"]-as_input(t))/small_dt
        coupled_errors.append(normalized_error(actual_rate, rate))
    sanity["coupled_local_errors"] = coupled_errors
    assert min(np.log2(np.array(coupled_errors[:-1]) /
                       coupled_errors[1:])) >= tol["MIN_SPATIAL_ORDER"]

    # Full cold start, not a saved steady-state warm start. No target is supplied
    # to the numerical solve. Compare its computed fields and diagnostics later.
    n, dt, steps = 128, 0.00001, 30000
    x, z = mesh(n)
    initial = 1-z + 0.01*np.cos(np.pi*x)*np.sin(np.pi*z)
    out = save_run("case1a", initial, 1.0e4, dt, steps,
                   published_nu=np.array(PUBLISHED_NU),
                   published_vrms=np.array(PUBLISHED_VRMS))
    residual, lap = thermal_terms(out["temperature"], out["u"], out["w"])
    residual_norm = float(np.sqrt(np.mean(residual[1:-1]**2)) /
                          np.sqrt(np.mean(lap**2)))
    balance = float(abs(out["nu_top"]-out["nu_bottom"])/PUBLISHED_NU)
    sanity["case1a"] = {
        k: float(out[k]) for k in ("nu_top", "nu_bottom", "vrms")
    }
    sanity["case1a"].update(
        temperature_range=[float(out["temperature"].min()),
                           float(out["temperature"].max())],
        residual_relative_rms=residual_norm, heat_balance_relative=balance,
    )
    print("Case 1a sanity: " + json.dumps(sanity["case1a"]), flush=True)
    for key, target in (("nu_top", PUBLISHED_NU), ("nu_bottom", PUBLISHED_NU),
                        ("vrms", PUBLISHED_VRMS)):
        assert abs(float(out[key])/target - 1) <= tol["BENCHMARK_RTOL"], (key, out[key])
    assert residual_norm <= tol["STEADY_RESIDUAL_RTOL"]
    assert balance <= tol["HEAT_BALANCE_RTOL"]
    assert out["temperature"].min() >= -tol["TEMPERATURE_ATOL"]
    assert out["temperature"].max() <= 1+tol["TEMPERATURE_ATOL"]

    # An independent PRECISION audit, not a second implementation or a golden:
    # replay all operations (including SciPy FFTs) in float32 to check that the
    # locked 1e-4 parity ceiling remains attainable after 30,000 time steps.
    # Only the float64 trajectory above is saved as the oracle output.
    float32 = convection(as_input(initial), 1.0e4, dt, steps, dtype=np.float32)
    precision_errors = {key: normalized_error(float32[key], out[key])
                        for key in out}
    assert max(precision_errors.values()) <= tol["TRAJECTORY_RTOL"], precision_errors
    r32, lap32 = thermal_terms(float32["temperature"].astype(np.float64),
                              float32["u"].astype(np.float64),
                              float32["w"].astype(np.float64))
    residual32 = float(np.sqrt(np.mean(r32[1:-1]**2))/np.sqrt(np.mean(lap32**2)))
    assert residual32 <= tol["STEADY_RESIDUAL_RTOL"], residual32
    sanity["float32_audit"] = {"relative_errors": precision_errors,
                               "steady_residual_relative_rms": residual32}

    # Audit the smaller float32 cases and all four refinement groups too. These
    # checks do not substitute for implementation tests; they establish that the
    # selected tolerances are attainable using single-precision arithmetic.
    audit_outputs = {}
    max_small_error = 0.0
    for name in SPECS:
        if name == "case1a":
            continue
        with np.load(OUT / f"{name}.npz", allow_pickle=False) as g:
            if "dt" in g:
                actual = convection(g["input_temperature"], float(g["rayleigh"]),
                                    float(g["dt"]), int(g["steps"]), dtype=np.float32)
            else:
                actual = stokes(g["input_temperature"], float(g["rayleigh"]),
                                dtype=np.float32)
            actual = {key: np.asarray(value, dtype=np.float32)
                      for key, value in actual.items()}
            audit_outputs[name] = actual
            for key in actual:
                scale = float(np.max(np.abs(g[key])))
                error = float(np.max(np.abs(actual[key].astype(np.float64)-g[key])))
                if scale:
                    max_small_error = max(max_small_error, error/scale)
                    assert error/scale <= tol["PARITY_RTOL"], (name, key, error/scale)
                else:
                    assert error <= tol["ZERO_ATOL"], (name, key, error)

    audit_groups = {}
    for field in ("psi", "u", "w"):
        errors = []
        for n in (16, 32, 64):
            name = f"stokes_refine_n{n}"
            with np.load(OUT / f"{name}.npz", allow_pickle=False) as g:
                errors.append(normalized_error(audit_outputs[name][field],
                                               g[f"exact_{field}"]))
        audit_groups[f"stokes_{field}"] = errors
    for group, cases in (
        ("diffusion", [f"diffusion_n{n}" for n in (8, 16, 32)]),
        ("time", [f"time_s{s}" for s in (10, 20, 40)]),
        ("coupled", [f"coupled_refine_n{n}" for n in (8, 16, 32)]),
    ):
        errors = []
        for name in cases:
            with np.load(OUT / f"{name}.npz", allow_pickle=False) as g:
                actual = audit_outputs[name]["temperature"].astype(np.float64)
                if group == "coupled":
                    actual = (actual-g["input_temperature"])/float(g["dt"])
                    errors.append(normalized_error(actual, g["analytic_rate"]))
                else:
                    errors.append(normalized_error(actual-g["base"],
                                                   g["exact_temperature"]-g["base"]))
        audit_groups[group] = errors
    for group, errors in audit_groups.items():
        orders = np.log2(np.array(errors[:-1])/errors[1:])
        threshold = tol["MIN_TEMPORAL_ORDER"] if group == "time" else tol["MIN_SPATIAL_ORDER"]
        assert min(orders) >= threshold, (group, errors, orders)
    sanity["float32_audit"].update(max_small_case_parity_error=max_small_error,
                                    refinement_errors=audit_groups)

    meta = {
        "project": "48-mantle-convection", "seed": SEED,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "reference": {
            "citation": "Blankenbach et al. (1989), GJI 98, 23-38, case 1a",
            "doi": "10.1111/j.1365-246X.1989.tb04425.x",
            "published_nu": PUBLISHED_NU, "published_vrms": PUBLISHED_VRMS,
            "parameters": {"Ra": 10000, "aspect_ratio": 1, "viscosity": 1,
                           "internal_heating": 0, "Prandtl": "infinite"},
            "boundary_conditions": "All walls free slip and impermeable; Tbottom=1, Ttop=0; insulated sides.",
            "discrete_oracle": "Float64 SciPy DST-I double Poisson + centered differences + midpoint RK2.",
            "published_source_availability": "Paper not bundled; no network used. Scalars transcribed; verify citation/table at review.",
        },
        "precision": "Inputs saved float32, then promoted to float64 before reference computation; reference and analytic outputs float64.",
        "conventions": "Unit square [z,x], upward z; u=psi_z, w=-psi_x. No internal heating.",
        "tolerances": {name: {"value": tol[name], "reason": reason}
                       for name, reason in tol["REASONS"].items()},
        "cases": SPECS, "sanity": sanity,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(OUT.glob("*.npz"))},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    total = sum(p.stat().st_size for p in OUT.iterdir())
    assert total < 20_000_000
    print(json.dumps({"versions": meta["versions"], "sanity": sanity,
                      "golden_bytes": total, "seconds": time.monotonic()-started},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
