"""Float64 SciPy Hilbert oracle and analytic time-shift cycle-skipping judge.

No wave-equation library is used: this is an explicitly restricted one-parameter
arrival inversion. A homogeneous nondispersive path translates its source trace
by its travel time. It is not integrated with project 13's wave-equation propagator.
"""

from __future__ import annotations

import importlib.util
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bounds", ROOT / "tests/tolerances.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
DEST = ROOT / "judge/golden"
SEED = 2301


def oracle(predicted, observed, dt, kind, epsilon):
    """Independent analytic adjoint: H.T = -H, including odd/even lengths."""
    x, y = np.asarray(predicted, np.float64), np.asarray(observed, np.float64)
    weight = dt / len(x)
    if kind == "l2":
        residual = x - y
        gradient = weight * residual
    else:
        hx, hy = hilbert(x, axis=-1).imag, hilbert(y, axis=-1).imag
        ex = np.sqrt(x*x + hx*hx + epsilon*epsilon)
        ey = np.sqrt(y*y + hy*hy + epsilon*epsilon)
        residual = ex - ey
        scaled = residual / ex
        gradient = weight * (scaled*x - hilbert(scaled*hx, axis=-1).imag)
    return {"objective": np.asarray(weight * np.sum(residual**2) / 2),
            "gradient": gradient}


def float32_audit(predicted, observed, dt, kind, epsilon):
    """Feasibility probe only; NEVER the reference used to write expected values.

    Evaluate the literal API formula, including envelope subtraction, entirely
    in float32 (SciPy's FFT Hilbert preserves complex64 for float32 input).
    This deliberately does not use a cancellation-free residual rewrite.
    It models CPU float32 arithmetic, not any particular GPU FFT/reduction.
    """
    x, y = np.asarray(predicted, np.float32), np.asarray(observed, np.float32)
    weight = np.float32(dt / len(x))
    if kind == "l2":
        residual = x - y
        gradient = weight * residual
    else:
        ax, ay = hilbert(x, axis=-1), hilbert(y, axis=-1)
        assert ax.dtype == ay.dtype == np.dtype("complex64")
        hx, hy = ax.imag, ay.imag
        eps = np.float32(epsilon)
        ex = np.sqrt(x*x + hx*hx + eps*eps)
        ey = np.sqrt(y*y + hy*hy + eps*eps)
        residual = ex - ey
        scaled = residual / ex
        gradient = weight * (scaled*x - hilbert(scaled*hx, axis=-1).imag)
    objective = weight * np.sum(residual**2, dtype=np.float32) / np.float32(2)
    assert objective.dtype == gradient.dtype == np.dtype("float32")
    return {"objective": objective, "gradient": gradient}


def relative_error(actual, expected):
    """Global infinity-norm relative error; all contractions are float64."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))


def audit_fd(x, y, dt, kind, epsilon, directions, h, expected):
    """Check float64 truncation separately from actual float32 arithmetic noise."""
    base = float32_audit(x, y, dt, kind, epsilon)
    fd32, fd64 = [], []
    for direction in directions:
        # Exactly the perturbations constructed in test_centered_gradient.
        plus = float32_audit((x+h*direction).astype(np.float32), y, dt, kind, epsilon)
        minus = float32_audit((x-h*direction).astype(np.float32), y, dt, kind, epsilon)
        fd32.append((float(plus["objective"])-float(minus["objective"]))/ (2*h))
        # Unrounded float64 perturbations isolate central-FD truncation error.
        plus64 = oracle(x.astype(np.float64)+h*direction.astype(np.float64),
                        y, dt, kind, epsilon)
        minus64 = oracle(x.astype(np.float64)-h*direction.astype(np.float64),
                         y, dt, kind, epsilon)
        fd64.append((float(plus64["objective"])-float(minus64["objective"]))/ (2*h))
    adjoint32 = np.sum(base["gradient"].astype(np.float64)[None] * directions,
                       axis=(1, 2))
    errors = {
        "fd_step": h,
        "float64_locked_step_truncation_error": relative_error(fd64, expected),
        "literal_float32_fd_vs_float32_adjoint_error": relative_error(fd32, adjoint32),
        "literal_float32_fd_vs_float64_adjoint_error": relative_error(fd32, expected),
        "literal_float32_objective_parity_error": relative_error(base["objective"],
            oracle(x, y, dt, kind, epsilon)["objective"]),
        "literal_float32_gradient_parity_error": relative_error(base["gradient"],
            oracle(x, y, dt, kind, epsilon)["gradient"]),
    }
    for key in ("float64_locked_step_truncation_error",
                "literal_float32_fd_vs_float32_adjoint_error",
                "literal_float32_fd_vs_float64_adjoint_error"):
        assert errors[key] <= T.FD_AUDIT_RTOL, (kind, key, errors)
    for key in ("literal_float32_objective_parity_error",
                "literal_float32_gradient_parity_error"):
        assert errors[key] <= T.PARITY_RTOL, (kind, key, errors)
    return errors


def recover(delays, traces, observed, initial, kind):
    """Float64 independent local march and stationary-point bisection."""
    def slope(delay):
        i = int(np.clip(np.searchsorted(delays, delay, side="right") - 1,
                        0, len(delays) - 2))
        derivative = (traces[i+1] - traces[i]) / (delays[i+1] - delays[i])
        trace = traces[i] + (delay-delays[i])*derivative
        out = oracle(trace[None], observed, T.TRACE_DT, kind, T.ENVELOPE_EPSILON)
        return float(np.sum(out["gradient"] * derivative))

    current = float(initial)
    grad = slope(current)
    if grad == 0:
        return current
    step = -np.sign(grad) * T.SEARCH_STEP_SECONDS
    for _ in range(T.SEARCH_MAX_STEPS):
        candidate = current + step
        if not delays[0] <= candidate <= delays[-1]:
            return current
        next_grad = slope(candidate)
        if grad * next_grad <= 0:
            left, right = sorted((current, candidate))
            left_grad = slope(left)
            for _ in range(T.SEARCH_BISECTIONS):
                middle = (left+right)/2
                mid_grad = slope(middle)
                if left_grad * mid_grad <= 0:
                    right = middle
                else:
                    left, left_grad = middle, mid_grad
                if right-left <= T.SEARCH_STOP_SECONDS:
                    break
            return (left+right)/2
        current, grad = candidate, next_grad
    raise AssertionError("oracle inversion exceeded budget")


def main():
    DEST.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    cases = {}
    audits = {}
    # Quantize INPUTS once: oracle then evaluates their exact float32 values in
    # float64. This avoids conflating tiny input perturbations with GPU error.
    for name, shape, scale in [("odd", (2, 63), 1.0), ("even", (3, 64), 1.0),
                                ("small", (1, 32), 1e-4)]:
        x = (scale * rng.normal(size=shape)).astype(np.float32)
        y = (scale * rng.normal(size=shape)).astype(np.float32)
        # Deliberately include DC and Nyquist energy in the even fixture.
        if name == "even":
            x += np.asarray(0.4 + 0.2*(-1.)**np.arange(shape[1]), np.float32)
        directions = rng.normal(size=(3, *shape))
        directions /= np.max(np.abs(directions), axis=(1, 2), keepdims=True)
        directions *= scale
        directions = directions.astype(np.float32)
        data = {"predicted": x, "observed": y, "directions": directions,
                "dt": np.asarray(0.004), "epsilon": np.asarray(T.ENVELOPE_EPSILON)}
        for kind in ("l2", "envelope"):
            h_locked = (T.FD_L2_STEP if kind == "l2" else
                        T.FD_SMALL_ENVELOPE_STEP if name == "small" else T.FD_ENVELOPE_STEP)
            data[f"{kind}_fd_step"] = np.asarray(h_locked)
            out = oracle(x, y, 0.004, kind, T.ENVELOPE_EPSILON)
            data.update({f"{kind}_{key}": value for key, value in out.items()})
            fd, analytic = [], []
            for direction in directions:
                h = T.ORACLE_FD_STEP
                d64 = direction.astype(np.float64)
                plus = oracle(x.astype(float)+h*d64, y, 0.004, kind, T.ENVELOPE_EPSILON)
                minus = oracle(x.astype(float)-h*d64, y, 0.004, kind, T.ENVELOPE_EPSILON)
                fd.append((plus["objective"]-minus["objective"])/(2*h))
                analytic.append(np.sum(out["gradient"]*direction))
            error = np.max(np.abs(np.array(fd)-analytic))/np.max(np.abs(analytic))
            assert error < T.ORACLE_GRAD_RTOL, (name, kind, error)
            data[f"{kind}_directional"] = np.asarray(analytic)
            audits[f"{name}_{kind}"] = {"float64_adjoint_fd_error": float(error),
                                       **audit_fd(x, y, 0.004, kind, T.ENVELOPE_EPSILON,
                                                  directions, h_locked, analytic),
                                       "objective": float(out["objective"]),
                                       "gradient_max": float(np.max(np.abs(out["gradient"])))}
        np.savez_compressed(DEST / f"{name}.npz", **data)
        cases[name] = {"shape": list(shape), "amplitude_scale": scale,
                       "direction_count": len(directions),
                       "direction_max_abs": np.max(np.abs(directions), axis=(1, 2)).tolist(),
                       "fd_steps": {kind: float(data[f"{kind}_fd_step"])
                                    for kind in ("l2", "envelope")}}

    zero = np.zeros((2, 64), dtype=np.float32)
    nonzero = rng.normal(size=zero.shape).astype(np.float32)
    boundary = {"zero": zero, "nonzero": nonzero, "dt": np.asarray(0.004),
                "epsilon": np.asarray(T.ENVELOPE_EPSILON)}
    for label, observed_zero in [("zero_prediction", nonzero), ("zero_pair", zero)]:
        for kind in ("l2", "envelope"):
            out = oracle(zero, observed_zero, 0.004, kind, T.ENVELOPE_EPSILON)
            for key, value in out.items():
                boundary[f"{label}_{kind}_{key}"] = value
    np.savez_compressed(DEST / "boundary.npz", **boundary)

    # Gaussian-modulated 10 Hz wavelet, width .05 s, center .512 s.
    # Translate on a dense saved dictionary; tests never synthesize traces.
    delays = np.linspace(-0.18, 0.18, 1441)
    time = np.arange(512) * T.TRACE_DT
    tau = time[None, :] - 0.512 - delays[:, None]
    traces = (np.exp(-0.5*(tau/0.05)**2)*np.cos(2*np.pi*10*tau)).astype(np.float32)
    observed = traces[720:721].copy()
    initial = np.arange(1, 13, dtype=float)*0.01
    finals = {}
    basins = {}
    for kind in ("l2", "envelope"):
        result = np.array([[recover(delays, traces.astype(float), observed, sign*s, kind)
                            for s in initial] for sign in (-1, 1)])
        success = np.abs(result) <= T.RECOVERY_SECONDS
        basin = []
        for row in success:
            first_fail = np.flatnonzero(~row)
            count = int(first_fail[0]) if len(first_fail) else len(row)
            basin.append(float(initial[count-1]) if count else 0.)
        finals[f"{kind}_final"] = result
        finals[f"{kind}_success"] = success
        basins[kind] = basin
    assert min(basins["l2"]) >= T.MIN_L2_BASIN_SECONDS
    ratios = np.array(basins["envelope"]) / basins["l2"]
    assert min(ratios) >= T.MIN_BASIN_RATIO
    probe_indices = np.array([240, 480, 640, 800, 960, 1200], dtype=np.int64)
    for kind in ("l2", "envelope"):
        probes = [oracle(traces[i:i+1], observed, T.TRACE_DT, kind,
                         T.ENVELOPE_EPSILON) for i in probe_indices]
        finals[f"{kind}_probe_objective"] = np.array([p["objective"] for p in probes])
        finals[f"{kind}_probe_gradient"] = np.concatenate([p["gradient"] for p in probes])
    np.savez_compressed(DEST / "cycle.npz", delays=delays, traces=traces,
                        observed=observed, initial=initial, dt=np.asarray(T.TRACE_DT),
                        epsilon=np.asarray(T.ENVELOPE_EPSILON),
                        probe_indices=probe_indices, **finals)
    reasons = {"PARITY_RTOL": "Project-wide float32 relative parity limit",
               "ZERO_ATOL": "absolute safeguard only for analytically exact zeros",
               "GRAD_RTOL": "Project-wide centered finite-difference gradient limit",
               "ORACLE_GRAD_RTOL": "float64 analytic Hilbert-adjoint verification; tighter than GPU bound",
               "ORACLE_FD_STEP": "small float64 step with negligible truncation and measured cancellation",
               "RECOVERY_SECONDS": "one quarter of a trace sampling interval",
               "MIN_BASIN_RATIO": "mandatory project pass threshold",
               "MIN_L2_BASIN_SECONDS": "requires at least the smallest tested L2 delay to converge",
               "FD_L2_STEP": "quadratic objective permits larger step without truncation; reduces float32 noise",
               "FD_ENVELOPE_STEP": "unit-scale envelope: float32 subtraction noise and float64 truncation audited",
               "FD_SMALL_ENVELOPE_STEP": "below epsilon: larger step resolves literal float32 envelope subtraction",
               "FD_AUDIT_RTOL": "at least 3x margin for literal float32 FD and float64 truncation audits",
               "SEARCH_STEP_SECONDS": "1/50 carrier period; local march cannot jump a carrier cycle",
               "SEARCH_STOP_SECONDS": "bracket uncertainty negligible beside arrival recovery threshold",
               "SEARCH_MAX_STEPS": "covers entire dictionary from every tested initial delay",
               "SEARCH_BISECTIONS": "ample iterations to resolve the fixed 2 ms stationary-point bracket",
               "ENVELOPE_EPSILON": "smooth finite derivative at silent samples; small versus unit signal amplitude",
               "TRACE_DT": "50 samples per 10 Hz carrier period"}
    cycle_ranges = {}
    for kind in ("l2", "envelope"):
        values = [float(oracle(traces[i:i+1], observed, T.TRACE_DT, kind,
                               T.ENVELOPE_EPSILON)["objective"])
                  for i in (240, 480, 720, 960, 1200)]
        assert min(values) >= 0 and values[2] == 0
        cycle_ranges[kind] = values
    meta = {"seed": SEED, "versions": {"python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pytest": version("pytest")},
            "reference": "SciPy signal.hilbert (periodic discrete Hilbert); analytic Gaussian carrier translation",
            "inputs": cases, "cycle": {"dt_seconds": T.TRACE_DT, "ntime": 512,
            "frequency_hz": 10., "gaussian_width_seconds": 0.05,
            "center_seconds": 0.512, "delay_grid_seconds": [-0.18, 0.18, 0.00025],
            "initial_seconds": initial.tolist(), "signs": [-1, 1],
            "basins_seconds": basins, "basin_ratios": ratios.tolist(),
            "convergence": "contiguous tested delays from zero, both signs; abs(final delay)<=RECOVERY_SECONDS",
            "scope": "single homogeneous-path arrival; no spatial velocity or PDE validation"},
            "cycle_objectives_at_delays_minus120_minus60_zero_60_120_ms": cycle_ranges,
            "derivative_audits": audits,
            "float32_audit_scope": "literal formula, SciPy complex64 FFT Hilbert, float32 envelope and reductions; CPU feasibility only, no GPU pass claimed",
            "tolerances_and_controls": {k: {"value": v, "reason": reasons[k]}
                for k, v in vars(T).items() if k.isupper()},
            "total_npz_bytes": sum(p.stat().st_size for p in DEST.glob("*.npz"))}
    (DEST / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps({"basins_seconds": basins, "ratios": ratios.tolist(),
                      "derivative_audits": audits,
                      "golden_bytes": meta["total_npz_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
