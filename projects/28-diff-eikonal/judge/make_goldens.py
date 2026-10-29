"""Float64 discrete adjoint oracle, independently checked by FD and sweeping.

Reference equations: first-order upwind Godunov eikonal residual; solve
J_T.T lambda = P.T W(PT-d), then g_s = 2*s*lambda. This is the
adjoint-state tomography judge the project specifies, not an autodiff oracle.
Heap-ordered marching supplies the causal reference solution; an independent
Gauss-Seidel fast-sweeping implementation checks its fixed point. No network.
An additional all-float32 sweep/adjoint audit checks the API's required stable
increment update on every saved base and perturbed model, not just the loss.
"""
from __future__ import annotations

import heapq
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tolerances", ROOT / "tests/tolerances.py")
tol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tol)
SEED = 2801


def neighbors(index, shape):
    for axis in range(2):
        for sign in (-1, 1):
            q = list(index)
            q[axis] += sign
            if 0 <= q[axis] < shape[axis]:
                yield tuple(q)


def axis_minima(t, index):
    result = []
    for axis in range(2):
        options = []
        for sign in (-1, 1):
            q = list(index)
            q[axis] += sign
            if 0 <= q[axis] < t.shape[axis]:
                q = tuple(q)
                options.append((t[q], q))
        result.append(min(options))
    return result


def local_root(t, index, s, h):
    pairs = sorted((v, h[d]) for d, (v, _) in enumerate(axis_minima(t, index)))
    a, ha = pairs[0]
    b, hb = pairs[1]
    if not np.isfinite(a):
        return np.inf
    one = a + s * ha
    if one <= b:
        return one
    # Stable quadratic using T=a+u, rather than subtracting two large times.
    delta = b - a
    aa = 1 / ha**2 + 1 / hb**2
    bb = -2 * delta / hb**2
    cc = delta**2 / hb**2 - s**2
    return a + (-bb + np.sqrt(bb**2 - 4 * aa * cc)) / (2 * aa)


def marching(s, h, source):
    t = np.full(s.shape, np.inf)
    accepted = np.zeros(s.shape, bool)
    source = tuple(source)
    t[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        value, i = heapq.heappop(heap)
        if accepted[i]:
            continue
        accepted[i] = True
        t[i] = value
        for q in neighbors(i, s.shape):
            if not accepted[q]:
                # Only accepted values may enter a marching update.
                old = t[q]
                masked = np.where(accepted, t, np.inf)
                candidate = local_root(masked, q, s[q], h)
                if candidate < old:
                    t[q] = candidate
                    heapq.heappush(heap, (candidate, q))
    assert np.isfinite(t).all()
    return t


def sweeping(s, h, source):
    s = np.asarray(s, np.float64)
    t = np.full(s.shape, np.inf)
    source = tuple(source)
    t[source] = 0
    for _ in range(200):
        old = t.copy()
        for rz in (False, True):
            for rx in (False, True):
                for z in sorted(range(s.shape[0]), reverse=rz):
                    for x in sorted(range(s.shape[1]), reverse=rx):
                        i = (z, x)
                        if i != source:
                            t[i] = min(t[i], local_root(t, i, s[i], h))
        if np.max(np.abs(t - old)) <= tol.ORACLE_SWEEP_ATOL:
            return t
    raise RuntimeError("Sweeping did not converge")


def local_root_float32(t, index, s, h):
    """Audit the required increment form with float32 at EVERY operation.

    Keep this separate from local_root: promoting its intermediates to float64
    and rounding only the final answer would hide the failure found in an earlier review.
    Python numeric constants must not promote the arithmetic here.
    """
    f = np.float32
    pairs = sorted((v, h[d]) for d, (v, _) in enumerate(axis_minima(t, index)))
    a, ha = pairs[0]
    b, hb = pairs[1]
    if not np.isfinite(a):
        return f(np.inf)
    one = a + s * ha
    if one <= b:
        return one
    delta = b - a
    ia, ib = f(1) / (ha * ha), f(1) / (hb * hb)
    aa = ia + ib
    bb = f(-2) * delta * ib
    cc = delta * delta * ib - s * s
    result = a + (-bb + np.sqrt(bb * bb - f(4) * aa * cc)) / (f(2) * aa)
    assert result.dtype == np.float32
    return result


def sweeping_float32(s, h, source):
    """Deterministic Gauss-Seidel sweeps to an unchanged float32 fixed point."""
    s, h = np.asarray(s, np.float32), np.asarray(h, np.float32)
    t = np.full(s.shape, np.inf, np.float32)
    source = tuple(source)
    t[source] = np.float32(0)
    for _ in range(200):
        old = t.copy()
        for rz in (False, True):
            for rx in (False, True):
                for z in sorted(range(s.shape[0]), reverse=rz):
                    for x in sorted(range(s.shape[1]), reverse=rx):
                        i = (z, x)
                        if i != source:
                            t[i] = min(t[i], local_root_float32(t, i, s[i], h))
        if np.array_equal(t, old):
            assert np.isfinite(t).all()
            return t
    raise RuntimeError("Float32 audit sweeping did not reach an unchanged fixed point")


def solve_float32(s, h, sources, receivers, observed, weights):
    """Audit float32 sweeps, loss reduction and discrete adjoint arithmetic.

    These diagnostics never replace the float64 goldens. They establish that
    the locked FD probes are feasible with the specified float32 local root.
    """
    f = np.float32
    s, h = np.asarray(s, np.float32), np.asarray(h, np.float32)
    ts = np.array([sweeping_float32(s, h, src) for src in sources], np.float32)
    predicted = ts[:, receivers[:, 0], receivers[:, 1]]
    residual = predicted - observed
    objective = f(0.5) * np.sum(weights * residual * residual, dtype=np.float32)
    gradient = np.zeros_like(s)
    for shot, source in enumerate(sources):
        t = ts[shot]
        rhs = np.zeros_like(s)
        np.add.at(rhs, (receivers[:, 0], receivers[:, 1]), weights[shot] * residual[shot])
        for flat in np.argsort(t, axis=None)[::-1]:
            i = np.unravel_index(flat, s.shape)
            if i == tuple(source):
                continue
            active = [(q, f(2) * (t[i] - v) / (h[d] * h[d]))
                      for d, (v, q) in enumerate(axis_minima(t, i)) if t[i] > v]
            diagonal = sum((c for _, c in active), start=f(0))
            lam = rhs[i] / diagonal
            gradient[i] += f(2) * s[i] * lam
            for q, c in active:
                rhs[q] += c * lam
    out = {"traveltime": ts, "predicted": predicted,
           "objective": np.array(objective), "gradient": gradient}
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in out.values())
    return out


def relative_error(a, b):
    """Global infinity-norm relative error, matching the frozen harness."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a-b)) / np.max(np.abs(b)))


def solve(s, h, sources, receivers, observed, weights):
    ts = np.array([marching(s, h, source) for source in sources])
    predicted = ts[:, receivers[:, 0], receivers[:, 1]]
    residual = predicted - observed
    objective = 0.5 * np.sum(weights * residual**2)
    gradient = np.zeros_like(s)
    for shot, source in enumerate(sources):
        t = ts[shot]
        rhs = np.zeros_like(s)
        np.add.at(rhs, (receivers[:, 0], receivers[:, 1]), weights[shot] * residual[shot])
        # J_T is triangular in increasing arrival time. Back substitution
        # of its transpose propagates adjoint loads towards the source.
        for flat in np.argsort(t, axis=None)[::-1]:
            i = np.unravel_index(flat, s.shape)
            if i == tuple(source):
                continue  # fixed Dirichlet row; no slowness sensitivity
            active = [(q, 2 * (t[i] - v) / h[d]**2)
                      for d, (v, q) in enumerate(axis_minima(t, i)) if t[i] > v]
            diagonal = sum(c for _, c in active)
            lam = rhs[i] / diagonal
            gradient[i] += 2 * s[i] * lam
            for q, c in active:
                rhs[q] += c * lam
    return {"traveltime": ts, "predicted": predicted, "objective": np.array(objective), "gradient": gradient}


def main():
    rng = np.random.default_rng(SEED)
    outdir = ROOT / "judge/golden"
    outdir.mkdir(exist_ok=True)
    reports = {}
    cases = [("homogeneous", (7, 9), [40., 55.], [[0, 0]]),
             ("smooth_multishot", (11, 13), [35., 50.], [[0, 1], [10, 11], [5, 6]]),
             ("layered", (13, 10), [60., 30.], [[0, 0], [12, 9]])]
    for name, shape, spacing, source_list in cases:
        z, x = np.indices(shape)
        if name == "homogeneous":
            s = np.full(shape, 1 / 2000, np.float32)
        elif name == "smooth_multishot":
            s = (0.00046 * (1 + 0.14 * np.sin(0.37*x+0.19*z) + 0.006*rng.normal(size=shape))).astype(np.float32)
        else:
            s = (0.00055 - 0.000009*z + 0.000012*np.sin(0.7*x+0.21*z)).astype(np.float32)
        sources = np.array(source_list, np.int32)
        receivers = np.array([[0, 0], [shape[0]-1, shape[1]-1], [0, shape[1]-1],
                              [shape[0]-1, 0], [shape[0]//2, shape[1]//2],
                              [2, shape[1]-2], [shape[0]-2, 2], [0, 0]], np.int32)
        h = np.array(spacing)
        truth = s.astype(np.float64) * (1 + 0.07*np.cos(0.31*x+0.43*z))
        truth_t = np.array([marching(truth, h, src) for src in sources])
        observed = (truth_t[:, receivers[:, 0], receivers[:, 1]] +
                    0.012*np.sin(np.arange(len(sources)*len(receivers)).reshape(len(sources), -1)+0.4)).astype(np.float32)
        weights = rng.uniform(0.25, 1.5, observed.shape).astype(np.float32)
        weights[0, 0] = 0
        ref = solve(s.astype(np.float64), h, sources, receivers, observed, weights)
        audit = solve_float32(s, h, sources, receivers, observed, weights)
        audit_parity = {key: relative_error(audit[key], ref[key]) for key in ref}
        assert max(audit_parity.values()) <= tol.PARITY_REL, (name, audit_parity)
        sweep_error = 0.
        residual_error = 0.
        for shot, src in enumerate(sources):
            t = ref["traveltime"][shot]
            sweep_error = max(sweep_error, float(np.max(np.abs(sweeping(s, h, src)-t))/np.max(t)))
            for i in np.ndindex(shape):
                if i != tuple(src):
                    lhs = sum((max(t[i]-a, 0)/h[d])**2 for d, (a, _) in enumerate(axis_minima(t, i)))
                    residual_error = max(residual_error, abs(lhs/float(s[i])**2-1))
        assert sweep_error <= tol.ORACLE_FORWARD_REL, sweep_error
        assert residual_error <= tol.ORACLE_RESIDUAL_REL, residual_error
        if name == "homogeneous":
            assert np.max(np.abs(ref["traveltime"][0, 0] - np.arange(shape[1])*h[1]*float(s[0, 0]))) <= tol.ORACLE_RESIDUAL_REL
        # Two dense directions and three coordinate probes with nonzero sensitivity.
        # High-signal dense probes avoid cancellation in float32 loss differences.
        directions = [np.ones(shape), np.where(ref["gradient"] >= 0, 1., -1.)]
        for flat in np.argsort(np.abs(ref["gradient"]), axis=None)[-3:]:
            direction = np.zeros(shape)
            direction.flat[flat] = 1
            directions.append(direction)
        directions = np.array(directions, np.float32)
        steps = np.array(tol.FD_STEPS)
        plus = (s + steps[:, None, None]*directions).astype(np.float32)
        minus = (s - steps[:, None, None]*directions).astype(np.float32)
        fd_values, fd_errors, quantization_errors = [], [], []
        sweep_fd_errors, sweep_reference_errors = [], []
        for k, direction in enumerate(directions):
            # Verify that a finite-difference step is exactly the advertised
            # direction after serialization to the float32 implementation API.
            np.testing.assert_array_equal((plus[k].astype(np.float64)-minus[k])/(2*steps[k]), direction)
            op = solve(plus[k].astype(np.float64), h, sources, receivers, observed, weights)
            om = solve(minus[k].astype(np.float64), h, sources, receivers, observed, weights)
            fp, fm = op["objective"], om["objective"]
            fd = float((fp-fm)/(2*steps[k]))
            dot = float(np.sum(ref["gradient"]*direction))
            error = abs(fd-dot)/max(abs(fd), abs(dot))
            assert error <= tol.ORACLE_FD_REL, (name, k, error)
            fd_values.append(fd)
            fd_errors.append(error)
            # Audit unavoidable float32 output/loss rounding before locking h.
            def rounded_loss(output, observed=observed, weights=weights):
                r = output["predicted"].astype(np.float32) - observed
                return np.float32(0.5) * np.sum(weights * r**2, dtype=np.float32)
            rounded_fd = (float(rounded_loss(op))-float(rounded_loss(om)))/(2*steps[k])
            quantization_errors.append(abs(rounded_fd-dot)/abs(dot))
            # Unlike the rounded-loss audit above, round every update, sweep,
            # adjoint operation and loss reduction to float32. Mirror BOTH
            # acceptance conditions in test_centered_finite_difference.
            ap = solve_float32(plus[k], h, sources, receivers, observed, weights)
            am = solve_float32(minus[k], h, sources, receivers, observed, weights)
            audit_fd = (float(ap["objective"])-float(am["objective"]))/(2*steps[k])
            audit_dot = float(np.sum(audit["gradient"].astype(np.float64)*direction))
            sweep_fd_errors.append(abs(audit_fd-audit_dot)/max(abs(audit_fd), abs(audit_dot)))
            sweep_reference_errors.append(abs(audit_fd-fd)/abs(fd))
        assert max(quantization_errors) <= tol.FD_REL, (name, quantization_errors)
        assert max(sweep_fd_errors) <= tol.FD_REL, (name, sweep_fd_errors)
        assert max(sweep_reference_errors) <= tol.FD_REL, (name, sweep_reference_errors)
        np.savez(outdir / f"{name}.npz", slowness=s, spacing=h, sources=sources,
                 receivers=receivers, observed=observed, weights=weights,
                 **ref, directions=directions, fd_plus=plus, fd_minus=minus,
                 fd_reference=np.array(fd_values))
        reports[name] = {"shape": shape, "spacing": spacing, "sources": source_list,
                        "receivers": receivers.tolist(), "slowness_range": [float(s.min()), float(s.max())],
                        "time_range": [float(ref["traveltime"].min()), float(ref["traveltime"].max())],
                        "objective": float(ref["objective"]), "gradient_norm": float(np.linalg.norm(ref["gradient"])),
                        "sweep_relative_error": sweep_error, "residual_relative_error": residual_error,
                        "fd_relative_errors": fd_errors,
                        "float32_rounded_loss_fd_errors": quantization_errors,
                        "float32_stable_sweep_parity": audit_parity,
                        "float32_stable_sweep_fd_gradient_errors": sweep_fd_errors,
                        "float32_stable_sweep_fd_reference_errors": sweep_reference_errors}
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pytest")}
    meta = {"seed": SEED, "python": platform.python_version(), "libraries": versions,
            "oracle": "Float64 causal Godunov marching + discrete adjoint-state transpose solve; independent sweeping and centered FD",
            "conventions": "See tests/api.py; source node fixed at zero; gradient with respect to nodal slowness; no regularization",
            "precision": "Inputs rounded to float32 before float64 reference evaluation; binary FD increments exact for these inputs",
            "required_root_evaluation": "Stable increment T=a+u with delta=b-a; see tests/api.py. Absolute-T quadratic expansion is excluded.",
            "float32_audit": "Every local root, sweep, adjoint and objective operation is float32; sweeps stop at bitwise unchanged fields. Diagnostic only; goldens remain float64.",
            "tolerances_source": "tests/tolerances.py", "tolerances": {k: v for k, v in vars(tol).items() if k.isupper()},
            "tolerance_reasons": (ROOT / "tests/tolerances.py").read_text(), "cases": reports}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps({"libraries": versions, "cases": reports}, indent=2))


if __name__ == "__main__":
    main()
