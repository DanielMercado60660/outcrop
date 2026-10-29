"""Float64 Deepwave scalar oracle; run with the repository's pinned Python.

No implementation is imported. Fixtures are saved after float32 quantization of
inputs, so parity measures implementation arithmetic rather than input rounding.
"""
from __future__ import annotations

import hashlib
import json
import platform
import runpy
import time
import warnings
from importlib import metadata
from pathlib import Path

import deepwave
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "judge" / "golden"
TOL = runpy.run_path(str(ROOT / "tests" / "tolerances.py"))
SEED = 50001316
MEMORY_NAMES = ("psiy_m1", "psix_m1", "zetay_m1", "zetax_m1")
REASONS = {
    "PARITY_REL": "Project-wide float32 max-absolute error / max-absolute reference at identical discretization, in physical field units.",
    "RENDER_ROUND_ABS": "Four float32 epsilons for colormap division/storage; absolute color bound is PARITY_REL*peak pressure/render_scale plus this allowance (colormap Lipschitz gain 1/render_scale).",
    "RENDER_AUDIT_MARGIN_MIN": "Adopted rule: Deepwave float32 and independent written-recurrence float32 must each have at least 3x render headroom.",
    "ORACLE_CALIBRATION_REL": "Float64 reference source-sign/scale calibration against the exact first step.",
    "ORACLE_SCHEME_REL": "Oracle-only NumPy evaluation of SCHEME.md vs Deepwave; reordered float64 operations over 160 steps.",
    "EXACT_ABS": "Zero-input linear physics and constant render channels are exact.",
    "GRAD_FD_REL": "Project-wide centered finite difference gradient threshold for float32.",
    "FD_STEP_MPS": "2 m/s centered perturbation balances float32 objective cancellation and truncation.",
    "GRAD_COS_MIN": "Project 16 direction agreement, applied to live FWI gradient.",
    "CONVERGENCE_ORDER": "Fourth-order space with dt proportional to h has second-order total error.",
    "CONVERGENCE_ORDER_SLACK": "Project-wide order acceptance band of 0.2.",
    "MIN_STEPS_PER_SECOND": "Project 50: at least 60 simulation+render frames per second.",
    "PERFORMANCE_STEPS": "240 resident steps amortize launch; full wall time is charged.",
    "PERFORMANCE_GRID_SIZE": "Project 50 mandates 1000 by 1000 physical cells.",
    "DESCENT_RATIO_MAX": "One conservative oracle-selected live FWI step must reduce objective by at least 1%.",
}


def render(u, scale):
    q = np.clip(u / scale, -1.0, 1.0)
    return np.stack((np.maximum(q, 0), np.zeros_like(q),
                     np.maximum(-q, 0), np.ones_like(q)), axis=-1)


def documented_reference(case, *, dtype=np.float64):
    """Independent expression of tests/SCHEME.md; no Deepwave helpers.

    Used only to validate the written contract, never to replace Deepwave as
    the golden oracle. Audit small cases, including nonzero resumed memories.
    """
    order = int(case["accuracy"])
    g, w = order//2, int(case["pml_width"])
    periodic = str(case["boundary"]) == "periodic"
    first, second = {
        2: ([1/2], [-2, 1]),
        4: ([2/3, -1/12], [-5/2, 4/3, -1/12]),
        6: ([3/4, -3/20, 1/60], [-49/18, 3/2, -3/20, 1/90]),
        8: ([4/5, -1/5, 4/105, -1/280],
            [-205/72, 8/5, -1/5, 8/315, -1/560]),
    }[order]

    def derivative(field, axis, spacing, *, twice=False):
        spacing = dtype(spacing)
        result = second[0]*field if twice else np.zeros_like(field)
        for j, coef in enumerate(second[1:] if twice else first, 1):
            plus, minus = np.roll(field, -j, axis), np.roll(field, j, axis)
            result = result + coef*(plus+minus if twice else plus-minus)
        result = result/spacing**(2 if twice else 1)
        if not periodic:
            ends = [slice(None)]*field.ndim
            for end in (slice(0, g), slice(-g, None)):
                ends[axis] = end
                result[tuple(ends)] = 0
        return result

    state = case["state"].astype(dtype)
    velocity = np.pad(case["velocity"].astype(dtype), w, mode="edge")
    offset = w
    if not periodic:
        state = np.pad(state, ((0, 0), (g, g), (g, g)))
        velocity = np.pad(velocity, g, mode="edge")
        offset += g
    ny, nx = case["velocity"].shape
    profiles = []
    for axis, (n, spacing) in enumerate(zip(velocity.shape, case["grid_spacing"])):
        if periodic:
            a, b, c = (np.zeros(n, dtype=dtype) for _ in range(3))
        else:
            i = np.arange(n, dtype=dtype)
            ql = np.clip((g+w-i)/w, 0, 1)
            qr = np.clip((i-(n-1-g-w))/w, 0, 1)
            q = np.maximum(ql, qr)
            sigma0 = dtype(-3*float(case["max_velocity"])*np.log(.001)/(2*w*float(spacing)))
            sigma = sigma0*(ql**2+qr**2)
            alpha = np.pi*float(case["pml_freq"])*(1-q)
            total = sigma+alpha
            a = np.where(q > 0, np.exp(-total*float(case["dt"])), 0)
            b = np.divide(sigma, total, out=np.zeros_like(total), where=total != 0)*np.expm1(-total*float(case["dt"]))
            c = derivative(b, 0, spacing)
        shape = (n, 1) if axis == 0 else (1, n)
        profiles.append(tuple(v.reshape(shape) for v in (a, b, c)))

    def project(fields):
        if periodic:
            fields[2:] = 0
            return
        fields[:, :g] = fields[:, -g:] = 0
        fields[:, :, :g] = fields[:, :, -g:] = 0
        for ch in (2, 4):
            fields[ch, offset:offset+ny, :] = 0
        for ch in (3, 5):
            fields[ch, :, offset:offset+nx] = 0

    project(state)
    src = case["source_locations"]+offset
    rec = case["receiver_locations"]+offset
    traces, textures = [], []
    for sample in case["source_amplitudes"].T:
        traces.append(state[0, rec[:, 0], rec[:, 1]].copy())
        new = np.zeros_like(state)
        spatial = np.zeros_like(velocity)
        for axis, (spacing, (a, b, c)) in enumerate(zip(case["grid_spacing"], profiles)):
            du = derivative(state[0], axis, spacing)
            t = ((1+b)*derivative(state[0], axis, spacing, twice=True)
                 + c*du + derivative(a*state[2+axis], axis, spacing))
            spatial += (1+b)*t + a*state[4+axis]
            new[2+axis] = b*du + a*state[2+axis]
            new[4+axis] = b*t + a*state[4+axis]
        new[0] = 2*state[0]-state[1] + velocity**2*float(case["dt"])**2*spatial
        new[0, src[:, 0], src[:, 1]] -= velocity[src[:, 0], src[:, 1]]**2*float(case["dt"])**2*sample
        new[1] = state[0]
        project(new)
        state = new
        textures.append(render(state[0, rec[:, 0], rec[:, 1]], float(case["render_scale"])))
    physical = state[0, offset:offset+ny, offset:offset+nx]
    return {"expected_state": state if periodic else state[:, g:-g, g:-g],
            "expected_receivers": np.stack(traces, axis=1),
            "expected_rgba": render(physical, float(case["render_scale"])),
            "expected_texture_samples": np.stack(textures)}


def render_pressures(fixture):
    """Float64 pre-colormap fields, with the same scopes as the tested outputs."""
    p = int(fixture["pml_width"])
    ny, nx = fixture["velocity"].shape
    physical = fixture["expected_state"][0, p:p+ny, p:p+nx]
    loc = fixture["receiver_locations"]
    samples = np.concatenate((fixture["expected_receivers"][:, 1:].T,
                              physical[loc[:, 0], loc[:, 1]][None]), axis=0)
    return {"expected_rgba": physical, "expected_texture_samples": samples}


def audit_render(fixture, result):
    """Verify propagated pressure budgets and keep >=3x float32 headroom."""
    audit = {}
    for key, pressure in render_pressures(fixture).items():
        bound = TOL["render_abs_tolerance"](
            float(np.max(np.abs(pressure))), float(fixture["render_scale"]))
        diff = float(np.max(np.abs(result[key].astype(np.float64)-fixture[key])))
        assert diff*TOL["RENDER_AUDIT_MARGIN_MIN"] <= bound, (key, diff, bound)
        audit[key] = {"absolute_error": diff, "absolute_bound": bound,
                      "budget_fraction": diff/bound,
                      "pressure_peak_au": float(np.max(np.abs(pressure)))}
    return audit


def audit_render_mutations(fixture):
    """Make sure the pressure-derived bound still rejects rendering defects."""
    mutations = {}
    scale = float(fixture["render_scale"])
    for key, pressure in render_pressures(fixture).items():
        bound = TOL["render_abs_tolerance"](float(np.max(np.abs(pressure))), scale)
        q = pressure/scale  # deliberately omit clip for the third mutation
        variants = {
            "swapped_red_blue": render(-pressure, scale),
            "doubled_render_scale": render(pressure, 2*scale),
            "omitted_clip": np.stack((np.maximum(q, 0), np.zeros_like(q),
                                       np.maximum(-q, 0), np.ones_like(q)), axis=-1),
        }
        mutations[key] = {}
        for name, actual in variants.items():
            error = float(np.max(np.abs(actual-fixture[key])))
            # Range violations are independently rejected by tests as well.
            mutations[key][name] = error > bound
    return mutations


def reference(case, *, grad=False, dtype=torch.float64):
    """Use only the public Deepwave scalar interface, not its stencil code."""
    v = torch.tensor(case["velocity"], dtype=dtype, requires_grad=grad)
    state = torch.tensor(case["state"], dtype=dtype)
    initial = {"wavefield_0": state[0:1], "wavefield_m1": state[1:2]}
    initial["auxiliary_state"] = deepwave.AuxiliaryState(
        {name: state[i+2:i+3] for i, name in enumerate(MEMORY_NAMES)},
        propagator="scalar", ndim=2, accuracy=int(case["accuracy"]),
        pml_width=(int(case["pml_width"]),)*4,
        boundary=(str(case["boundary"]),)*4, origin=(0, 0),
        grid_spacing=tuple(case["grid_spacing"]),
        pml_freq=float(case["pml_freq"]), max_vel=float(case["max_velocity"]),
        internal_dt=float(case["dt"]), detached_deliberately=True)
    # Use the public auxiliary bundle: this installed version rejects the
    # individual legacy keys despite scalar's docstring saying they are accepted.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        out = deepwave.scalar(
            v, case["grid_spacing"].tolist(), float(case["dt"]),
            source_amplitudes=torch.tensor(case["source_amplitudes"], dtype=dtype)[None],
            source_locations=torch.tensor(case["source_locations"], dtype=torch.int64)[None],
            receiver_locations=torch.tensor(case["receiver_locations"], dtype=torch.int64)[None],
            initial_state=initial, origin=[0, 0],
            accuracy=int(case["accuracy"]), pml_width=int(case["pml_width"]),
            pml_freq=float(case["pml_freq"]), max_vel=float(case["max_velocity"]),
            boundary=None if str(case["boundary"]) == "pml" else "periodic",
            model_gradient_sampling_interval=1, freq_taper_frac=0.0,
            time_pad_frac=0.0, time_taper=False, storage_compression=False,
        )
    assert out.info.step_ratio == 1, "CFL substepping would change the API recurrence"
    predicted = out.receiver_amplitudes[0]
    result = {"expected_receivers": predicted.detach().numpy()}
    if grad:
        objective = 0.5 * torch.sum((predicted - torch.tensor(case["observed"], dtype=dtype))**2)
        objective.backward()
        result.update(expected_objective=objective.detach().numpy(),
                      expected_gradient=v.grad.detach().numpy())
        return result
    final = np.stack([out.wavefield_0[0].detach().numpy(),
                      out.wavefield_m1[0].detach().numpy()] +
                     [out.auxiliary_state[n][0].detach().numpy() for n in MEMORY_NAMES])
    p = int(case["pml_width"])
    ny, nx = case["velocity"].shape
    physical = final[0, p:p+ny, p:p+nx]
    loc = case["receiver_locations"]
    # Deepwave receivers hold u_0 ... u_(nt-1); shift to POST-update probes.
    samples = np.concatenate((predicted.detach().numpy()[:, 1:],
                              physical[loc[:, 0], loc[:, 1]][:, None]), axis=1).T
    result.update(expected_state=final,
                  expected_rgba=render(physical, float(case["render_scale"])),
                  expected_texture_samples=render(samples, float(case["render_scale"])))
    return result


def base(ny, nx, nt, *, accuracy=4, pml=8, boundary="pml"):
    return {"velocity": np.full((ny, nx), 1800, np.float32),
                "state": np.zeros((6, ny+2*pml, nx+2*pml), np.float32),
                "source_amplitudes": np.zeros((2, nt), np.float32),
                "source_locations": np.array([[ny//3, nx//3], [2*ny//3, 2*nx//3]], np.int64),
                "receiver_locations": np.array([[0, 0], [ny-1, nx-1], [ny//3, nx//3],
                                             [2*ny//3, 2*nx//3], [ny//2, nx//2],
                                             [0, nx//2], [ny//2, 0]], np.int64),
                "grid_spacing": np.array([8., 9.], np.float64), "dt": np.array(0.001),
                "accuracy": np.array(accuracy), "pml_width": np.array(pml),
                "pml_freq": np.array(25.), "max_velocity": np.array(2500.),
                "boundary": np.array(boundary), "render_scale": np.array(0.25)}


def wavelet(nt, dt, f=25., delay=0.035):
    a = (np.pi*f*(np.arange(nt)*dt-delay))**2
    return ((1-2*a)*np.exp(-a)).astype(np.float32)


def write(name, case, result, summaries):
    for value in result.values():
        assert np.all(np.isfinite(value)), f"Non-finite oracle output in {name}"
    np.savez_compressed(DEST / f"{name}.npz", **case, **result)
    summaries[name] = {
        "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                       **({"value": v.item()} if v.ndim == 0 else {})}
                   for k, v in case.items()},
        "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                         "min": float(np.min(v)), "max": float(np.max(v))}
                    for k, v in result.items() if np.issubdtype(v.dtype, np.number)},
    }
    print(name, {k: (float(np.min(v)), float(np.max(v))) for k, v in result.items()
                 if k in ("expected_state", "expected_objective", "expected_gradient")}, flush=True)


def main():
    started = time.perf_counter()
    # Refuse silent upgrades: keep the reviewed pinned runtime.
    for line in (ROOT / "judge" / "requirements.lock").read_text().splitlines():
        if line and not line.startswith("#"):
            package, version = line.split("==")
            assert metadata.version(package) == version, (package, version)
    DEST.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(SEED)
    summaries = {}
    for order in (2, 4, 6, 8):
        c = base(25, 31, 160, accuracy=order)
        yy, xx = np.indices(c["velocity"].shape)
        c["velocity"] = (1600 + 300*(yy >= 12) + 120*np.sin(xx/5)
                         + rng.uniform(-30, 30, yy.shape)).astype(np.float32)
        c["source_amplitudes"][0] = wavelet(160, .001)
        c["source_amplitudes"][1] = -0.6*wavelet(160, .001, f=20., delay=.055)
        full = reference(c)
        assert np.max(np.abs(full["expected_state"][2:])) > 0, "PML not exercised"
        write(f"evolved_o{order}", c, full, summaries)
        # Make a separate continuation fixture with nonzero PML memory. Inputs
        # are quantized before recomputing the one-step float64 reference.
        one = dict(c)
        one["state"] = full["expected_state"].astype(np.float32)
        one["velocity"] = c["velocity"].copy()
        one["velocity"][6:11, 7:13] += 100  # A painted patch during propagation.
        one["source_amplitudes"] = np.array([[.7], [-.4]], np.float32)
        write(f"one_step_o{order}", one, reference(one), summaries)

    rest = base(17, 23, 5)
    rest_out = reference(rest)
    assert np.count_nonzero(rest_out["expected_state"]) == 0
    write("rest", rest, rest_out, summaries)

    impulse = base(17, 23, 1)
    impulse["source_amplitudes"][:, 0] = [1., -.5]
    impulse_out = reference(impulse)
    physical = impulse_out["expected_state"][0, 8:-8, 8:-8]
    expected = np.zeros_like(physical)
    for s, (y, x) in enumerate(impulse["source_locations"]):
        expected[y, x] = -float(impulse["velocity"][y, x])**2 * float(impulse["dt"])**2 * float(impulse["source_amplitudes"][s, 0])
    assert np.max(np.abs(physical-expected)) <= np.max(np.abs(expected))*TOL["ORACLE_CALIBRATION_REL"]
    assert np.count_nonzero(impulse_out["expected_receivers"]) == 0
    write("source_impulse", impulse, impulse_out, summaries)

    conv_errors = []
    for n in (16, 32, 64):
        c = base(n, n, n, pml=0, boundary="periodic")
        h, speed = 640./n, 1600.
        dt = .4*h/speed
        y, x = np.indices((n, n))
        mode = np.sin(2*np.pi*x/n)*np.sin(2*np.pi*y/n)
        omega = speed * np.sqrt(2) * 2*np.pi/640.
        c["velocity"].fill(speed)
        c["grid_spacing"] = np.array([h, h])
        c["dt"] = np.array(dt)
        c["max_velocity"] = np.array(speed)
        c["pml_freq"] = np.array(omega/(2*np.pi))  # Actual mode frequency; no PML.
        c["state"][0] = mode
        c["state"][1] = mode*np.cos(omega*dt)
        result = reference(c)
        result["analytic_final"] = mode*np.cos(omega*dt*n)
        err = np.linalg.norm(result["expected_state"][0]-result["analytic_final"])
        conv_errors.append(err/n)  # RMS, independent of number of cells.
        write(f"convergence_n{n}", c, result, summaries)
    orders = np.log2(np.array(conv_errors[:-1])/conv_errors[1:])
    assert np.all(np.abs(orders-TOL["CONVERGENCE_ORDER"]) <= TOL["CONVERGENCE_ORDER_SLACK"]), orders

    c = base(19, 23, 120, pml=8)
    y, x = np.indices(c["velocity"].shape)
    c["velocity"] = (1750+180*(y>9)+50*np.sin(x/4)).astype(np.float32)
    c["source_amplitudes"][0] = wavelet(120, .001)
    c["source_amplitudes"][1] = -.4*wavelet(120, .001, delay=.05)
    truth = dict(c)
    anomaly = np.exp(-((y-10)**2/16+(x-12)**2/25))
    truth["velocity"] = (c["velocity"]+160*anomaly).astype(np.float32)
    c["observed"] = reference(truth)["expected_receivers"].astype(np.float32)
    result = reference(c, grad=True)
    grad = result["expected_gradient"]
    directions = np.stack((np.ones_like(grad), anomaly, grad/np.max(np.abs(grad)))).astype(np.float32)
    fd = []
    eps = TOL["FD_STEP_MPS"]
    for direction in directions:
        plus, minus = dict(c), dict(c)
        plus["velocity"] = (c["velocity"]+eps*direction).astype(np.float32)
        minus["velocity"] = (c["velocity"]-eps*direction).astype(np.float32)
        # Use actual quantized perturbation direction in the adjoint comparison.
        actual_direction = (plus["velocity"].astype(np.float64)-minus["velocity"])/(2*eps)
        jp = reference(plus, grad=True)["expected_objective"]
        jm = reference(minus, grad=True)["expected_objective"]
        estimate = float((jp-jm)/(2*eps))
        exact = float(np.sum(grad*actual_direction))
        assert abs(estimate-exact)/abs(exact) <= TOL["GRAD_FD_REL"], (estimate, exact)
        fd.append(estimate)
    # Oracle chooses once; implementations cannot tune a line search to the test.
    alpha = 20./np.max(np.abs(grad))
    trial = dict(c)
    trial["velocity"] = (c["velocity"]-alpha*grad).astype(np.float32)
    ratio = float(reference(trial, grad=True)["expected_objective"]/result["expected_objective"])
    assert ratio <= TOL["DESCENT_RATIO_MAX"], ratio
    result.update(directions=directions, expected_fd=np.array(fd),
                  descent_step=np.array(alpha), expected_descent_ratio=np.array(ratio))
    write("fwi", c, result, summaries)

    n = TOL["PERFORMANCE_GRID_SIZE"]
    c = base(n, n, TOL["PERFORMANCE_STEPS"], pml=8, accuracy=2)
    # A broad initialized packet forces visible dynamics away from just one source.
    # Only two state planes are populated; compressed fixtures stay small.
    y, x = np.indices((n, n))
    packet = np.exp(-((y-500.)**2+(x-500.)**2)/(2*24.**2))*np.cos((x-500.)/8.)
    packet[(y-500.)**2+(x-500.)**2 > 120.**2] = 0
    c["state"][0, 8:8+n, 8:8+n] = packet.astype(np.float32)
    c["state"][1, 8:8+n, 8:8+n] = packet.astype(np.float32)
    c["velocity"] = (1700.+200.*(y>=500)+100.*(x>=600)).astype(np.float32)
    c["source_amplitudes"][0] = wavelet(TOL["PERFORMANCE_STEPS"], .001)
    c["source_amplitudes"][1] = -.7*wavelet(TOL["PERFORMANCE_STEPS"], .001, delay=.07)
    write("performance", c, reference(c), summaries)

    # Audit the locked thresholds against a float32 run of the independent
    # reference itself. This is judge validation, not an implementation pass.
    audits, scheme_audits = {}, {}
    render_audits, render_mutations, independent32_audits = {}, {}, {}
    float32_conv = []
    for name in summaries:
        with np.load(DEST / f"{name}.npz", allow_pickle=False) as saved:
            fixture = {k: saved[k] for k in saved.files}
        if name != "performance":
            # Small fixtures cover every order, PML corners, painted/resumed
            # state and periodic closure. FWI's forward traces are audited too.
            documented = documented_reference(fixture)
            scheme_errors = {}
            for key, actual in documented.items():
                if key not in fixture:
                    continue  # FWI saves receiver data but no final/render data.
                expected = fixture[key]
                for ch in range(6) if key == "expected_state" else [None]:
                    a = actual if ch is None else actual[ch]
                    b = expected if ch is None else expected[ch]
                    scale = float(np.max(np.abs(b)))
                    difference = float(np.max(np.abs(a-b)))
                    error = difference/scale if scale else difference
                    assert error <= (TOL["ORACLE_SCHEME_REL"] if scale else TOL["EXACT_ABS"]), (name, key, ch, error)
                    scheme_errors[f"{key}:{ch}"] = error
            scheme_audits[name] = scheme_errors
        result32 = reference(fixture, grad=name == "fwi", dtype=torch.float32)
        if name != "fwi":
            independent32 = documented_reference(fixture, dtype=np.float32)
            assert all(a.dtype == np.float32 for a in independent32.values())
            render_audits[name] = {
                "deepwave_float32": audit_render(fixture, result32),
                "numpy_recurrence_float32": audit_render(fixture, independent32),
            }
            # Check that the independent render audit is based on a physically
            # correct float32 recurrence, not only plausible normalized colors.
            independent_errors = {}
            for key in ("expected_state", "expected_receivers"):
                for ch in range(6) if key == "expected_state" else [None]:
                    a = independent32[key] if ch is None else independent32[key][ch]
                    b = fixture[key] if ch is None else fixture[key][ch]
                    peak = float(np.max(np.abs(b)))
                    diff = float(np.max(np.abs(a.astype(np.float64)-b)))
                    err = diff/peak if peak else diff
                    assert err <= (TOL["PARITY_REL"] if peak else TOL["EXACT_ABS"]), (name, key, ch, err)
                    independent_errors[f"{key}:{ch}"] = err
            independent32_audits[name] = independent_errors
            render_mutations[name] = audit_render_mutations(fixture)
        errors32 = {}
        for key, actual in result32.items():
            if key in ("expected_rgba", "expected_texture_samples"):
                # Color errors use the pressure-propagated absolute budget,
                # audited above; alpha=1 is not a physical normalization.
                continue
            expected64 = fixture[key]
            pieces = range(6) if key == "expected_state" else [None]
            for piece in pieces:
                a = actual if piece is None else actual[piece]
                b = expected64 if piece is None else expected64[piece]
                scale = float(np.max(np.abs(b)))
                diff = float(np.max(np.abs(a.astype(np.float64)-b)))
                err = diff/scale if scale else diff
                threshold = TOL["PARITY_REL"] if scale else TOL["EXACT_ABS"]
                assert err <= threshold, (name, key, piece, err)
                errors32[f"{key}:{piece}"] = err
        if name.startswith("convergence"):
            float32_conv.append(float(np.sqrt(np.mean(
                (result32["expected_state"][0].astype(np.float64)-fixture["analytic_final"])**2))))
        if name == "fwi":
            for i, direction in enumerate(fixture["directions"]):
                plus, minus = dict(fixture), dict(fixture)
                plus["velocity"] = (fixture["velocity"]+eps*direction).astype(np.float32)
                minus["velocity"] = (fixture["velocity"]-eps*direction).astype(np.float32)
                jp = reference(plus, grad=True, dtype=torch.float32)["expected_objective"]
                jm = reference(minus, grad=True, dtype=torch.float32)["expected_objective"]
                fd32 = (float(jp)-float(jm))/(2*eps)
                actual_dir = (plus["velocity"].astype(np.float64)-minus["velocity"])/(2*eps)
                adjoint32 = float(np.sum(result32["expected_gradient"].astype(np.float64)*actual_dir))
                err = abs(fd32-adjoint32)/abs(fd32)
                assert err <= TOL["GRAD_FD_REL"], (i, fd32, adjoint32, err)
                ref_err = abs(fd32-fixture["expected_fd"][i])/abs(fixture["expected_fd"][i])
                assert ref_err <= TOL["GRAD_FD_REL"], (i, ref_err)
                errors32[f"centered_fd_direction_{i}"] = err
        audits[name] = errors32
    for key in ("expected_rgba", "expected_texture_samples"):
        for mutation in ("swapped_red_blue", "doubled_render_scale", "omitted_clip"):
            assert any(a[key][mutation] for a in render_mutations.values()), (key, mutation)
    float32_orders = np.log2(np.array(float32_conv[:-1])/float32_conv[1:])
    assert np.all(np.abs(float32_orders-TOL["CONVERGENCE_ORDER"]) <= TOL["CONVERGENCE_ORDER_SLACK"])
    versions = {p: metadata.version(p) for p in ("numpy", "torch", "deepwave", "pytest", "ruff")}
    ref_source = Path(deepwave.__file__).parent / "scalar.py"
    ref_binary = ref_source.parent / "libdeepwave_C.dylib"
    source_hashes = {name: hashlib.sha256((ref_source.parent / name).read_bytes()).hexdigest()
                     for name in ("scalar.py", "_common.py", "_regular_grid.py",
                                  "scalar.c", "scalar_kernels.h", "regular_grid.h",
                                  "libdeepwave_C.dylib")}
    meta = {"seed": SEED, "python": platform.python_version(), "versions": versions,
                "reference": "Deepwave scalar CPU float64; analytic periodic homogeneous standing wave",
                "deepwave_scalar_sha256": hashlib.sha256(ref_source.read_bytes()).hexdigest(),
                "deepwave_binary_sha256": hashlib.sha256(ref_binary.read_bytes()).hexdigest(),
                "deepwave_source_sha256": source_hashes,
                "scheme_sha256": hashlib.sha256((ROOT / "tests" / "SCHEME.md").read_bytes()).hexdigest(),
                "documented_scheme_float64_audit": scheme_audits,
                "render_target_format": "MTLPixelFormatRGBA32Float; linear float32; no intermediate quantization",
                "tolerances": {k: {"value": TOL[k], "reason": v} for k, v in REASONS.items()},
                "inputs": summaries, "convergence_rms": conv_errors, "convergence_orders": orders.tolist(),
                "fwi_descent_ratio": ratio, "generation_seconds": time.perf_counter()-started,
                "float32_reference_audit": audits,
                "independent_numpy_float32_physics_audit": independent32_audits,
                "render_float32_audit": render_audits,
                "render_mutation_rejections": render_mutations,
                "float32_reference_convergence_orders": float32_orders.tolist(),
                "conventions": "tests/api.py; no CFL substeps, uncompressed stride-1 gradient, SI geometry",
                "tag": "T", "paths": ["metal"], "upstream_contracts_available": True,
                "upstream_conventions": "13: uniform spacing generalized to dy,dx; 16: mean objective multiplied by nr*nt"}
    (DEST / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    total = sum(p.stat().st_size for p in DEST.iterdir() if p.is_file())
    assert total < 20_000_000, f"Goldens too large: {total} bytes"
    print(f"Convergence RMS={conv_errors}; orders={orders.tolist()}", flush=True)
    print(f"Float32 audit passed; convergence orders={float32_orders.tolist()}", flush=True)
    worst_fraction = max(output["budget_fraction"] for case in render_audits.values()
                         for ref in case.values() for output in ref.values())
    print(f"Both float32 render audits passed; worst budget fraction={worst_fraction:.8g}; minimum margin={1/worst_fraction:.3f}x; all three render mutations rejected", flush=True)
    print(f"Written scheme float64 audit passed; maximum relative error={max(max(v.values()) for v in scheme_audits.values()):.8g}", flush=True)
    print(f"FWI descent ratio={ratio:.8g}; goldens={total} bytes; elapsed={time.perf_counter()-started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
