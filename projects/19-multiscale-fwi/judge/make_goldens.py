"""Independent CPU/float64 Deepwave oracle; run from the project directory.

No implementation is imported. This is full-waveform (not travel-time)
inversion on a small, two-region velocity basis in a 2D acoustic medium.
"""
import hashlib
import json
import platform
import runpy
import time
from importlib.metadata import version
from pathlib import Path

import deepwave
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "judge" / "golden"
CRITERIA = runpy.run_path(str(ROOT / "tests" / "tolerances.py"))["CRITERIA"]
TOL = {k: v[0] for k, v in CRITERIA.items()}
SEED = 190028
FD_STEP = 1.0  # m/s; recorded in each fixture, not an error tolerance.
PROTOCOL = {
    "steps_per_stage": 35, "max_step": 80.0, "lower": 1500.0,
    "upper": 2900.0, "backtracks": 16, "armijo": 1e-4,
    "objective_stop": TOL["optimizer_stop"], "gradient_floor": 1e-30,
}
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(SEED)


def tensor(x, dtype=torch.float64):
    return torch.as_tensor(x, dtype=dtype)


def forward(parameters, g):
    velocity = torch.einsum("p,pzx->zx", parameters, tensor(g["basis"], parameters.dtype))
    return deepwave.scalar(
        velocity, float(g["dx"]), float(g["dt"]),
        source_amplitudes=tensor(g["source"], parameters.dtype),
        source_locations=tensor(g["sources"], torch.int64),
        receiver_locations=tensor(g["receivers"], torch.int64),
        accuracy=4, pml_width=12, pml_freq=12.0, max_vel=3000.0,
        model_gradient_sampling_interval=1, freq_taper_frac=0.0,
        time_pad_frac=0.0, time_taper=False,
    )[-1]


def evaluate(parameters, g, cutoff, gradient=True, dtype=torch.float64):
    p = torch.tensor(parameters, dtype=dtype, requires_grad=gradient)
    predicted = forward(p, g)
    observed = tensor(g["observed"], dtype)
    weights = 1 / torch.sqrt(1 + (torch.fft.rfftfreq(
        predicted.shape[-1], float(g["dt"]), dtype=dtype
    ) / cutoff) ** 8)
    residual = torch.fft.irfft(torch.fft.rfft(predicted - observed) * weights,
                               n=predicted.shape[-1])
    filtered = torch.fft.irfft(torch.fft.rfft(observed) * weights,
                               n=predicted.shape[-1])
    objective = 0.5 * residual.square().sum() / filtered.square().sum()
    grad = None
    if gradient:
        objective.backward()
        grad = p.grad.numpy().copy()
    return objective.item(), grad, predicted.detach().numpy().copy()


def invert(initial, g, cutoffs, dtype=torch.float64):
    p = initial.astype(np.float64 if dtype == torch.float64 else np.float32)
    history, objectives = [p.copy()], []
    for cutoff in cutoffs:
        stalled = False
        for _ in range(PROTOCOL["steps_per_stage"]):
            if not stalled:
                value, grad, _ = evaluate(p, g, cutoff, dtype=dtype)
                stalled = value <= PROTOCOL["objective_stop"]
                if not stalled:
                    direction = -PROTOCOL["max_step"] * grad / max(
                        np.max(np.abs(grad)), PROTOCOL["gradient_floor"])
                    accepted = False
                    for k in range(PROTOCOL["backtracks"]):
                        candidate = np.clip(p + 2.0**(-k) * direction,
                                            PROTOCOL["lower"], PROTOCOL["upper"])
                        trial, _, _ = evaluate(candidate, g, cutoff, False, dtype)
                        if trial <= value + PROTOCOL["armijo"] * np.dot(grad, candidate - p):
                            p = candidate
                            accepted = True
                            break
                    stalled = not accepted
            objectives.append(value)
            history.append(p.copy())
        print(f"  cutoff={cutoff:g} parameters={p} objective={evaluate(p, g, cutoff, False, dtype)[0]:.8g}", flush=True)
    return {"parameters": p, "trajectory": np.array(history),
            "objective_history": np.array(objectives)}


def relative(a, b):
    return float(np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b)))


def model_error(p, g):
    return float(np.linalg.norm(np.einsum("p,pzx->zx", p - g["truth"], g["basis"])))


def audit_single_resolution(g, single):
    """Measure the shallow constrained minimum; do not change the optimizer.

    A strict relative velocity comparison cannot distinguish optimizer error
    from float32 objective noise here. Save the exact scope of an absolute
    velocity allowance and audit its objective/gradient consequences.
    """
    endpoint, states = single["parameters"], single["trajectory"]
    free = (endpoint > PROTOCOL["lower"]) & (endpoint < PROTOCOL["upper"])
    assert np.count_nonzero(free) == 1  # Specific to these two-region fixtures.
    near = np.max(np.abs(states - endpoint), axis=1) <= TOL["single_flat_velocity"]
    bound = np.all(states[:, ~free] == endpoint[~free], axis=1)
    g["single_flat_mask"] = (near & bound)[:, None] & free
    assert not g["single_flat_mask"][0].any()
    assert g["single_flat_mask"][-1].any()

    offsets = TOL["single_flat_velocity"] * np.array([-1., -.5, -.3, -.2, 0., .2, .3, .5, 1.])
    parameters = endpoint + offsets[:, None] * free
    values, gradients, values32, gradients32 = [], [], [], []
    for p in parameters:
        value, grad, _ = evaluate(p, g, g["cutoffs"][-1])
        value32, grad32, _ = evaluate(p, g, g["cutoffs"][-1], dtype=torch.float32)
        values.append(value)
        gradients.append(grad)
        values32.append(value32)
        gradients32.append(grad32)
    values, gradients = np.array(values), np.array(gradients)
    values32, gradients32 = np.array(values32), np.array(gradients32)
    g.update(single_resolution_offsets=offsets,
             single_resolution_parameters=parameters,
             single_resolution_objectives=values,
             single_resolution_gradients=gradients,
             single_resolution_objectives32=values32,
             single_resolution_gradients32=gradients32)
    zero = int(np.flatnonzero(offsets == 0)[0])
    curvature = float(((gradients[-1] - gradients[0]) / (offsets[-1] - offsets[0]))[free][0])
    assert curvature > 0
    projected_max = []
    for grad in [gradients, gradients32]:
        projected = parameters - np.clip(parameters - grad, PROTOCOL["lower"], PROTOCOL["upper"])
        projected_max.append(float(np.max(np.abs(projected))))
    assert max(projected_max) <= TOL["baseline_stationarity"]
    rises = (values - values[zero]) / values[zero]
    assert np.max(np.abs(rises)) <= TOL["parity"]
    return {
        "free_coefficient": int(np.flatnonzero(free)[0]),
        "first_relaxed_state": int(np.flatnonzero(g["single_flat_mask"].any(axis=1))[0]),
        "offsets_m_per_s": offsets.tolist(),
        "float64_objective_relative_rise": rises.tolist(),
        "float32_objective_relative_noise": ((values32 - values) / values).tolist(),
        "curvature_per_velocity_squared": curvature,
        "max_projected_gradient_float64": projected_max[0],
        "max_projected_gradient_float32": projected_max[1],
        "noise_equivalent_velocity_m_per_s": float(np.sqrt(2 * np.max(np.abs(values32 - values)) / curvature)),
    }


def build_case(index, rng):
    nz, nx, nt, dx, dt = 25, 61, 500, 20.0, 0.002
    basis = np.zeros((2, nz, nx), np.float32)
    basis[0, :, :31] = 1
    basis[1, :, 31:] = 1
    peak = [12.0, 12.5][index]
    t = np.arange(nt, dtype=np.float64) * dt
    a = (np.pi * peak * (t - 0.12)) ** 2
    source = np.tile((1 - 2 * a) * np.exp(-a), (2, 1, 1)).astype(np.float32)
    g = {
        "basis": basis, "source": source,
        "sources": np.array([[[12, 6]], [[12, 54]]], np.int64),
        "receivers": np.array([[[12, x] for x in [16, 24, 36, 44]]] * 2, np.int64),
        "truth": np.array([[2200., 2600.], [2250., 2650.]][index]),
        "initial": np.array([1700., 1800.]),
        "dx": np.array(dx), "dt": np.array(dt), "peak_frequency": np.array(peak),
        "cutoffs": np.array([3., 5., 9., 18.]),
        "probe_cutoffs": np.array([3., 9., 18.]), "fd_step": np.array(FD_STEP),
        "steps_per_stage": np.array(PROTOCOL["steps_per_stage"], np.int64),
    }
    # All floating API inputs are quantized to float32 *before* reference work.
    g["observed_reference"] = forward(tensor(g["truth"]), g).numpy()
    g["observed"] = g["observed_reference"].astype(np.float32)
    g["probe"] = (g["truth"] - rng.uniform(120., 220., 2)).astype(np.float32)
    probes = np.stack([g["initial"], g["probe"], g["truth"]])
    values, gradients, predictions = [], [], []
    fd_checks, audit32 = [], []
    truth32_checks = []
    for pi, p in enumerate(probes):
        pv, pg = [], []
        for cutoff in g["probe_cutoffs"]:
            value, grad, predicted = evaluate(p, g, cutoff)
            pv.append(value)
            pg.append(grad)
            v32, g32, pred32 = evaluate(p, g, cutoff, dtype=torch.float32)
            assert relative(pred32, predicted) <= TOL["parity"]
            if pi == 2:
                print(f"case {index} truth float32 cutoff={cutoff:g}: J={v32:.9g}, gradient_max={np.max(np.abs(g32)):.9g}", flush=True)
                assert v32 <= TOL["zero_objective"]
                assert np.max(np.abs(g32)) <= TOL["zero_gradient"]
                truth32_checks.append({"cutoff": float(cutoff), "objective": v32,
                                       "gradient_max": float(np.max(np.abs(g32)))})
            else:
                assert relative(v32, value) <= TOL["parity"]
                assert relative(g32, grad) <= TOL["parity"]
        predictions.append(predicted)
        values.append(pv)
        gradients.append(pg)
    g["probe_parameters"] = probes
    g["objectives"] = np.array(values)
    g["gradients"] = np.array(gradients)
    g["predictions"] = np.array(predictions)
    for ci, cutoff in enumerate(g["probe_cutoffs"]):
        fd, fd32 = [], []
        for j in range(2):
            delta = np.eye(2)[j] * FD_STEP
            plus = evaluate(g["probe"] + delta, g, cutoff, False)[0]
            minus = evaluate(g["probe"] - delta, g, cutoff, False)[0]
            fd.append((plus - minus) / (2 * FD_STEP))
            plus32 = evaluate(g["probe"] + delta, g, cutoff, False, torch.float32)[0]
            minus32 = evaluate(g["probe"] - delta, g, cutoff, False, torch.float32)[0]
            fd32.append((plus32 - minus32) / (2 * FD_STEP))
        value32, grad32, pred32 = evaluate(g["probe"], g, cutoff, dtype=torch.float32)
        audit = {"cutoff": float(cutoff),
                 "objective_relative": relative(value32, g["objectives"][1, ci]),
                 "gradient_relative": relative(grad32, g["gradients"][1, ci]),
                 "waveform_relative": relative(pred32, g["predictions"][1]),
                 "fd_relative": relative(fd32, grad32),
                 "fd_component_relative": float(np.max(np.abs(np.asarray(fd32) - grad32) / np.abs(grad32)))}
        audit32.append(audit)
        assert relative(fd, g["gradients"][1, ci]) <= TOL["oracle_fd"]
        assert max(audit[k] for k in ["objective_relative", "gradient_relative", "waveform_relative"]) <= TOL["parity"]
        assert audit["fd_relative"] <= TOL["gradient_fd"]
        assert audit["fd_component_relative"] <= TOL["gradient_fd"]
        fd_checks.append(fd)
    g["finite_differences"] = np.array(fd_checks)
    print(f"case {index}: multiscale", flush=True)
    multi = invert(g["initial"], g, g["cutoffs"])
    print(f"case {index}: equal-budget single-scale", flush=True)
    single = invert(g["initial"], g, np.full_like(g["cutoffs"], g["cutoffs"][-1]))
    for label, result in [("multi", multi), ("single", single)]:
        g.update({f"{label}_{k}": v for k, v in result.items()})
    resolution_audit = audit_single_resolution(g, single)
    optimizer_audits = {}
    for label, schedule, reference in [("multi", g["cutoffs"], multi),
                                       ("single", np.full_like(g["cutoffs"], g["cutoffs"][-1]), single)]:
        print(f"case {index}: float32 CPU optimizer audit {label}", flush=True)
        audited = invert(g["initial"], g, schedule, torch.float32)
        optimizer_audits[label] = {}
        for key in reference:
            error = relative(audited[key], reference[key])
            optimizer_audits[label][key + "_relative"] = error
            absolute = np.max(np.abs(audited[key] - reference[key]))
            allowance = TOL["history_atol"] if key == "objective_history" else 0
            if label == "single" and key in ["parameters", "trajectory"]:
                mask = g["single_flat_mask"][-1] if key == "parameters" else g["single_flat_mask"]
                limits = np.where(mask, TOL["single_flat_velocity"],
                                  TOL["parity"] * np.max(np.abs(reference[key])))
                assert np.all(np.abs(audited[key] - reference[key]) <= limits)
                optimizer_audits[label][key + "_absolute_m_per_s"] = float(absolute)
            else:
                assert absolute <= TOL["parity"] * np.max(np.abs(reference[key])) + allowance
        if label == "single":
            p32 = audited["parameters"].astype(np.float64)
            _, grad32, _ = evaluate(p32, g, schedule[-1], dtype=torch.float32)
            projected32 = p32 - np.clip(p32 - grad32, PROTOCOL["lower"], PROTOCOL["upper"])
            optimizer_audits[label]["projected_gradient"] = float(np.max(np.abs(projected32)))
            assert optimizer_audits[label]["projected_gradient"] <= TOL["baseline_stationarity"]
    errors = {label: model_error(result["parameters"], g)
              for label, result in [("multi", multi), ("single", single)]}
    errors["initial"] = model_error(g["initial"], g)
    # Cell-centered discontinuity at x=30.5*dx; direct ray from x=6 to 44.
    travel_weights = dx * np.array([24.5, 13.5])
    g["travel_weights"] = travel_weights
    phase_cycles = float(np.dot(travel_weights, 1/g["initial"] - 1/g["truth"]) * peak)
    _, grad, _ = evaluate(single["parameters"], g, g["cutoffs"][-1])
    projected = single["parameters"] - np.clip(single["parameters"] - grad,
                                               PROTOCOL["lower"], PROTOCOL["upper"])
    stationarity = float(np.max(np.abs(projected)))
    assert errors["multi"] < TOL["improvement"] * errors["single"]
    assert errors["single"] >= TOL["baseline_error_fraction"] * errors["initial"]
    assert phase_cycles > TOL["half_cycle"]
    assert stationarity <= TOL["baseline_stationarity"]
    assert max(g["objectives"][2]) <= TOL["zero_objective"]
    assert np.max(np.abs(g["gradients"][2])) <= TOL["zero_gradient"]
    stats = {"model_grid_l2_errors_m_per_s": errors,
             "error_ratio": errors["multi"] / errors["single"],
             "initial_direct_arrival_error_cycles": phase_cycles,
             "single_projected_gradient": stationarity,
             "receiver_peak": float(np.max(np.abs(g["observed_reference"]))),
             "oracle_fd_relative": [relative(fd, gr) for fd, gr in zip(fd_checks, g["gradients"][1])],
             "float32_cpu_audit": audit32}
    stats["float32_optimizer_audit"] = optimizer_audits
    stats["float32_truth_audit"] = truth32_checks
    stats["single_resolution_audit"] = resolution_audit
    print(json.dumps(stats, indent=2), flush=True)
    np.savez_compressed(DEST / f"case_{index}.npz", **g)
    return stats


def main():
    started = time.monotonic()
    DEST.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    stats = {f"case_{i}": build_case(i, rng) for i in range(2)}
    package = Path(deepwave.__file__).parent
    hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
              for name in ["scalar.py", "scalar.c", "scalar_kernels.h", "libdeepwave_C.dylib"]}
    manifests = {}
    for name in stats:
        filename = DEST / f"{name}.npz"
        arrays = {}
        with np.load(filename, allow_pickle=False) as saved:
            for key in ["basis", "source", "sources", "receivers", "observed",
                        "truth", "initial", "probe_parameters", "dx", "dt",
                        "cutoffs", "probe_cutoffs", "fd_step", "steps_per_stage"]:
                a = saved[key]
                arrays[key] = {"shape": list(a.shape), "stored_dtype": str(a.dtype)}
                if a.size <= 20:
                    arrays[key]["values"] = a.tolist()
        manifests[name] = {"file": filename.name, "arrays": arrays,
                           "sha256": hashlib.sha256(filename.read_bytes()).hexdigest()}
    meta = {
        "project": "19-multiscale-fwi", "tag": "M", "paths": ["mlx"],
        "seed": SEED, "python": platform.python_version(),
        "versions": {name: version(name) for name in ["numpy", "torch", "deepwave", "pytest", "mlx"]},
        "reference": "Deepwave scalar, CPU float64, PyTorch autograd; two-dimensional scalar acoustic full-waveform inversion",
        "deepwave_source_sha256": hashes,
        "inputs": manifests,
        "input_precision": "Every floating API array is cast to float32 before use by implementations; source/basis/data/probe quantized to float32 before float64 reference work. Exact integer-valued truth/initial arrays are stored float64; coordinate indices int64.",
        "grid": {"shape_zx": [25, 61], "dx_m": 20., "dt_s": .002, "nt": 500,
                 "accuracy": 4, "pml_width": 12, "pml_freq_hz": 12., "max_vel": 3000.,
                 "time_order": 2, "time_pad_frac": 0., "freq_taper_frac": 0., "time_taper": False,
                 "survey_pad": None, "model_gradient_sampling_interval": 1},
        "filter": "Periodic, zero-phase real FFT on nt samples; H(f)=[1+(f/fc)^8]^(-1/2), irfft n=nt; no padding/taper; same operator on predicted and observed",
        "objective": "0.5 * sum((H(predicted-observed))^2) / sum((H(observed))^2), over all shots, receivers and times; no regularization",
        "parity_metric": "max(abs(actual-reference))/max(abs(reference)); history additionally permits history_atol. Only the saved single_flat_mask entries use single_flat_velocity absolute m/s INSTEAD OF relative parity. The mask selects free coefficients after the reference is within 1 m/s of its endpoint and its bound coefficients have reached their bounds. All other parameters/states retain 1e-4 parity. At zero-residual truth use absolute zero_objective/zero_gradient criteria.",
        "model_error_metric": "L2 norm of sum_p((parameters[p]-truth[p])*basis[p,z,x]) over all physical grid cells; same units m/s; no PML cells in the score",
        "optimizer": PROTOCOL,
        "gradient_check": {"step_m_per_s": FD_STEP, "method": "centered component differences at saved probe, independently in float64 and audited float32"},
        "tolerances": {k: {"value": value, "reason": reason} for k, (value, reason) in CRITERIA.items()},
        "cases": stats,
        "limitations": ["Two velocity parameters on a genuine 2D wave grid; no claim of resolved pixel-wise FWI or realistic survey coverage.",
                        "Single-scale can settle on the lower velocity bound; projected stationarity and >half-cycle initial phase error are checked.",
                        "The late single-scale free coordinate is unresolved at 1e-4 relative velocity parity: a saved float64/float32 objective sweep documents a 1 m/s absolute allowance there and a matching 5e-6 projected-stationarity bound. No optimizer or improvement threshold changed.",
                        "Same-grid solver parity; propagation convergence/PML validation belongs to dependency 13, not duplicated here.",
                        "Installed Deepwave is version 0.1.0; source and native-library hashes record the exact local build."],
    }
    meta["golden_bytes"] = sum(p.stat().st_size for p in DEST.glob("case_*.npz"))
    assert meta["golden_bytes"] < 20_000_000
    (DEST / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {meta['golden_bytes']} bytes in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
