"""Float64 Deepwave plain-FWI / neural-prior oracle. No implementation imports."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import runpy
import time
from pathlib import Path

import deepwave
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
TOL = runpy.run_path(str(ROOT.parent / "tests" / "tolerances.py"))
SEEDS = (20261020, 20261021)
STEPS = 160
LEARNING_RATE = 0.02
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def decode(p, latent, initial, scale, coarse_shape, hidden_size, mode):
    if mode == "grid":
        q = p.reshape(initial.shape)
    else:
        h, l = hidden_size, latent.numel()
        c = int(np.prod(coarse_shape))
        w1 = p[:h*l].reshape(h, l)
        b1 = p[h*l:h*l+h]
        w2 = p[h*l+h:h*l+h+c*h].reshape(c, h)
        b2 = p[-c:]
        hidden = torch.tanh(w1 @ latent / np.sqrt(l) + b1)
        coarse = (w2 @ hidden / np.sqrt(h) + b2).reshape(1, 1, *coarse_shape)
        q = F.interpolate(coarse, size=initial.shape, mode="bilinear", align_corners=True)[0, 0]
    return initial + scale * torch.tanh(q)


def predict(v, g):
    output = deepwave.scalar(
        v, float(g["spacing"]), float(g["dt"]),
        source_amplitudes=g["source_amplitudes"],
        source_locations=g["source_locations"],
        receiver_locations=g["receiver_locations"],
        accuracy=4, pml_width=12, pml_freq=18.0, max_vel=2600.0,
        model_gradient_sampling_interval=1,
        freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False,
    )
    assert output.info.step_ratio == 1, "Locked acoustic operator forbids internal resampling"
    return output[-1]


def evaluate(p, g, mode):
    v = decode(p, g["latent"], g["initial_velocity"], 500.0, (5, 7), 8, mode)
    pred = predict(v, g)
    loss = 0.5 * torch.mean(((pred - g["observed"]) / g["data_scale"]) ** 2)
    return loss, v, pred


def invert(p0, g, mode, steps=STEPS, lr=LEARNING_RATE):
    p = p0.clone().requires_grad_()
    m, s = torch.zeros_like(p), torch.zeros_like(p)
    history = []
    for t in range(1, steps + 1):
        loss, _, _ = evaluate(p, g, mode)
        grad, = torch.autograd.grad(loss, p)
        history.append(float(loss.detach()))
        with torch.no_grad():
            m.mul_(0.9).add_(grad, alpha=0.1)
            s.mul_(0.999).addcmul_(grad, grad, value=0.001)
            p -= lr * (m / (1 - 0.9**t)) / (torch.sqrt(s / (1 - 0.999**t)) + 1e-8)
    loss, v, pred = evaluate(p, g, mode)
    history.append(float(loss.detach()))
    return p.detach(), v.detach(), pred.detach(), np.array(history)


def inputs(seed, dtype=torch.float64):
    rng = np.random.default_rng(seed)
    nz, nx, nt = 24, 32, 280
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    initial = np.full((nz, nx), 2000.0)
    truth = initial + 360*np.exp(-((z-12)/5)**2-((x-12)/7)**2) - 240*np.exp(-((z-15)/4)**2-((x-24)/5)**2)
    source_locations = np.array([[[2, 6]], [[2, 16]], [[2, 25]]])
    receivers = np.stack([np.full(15, 21), np.arange(1, 31, 2)], axis=-1)
    receiver_locations = np.repeat(receivers[None], 3, axis=0)
    t = np.arange(nt)*0.001
    a = (np.pi*18*(t-0.06))**2
    wavelet = (1-2*a)*np.exp(-a)
    source_amplitudes = np.repeat(wavelet[None, None], 3, axis=0)
    latent = rng.normal(size=4)
    h, l, c = 8, 4, 35
    p0 = np.r_[rng.normal(size=h*l), np.zeros(h+c*h+c)]
    arrays = {"initial_velocity": initial, "truth": truth, "latent": latent,
              "source_locations": source_locations, "receiver_locations": receiver_locations,
              "source_amplitudes": source_amplitudes}
    # Quantize public inputs once, then evaluate those exact inputs in float64.
    g = {k: torch.tensor(v if v.dtype.kind in "iu" else v.astype(np.float32),
                         dtype=torch.int64 if v.dtype.kind in "iu" else dtype)
         for k, v in arrays.items()}
    g.update(spacing=10.0, dt=0.001)
    clean = predict(g["truth"], g).detach()
    data_scale = torch.sqrt(torch.mean(clean**2)).float().to(dtype)
    noise = torch.tensor(rng.normal(size=clean.shape), dtype=dtype)
    observed = (clean + 0.5*data_scale*noise).float().to(dtype)
    g.update(observed=observed, data_scale=data_scale)
    return g, torch.tensor(p0.astype(np.float32), dtype=dtype), clean


def as_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def relative(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a-b)) / np.max(np.abs(b)))


def model_error(v, truth):
    return float(torch.linalg.vector_norm(v-truth) / torch.linalg.vector_norm(truth))


def calibrate_solution(single, double, initial):
    errors = {name: relative(as_numpy(f), as_numpy(d))
              for name, f, d in zip(("parameters", "velocity", "predicted", "loss_history"), single, double)}
    # Tests measure the perturbation, avoiding a generous normalization by v0.
    errors["velocity_perturbation"] = relative(
        as_numpy(single[1])-as_numpy(initial).astype(np.float32),
        as_numpy(double[1])-as_numpy(initial))
    assert max(errors.values()) <= TOL["PARITY_RTOL"], errors
    return errors


def probe(p, g, mode, rng):
    p = p + torch.tensor(rng.normal(scale=0.2 if mode == "neural" else 0.1,
                                  size=p.shape), dtype=p.dtype)
    p = p.float().double().requires_grad_()
    loss, velocity, predicted = evaluate(p, g, mode)
    grad, = torch.autograd.grad(loss, p)
    if mode == "grid":
        groups = [np.arange(0, p.numel(), 2), np.arange(1, p.numel(), 2)]
    else:
        groups = [np.arange(0, 32), np.arange(32, 40),
                  np.arange(40, 320), np.arange(320, 355)]
    directions, fd, fd32 = [], [], []
    gf = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
          for k, v in g.items()}
    for indices in groups:
        # Align within each disjoint parameter block: no cancellation or missing layer.
        d = torch.zeros_like(p)
        d[indices] = grad[indices]
        d = (d / torch.linalg.vector_norm(d)).float().double()
        directions.append(as_numpy(d))
        h = TOL["FD_STEP"]
        lp = evaluate(p.detach()+h*d, g, mode)[0]
        lm = evaluate(p.detach()-h*d, g, mode)[0]
        fd.append(float(((lp-lm)/(2*h)).detach()))
        pf, df = p.detach().float(), d.float()
        lp32 = evaluate(pf+h*df, gf, mode)[0]
        lm32 = evaluate(pf-h*df, gf, mode)[0]
        fd32.append(float(((lp32-lm32)/(2*h)).detach()))
    directional = np.asarray(directions) @ as_numpy(grad)
    assert np.min(np.abs(directional)) > TOL["MIN_DIRECTIONAL_DERIVATIVE"]
    fd_errors = np.abs((np.asarray(fd)-directional)/directional)
    fd32_errors = np.abs((np.asarray(fd32)-directional)/directional)
    assert np.max(fd_errors) <= TOL["ORACLE_FD_RTOL"], fd_errors
    assert np.max(fd32_errors) <= TOL["FD_RTOL"], fd32_errors
    pf = p.detach().float().requires_grad_()
    lf, vf, predf = evaluate(pf, gf, mode)
    gradf, = torch.autograd.grad(lf, pf)
    calibration = {
        "float64_fd_relative_errors": fd_errors.tolist(),
        "float32_fd_relative_errors": fd32_errors.tolist(),
        "float32_gradient_relative_error": relative(as_numpy(gradf), as_numpy(grad)),
        "float32_prediction_relative_error": relative(as_numpy(predf), as_numpy(predicted)),
    }
    calibration["float32_velocity_perturbation_relative_error"] = relative(
        as_numpy(vf)-as_numpy(g["initial_velocity"]).astype(np.float32),
        as_numpy(velocity)-as_numpy(g["initial_velocity"]))
    assert calibration["float32_gradient_relative_error"] <= TOL["PARITY_RTOL"]
    assert calibration["float32_prediction_relative_error"] <= TOL["PARITY_RTOL"]
    assert calibration["float32_velocity_perturbation_relative_error"] <= TOL["PARITY_RTOL"]
    return {"parameters": as_numpy(p).astype(np.float32), "objective": as_numpy(loss),
            "velocity": as_numpy(velocity), "predicted": as_numpy(predicted),
            "gradient": as_numpy(grad), "directions": np.asarray(directions, dtype=np.float32),
            "directional_fd": np.asarray(fd)}, calibration


def main():
    start = time.monotonic()
    dest = ROOT / "golden"
    dest.mkdir(exist_ok=True)
    metrics, calibrations, files = {}, {}, {}
    for case, seed in zip(("noise_a", "noise_b"), SEEDS):
        g, p0, clean = inputs(seed)
        rng = np.random.default_rng(seed + 100)
        saved = {k: as_numpy(v).astype(np.float32) if isinstance(v, torch.Tensor) and v.is_floating_point()
                 else as_numpy(v) for k, v in g.items() if k != "truth"}
        saved.update(truth=as_numpy(g["truth"]), clean=as_numpy(clean),
                     neural_initial_parameters=as_numpy(p0).astype(np.float32),
                     grid_initial_parameters=np.zeros(g["truth"].numel(), dtype=np.float32),
                     coarse_shape=np.array([5, 7], dtype=np.int64), hidden_size=np.array(8),
                     velocity_scale=np.array(500.0), steps=np.array(STEPS),
                     learning_rate=np.array(LEARNING_RATE))
        initial_loss, _, initial_pred = evaluate(p0, g, "neural")
        saved.update(initial_objective=as_numpy(initial_loss), initial_prediction=as_numpy(initial_pred))
        case_metrics = {
            "initial_model_error": model_error(g["initial_velocity"], g["truth"]),
            "noise_rms_over_clean_rms": float(torch.sqrt(torch.mean((g["observed"]-clean)**2)))/float(g["data_scale"]),
            "clean_max_abs": float(clean.abs().max()), "data_scale": float(g["data_scale"]),
            "truth_min": float(g["truth"].min()), "truth_max": float(g["truth"].max()),
        }
        case_calibrations = {}
        for mode, p in (("grid", torch.zeros(g["truth"].numel(), dtype=torch.float64)), ("neural", p0)):
            probed, probe_calibration = probe(p, g, mode, rng)
            saved.update({f"{mode}_probe_{k}": v for k, v in probed.items()})
            one = invert(p, g, mode, steps=1)
            for label, value in zip(("parameters", "velocity", "predicted", "loss_history"), one):
                saved[f"{mode}_one_{label}"] = as_numpy(value)
            result = invert(p, g, mode)
            for label, value in zip(("parameters", "velocity", "predicted", "loss_history"), result):
                saved[f"{mode}_final_{label}"] = as_numpy(value)
            err = model_error(result[1], g["truth"])
            case_metrics[mode] = {"model_error": err, "initial_loss": float(result[-1][0]),
                                  "final_loss": float(result[-1][-1]),
                                  "velocity_min": float(result[1].min()), "velocity_max": float(result[1].max())}
            gf = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                  for k, v in g.items()}
            onef = invert(p.float(), gf, mode, steps=1)
            one_calibration = calibrate_solution(onef, one, g["initial_velocity"])
            resultf = invert(p.float(), gf, mode)
            end_calibration = calibrate_solution(resultf, result, g["initial_velocity"])
            case_calibrations[mode] = {"probe": probe_calibration,
                                       "one_step_float32_relative_errors": one_calibration,
                                       "inversion_float32_relative_errors": end_calibration}
        ratio = case_metrics["neural"]["model_error"] / case_metrics["grid"]["model_error"]
        case_metrics["neural_over_plain_error"] = ratio
        assert ratio <= TOL["MAX_ERROR_RATIO"], case_metrics
        assert case_metrics["neural"]["model_error"] < TOL["MAX_INITIAL_ERROR_RATIO"]*case_metrics["initial_model_error"]
        assert torch.isfinite(clean).all() and float(clean.abs().max()) > 0
        np.savez_compressed(dest / f"{case}.npz", **saved)
        metrics[case], calibrations[case] = case_metrics, case_calibrations
        files[f"{case}.npz"] = {"bytes": (dest / f"{case}.npz").stat().st_size,
                                "sha256": hashlib.sha256((dest / f"{case}.npz").read_bytes()).hexdigest()}
        print(case, json.dumps(case_metrics, sort_keys=True), flush=True)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "deepwave", "pytest", "mlx")}
    meta = {
        "project": "20-nnfwi", "tag": "M", "seeds": dict(zip(("noise_a", "noise_b"), SEEDS)),
        "python": platform.python_version(), "libraries": versions,
        "reference_dtype": "float64 CPU; public floating inputs quantized once to float32",
        "reference": "Deepwave scalar + PyTorch autograd; unregularized bounded plain FWI on exactly the same noisy gathers",
        "paper": "Zhu et al. (2022), Integrating deep neural networks with FWI: reparametrization, regularization, and uncertainty quantification, Geophysics 87(1). Concept only; compact synthetic experiment, not their published architecture/data.",
        "acquisition": {"shape": [24, 32], "spacing_m": 10.0, "dt_s": 0.001, "nt": 280, "source_frequency_Hz": 18.0,
                        "source_peak_s": 0.06, "shots": 3, "sources_per_shot": 1, "receivers_per_shot": 15,
                        "source_grid_indices": [[2, 6], [2, 16], [2, 25]], "receiver_depth_index": 21,
                        "receiver_x_indices": list(range(1, 31, 2)), "accuracy": 4, "pml_width": 12,
                        "pml_freq_Hz": 18.0, "max_vel_m_s": 2600.0, "internal_step_ratio": 1,
                        "model_gradient_sampling_interval": 1, "freq_taper_frac": 0.0, "time_pad_frac": 0.0, "time_taper": False},
        "synthetic_model": "v0=2000; truth=v0+360 exp(-((z-12)/5)^2-((x-12)/7)^2)-240 exp(-((z-15)/4)^2-((x-24)/5)^2), z,x in cell indices. Two independent 0.5*RMS(clean) Gaussian-noise realizations; same geology. All physical cells included in error.",
        "network": "Fixed latent length 4 -> dense tanh width 8 -> linear 5x7 image -> bilinear align_corners=True 24x32 -> v0+500*tanh(image). Train all 355 parameters; W1/sqrt(4), W2/sqrt(8); W1 normal, all remaining initial parameters zero. Grid mode trains 768 independent q with v=v0+500*tanh(q).",
        "optimizer": {"name": "Adam", "steps": STEPS, "learning_rate": LEARNING_RATE, "beta1": 0.9, "beta2": 0.999,
                      "epsilon": 1e-8, "bias_correction": True, "clipping": False, "weight_decay": 0.0,
                      "schedule": "fixed; return last iterate; no truth/clean data in implementation inputs"},
        "objective": "0.5*mean(((predicted-observed)/data_scale)^2), data_scale=float32 RMS(clean) fixed by judge; no explicit regularizer",
        "finite_difference": "Centered difference at h=0.01 in saved unit directions: even/odd grid blocks and each of W1,b1,W2,b2. Directions align with the float64 block gradient and are quantized to float32. Probe parameters are nonzero. Both float64 and float32 FD calibrated.",
        "tolerances": {key: {"value": TOL[key], "reason": reason} for key, reason in TOL["REASONS"].items()},
        "metrics": metrics, "calibration": calibrations, "files": files,
        "reference_scalar_sha256": hashlib.sha256(Path(deepwave.scalar.__code__.co_filename).read_bytes()).hexdigest(),
        "reference_native_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in Path(deepwave.__file__).parent.glob("libdeepwave*") if p.is_file()},
        "elapsed_seconds": time.monotonic()-start,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True)+"\n")
    assert sum(f.stat().st_size for f in dest.iterdir() if f.is_file()) < TOL["MAX_GOLDEN_BYTES"]
    print(f"Wrote {len(files)} fixtures in {time.monotonic()-start:.2f}s; float32 calibration and physical sanity passed.", flush=True)


if __name__ == "__main__":
    main()
