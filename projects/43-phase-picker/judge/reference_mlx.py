"""Round-4 GPU reference, adapted from Wave-0 reviewer/mlx_ref_tl.py.

This is judge evidence, never an implementation or an input to one. Not runtime
validated in the builder sandbox: mlx.core import reports no Metal device.
All network/optimizer computation is float32 on the GPU; NumPy prepares saved
augmentation on the CPU. No random draws occur after model initialization.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import runpy
import time
from itertools import pairwise
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from round4_config import (
    BATCH,
    DATA,
    LABEL_SIGMA,
    PARAMETERS,
    ROOT,
    SCHEDULE,
    SEED,
    STEPS,
    WIDTHS,
    training_contract,
    work_dir,
)

from mlx import nn

T = runpy.run_path(str(ROOT / "tests" / "tolerances.py"))


def digest(file):
    with Path(file).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_hashes():
    return {name: digest(ROOT / "judge" / name)
            for name in ("reference_mlx.py", "round4_config.py")}


def write_json(file, value):
    Path(file).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def interpolate(x, length):
    """Half-pixel linear interpolation, align_corners=False, clamped edges."""
    old = x.shape[1]
    pos = np.clip((np.arange(length, dtype=np.float32) + np.float32(.5)) *
                  np.float32(old / length) - np.float32(.5), 0, old - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.minimum(lo + 1, old - 1)
    w = mx.array((pos - lo.astype(np.float32))[None, :, None])
    return x[:, mx.array(lo), :] * (1 - w) + x[:, mx.array(hi), :] * w


class Picker(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.stem = nn.Conv1d(3, WIDTHS[0], 7, padding=3)
        self.down = [nn.Conv1d(a, b, 7, stride=4, padding=3)
                     for a, b in pairwise(WIDTHS)]
        self.up = [nn.Conv1d(a+b, b, 7, padding=3)
                   for a, b in zip(WIDTHS[:0:-1], WIDTHS[-2::-1])]
        self.head = nn.Conv1d(WIDTHS[0], 3, 1)
        # Discard framework defaults. Explicit draw order gives exact MLX seed
        # reproducibility and PyTorch-equivalent weight AND bias distributions.
        mx.random.seed(seed)
        for layer in [self.stem, *self.down, *self.up, self.head]:
            bound = (layer.weight.shape[1] * layer.weight.shape[2]) ** -.5
            layer.weight = mx.random.uniform(-bound, bound, layer.weight.shape, dtype=mx.float32)
            layer.bias = mx.random.uniform(-bound, bound, layer.bias.shape, dtype=mx.float32)
        mx.eval(self.parameters())
        assert sum(a.size for _, a in tree_flatten(self.parameters())) == PARAMETERS

    def __call__(self, x):
        x = x - x.mean(axis=1, keepdims=True)
        x /= mx.maximum(mx.sqrt((x*x).mean(axis=(1, 2), keepdims=True)), 1e-8)
        x = nn.relu(self.stem(mx.clip(x, -10, 10)))
        skips = [x]
        for layer in self.down:
            x = nn.relu(layer(x))
            skips.append(x)
        for layer, skip in zip(self.up, skips[-2::-1]):
            x = nn.relu(layer(mx.concatenate([interpolate(x, skip.shape[1]), skip], axis=-1)))
        return self.head(x)


def batch_data(waveforms, labels, schedule, step):
    x = waveforms[schedule["batch_indices"][step]].copy()
    y = labels[schedule["batch_indices"][step]].copy()
    for b, shift in enumerate(schedule["shifts"][step]):
        x[b] = np.roll(x[b], int(shift), axis=0)
        if shift > 0:
            x[b, :shift] = 0
        elif shift < 0:
            x[b, shift:] = 0
        present = y[b] >= 0
        y[b, present] += shift
        y[b, (y[b] < 0) | (y[b] >= 3000)] = -1
    x *= schedule["polarities"][step, :, None, None]
    angles = schedule["rotation_angles"][step]
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    north, east = x[:, :, 1].copy(), x[:, :, 2].copy()
    x[:, :, 1], x[:, :, 2] = c*north - s*east, s*north + c*east
    t = np.arange(3000, dtype=np.float32)
    phase = np.exp(np.float32(-.5) * ((t[None, :, None] - y[:, None, :].astype(np.float32)) /
                                     np.float32(LABEL_SIGMA))**2)
    phase *= (y >= 0)[:, None, :]
    phase /= np.maximum(np.float32(1), phase.sum(axis=-1, keepdims=True))
    target = np.concatenate([1-phase.sum(axis=-1, keepdims=True), phase], axis=-1)
    assert x.dtype == target.dtype == np.float32
    return mx.array(x), mx.array(target)


def gpu_setup():
    assert mx.metal.is_available(), "Round-4 calibration requires MLX on the GPU"
    mx.set_default_device(mx.gpu)


def load_split(split):
    meta = json.loads((ROOT / "judge" / "golden" / "meta.json").read_text())
    assert digest(DATA / f"{split}.npz") == meta["data_files"][split]["sha256"]
    with np.load(DATA / f"{split}.npz", allow_pickle=False) as z:
        return (np.ascontiguousarray(z["waveforms"].transpose(0, 2, 1)),
                np.column_stack([z["p_idx"], z["s_idx"]]))


def load_training(work):
    report = json.loads((work / "training.json").read_text())
    assert report["source_sha256"] == source_hashes(), "Source changed during the experiment"
    assert report["schedule_sha256"] == digest(SCHEDULE)
    assert report["contract"] == training_contract()
    assert report["mlx"] == importlib.metadata.version("mlx")
    assert report["numpy"] == np.__version__
    for name, expected in report["checkpoint_sha256"].items():
        assert digest(work / name) == expected
    return report


def train_chunk(seed=SEED, seconds=220):
    """Resume exactly; never turn a timed partial run into a final reference."""
    invocation_start = time.monotonic()
    gpu_setup()
    work = work_dir(seed)
    work.mkdir(parents=True, exist_ok=True)
    assert not (work / "completion.json").exists(), "Sealed evidence must not be silently replaced"
    net = Picker(seed)
    opt = optim.Adam(learning_rate=.001, betas=[.9, .999], eps=1e-8, bias_correction=True)
    step, elapsed, chunks = 0, 0., []
    loss_value = None
    if (work / "training.json").exists():
        report = load_training(work)
        assert report["seed"] == seed
        net.load_weights(str(work / "model.npz"))
        opt.state = tree_unflatten(mx.load(str(work / "optimizer.npz")))
        step, elapsed, chunks = report["steps"], report["training_seconds"], report["chunks"]
        loss_value = report["last_loss"]
    x_all, y_all = load_split("train")
    with np.load(SCHEDULE, allow_pickle=False) as z:
        schedule = {k: z[k] for k in z.files}
    mx.eval(net.parameters(), opt.state)
    setup_seconds = time.monotonic() - invocation_start

    def objective(model, x, target):
        return -(target * nn.log_softmax(model(x), axis=-1)).sum(axis=-1).mean()

    value_and_grad = nn.value_and_grad(net, objective)
    start_step = step
    start = time.monotonic()
    while (step < STEPS and time.monotonic() - start < seconds and
           elapsed + time.monotonic() - start < T["REFERENCE_MAX_TRAINING_S"]):
        x, target = batch_data(x_all, y_all, schedule, step)
        opt.learning_rate = mx.array(schedule["learning_rates"][step])
        loss, grad = value_and_grad(net, x, target)
        opt.update(net, grad)
        # Synchronize EVERY update: timing includes completed GPU work, data
        # preparation and optimizer work, not just lazy graph construction.
        mx.eval(net.parameters(), opt.state, loss)
        loss_value = float(loss)
        assert np.isfinite(loss_value)
        step += 1
        if step % 500 == 0:
            print(f"seed={seed} step={step}/{STEPS} loss={loss_value:.7f} chunk_s={time.monotonic()-start:.2f}", flush=True)
    chunk_seconds = time.monotonic() - start
    elapsed += chunk_seconds
    save_start = time.monotonic()
    net.save_weights(str(work / "model.npz"))
    mx.savez(str(work / "optimizer.npz"), **dict(tree_flatten(opt.state)))
    checkpoint_hashes = {name: digest(work / name) for name in ("model.npz", "optimizer.npz")}
    chunks.append({"start_step": start_step, "end_step": step, "setup_seconds": setup_seconds,
                   "training_seconds": chunk_seconds,
                   "checkpoint_seconds": time.monotonic() - save_start})
    report = {
        "seed": seed, "steps": step, "planned_steps": STEPS, "last_loss": loss_value,
        "training_seconds": elapsed, "chunks": chunks,
        "device": str(mx.default_device()), "device_info": mx.metal.device_info(),
        "mlx": importlib.metadata.version("mlx"), "numpy": np.__version__,
        "dtype": "float32", "parameters": PARAMETERS,
        "contract": training_contract(), "source_sha256": source_hashes(),
        "schedule_sha256": digest(SCHEDULE), "checkpoint_sha256": checkpoint_hashes,
        "status": "COMPLETE" if step == STEPS else "BUDGET_EXCEEDED" if elapsed >= T["REFERENCE_MAX_TRAINING_S"] else "IN_PROGRESS",
    }
    write_json(work / "training.json", report)
    print(json.dumps({k: report[k] for k in ("seed", "steps", "training_seconds", "status")}), flush=True)
    assert elapsed <= T["REFERENCE_MAX_TRAINING_S"], "Budget exhausted: do not evaluate or lower gates"


def decode_probabilities(probabilities):
    rows = []
    for trace in probabilities:
        row = []
        for phase in (1, 2):
            active = trace[:, phase] >= T["PICK_THRESHOLD"]
            edges = np.diff(np.r_[False, active, False].astype(np.int8))
            row.append([int(a + np.argmax(trace[a:b, phase]))
                        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))])
        rows.append(row)
    return rows


def score_picks(picks, labels):
    """Independent judge scorer; tests re-score these picks with tests/metrics."""
    counts, medians = [], []
    for phase in (0, 1):
        tp = fp = fn = 0
        errors = []
        for row, truth in zip(picks, labels[:, phase]):
            distances = np.abs(np.array(row[phase], dtype=np.int64) - truth)
            match = truth >= 0 and len(distances) > 0 and distances.min() <= T["MATCH_TOLERANCE_S"] / .01
            tp += int(match)
            fp += len(distances) - int(match)
            fn += int(truth >= 0) - int(match)
            if match:
                errors.append(float(distances.min()) * .01)
        counts.append([tp, fp, fn])
        medians.append(float(np.median(errors)) if errors else None)
    c = np.array(counts)
    return {"step": STEPS, "counts_P_S": counts, "matched_median_s_P_S": medians,
            "f1_P_S": (2*c[:, 0] / (2*c[:, 0] + c[:, 1] + c[:, 2])).tolist(),
            "f1_micro": float(2*c[:, 0].sum() / (2*c[:, 0] + c[:, 1] + c[:, 2]).sum())}


def probabilities(net, x):
    return np.array(mx.softmax(net(mx.array(x)), axis=-1))


def inference_audit(net, waveforms, labels):
    def ratio(a, b):
        return float(np.max(np.abs(a.astype(np.float64) - b) /
                            (T["PARITY_ATOL"] + T["PARITY_RTOL"] * np.abs(b))))

    first = probabilities(net, waveforms[:32])
    deterministic = bool(np.array_equal(first, probabilities(net, waveforms[:32])))
    ratios = []
    for i in (0, 1, int(np.flatnonzero(labels[:, 1] == -1)[0])):
        start = (i // BATCH) * BATCH
        p = probabilities(net, waveforms[start:start+BATCH])
        ratios.append(ratio(probabilities(net, waveforms[i:i+1]), p[i-start:i-start+1]))
    with np.load(ROOT / "judge" / "golden" / "normalization.npz", allow_pickle=False) as z:
        gain = ratio(probabilities(net, waveforms[:3] * z["gains"]), probabilities(net, waveforms[:3]))
        constant = probabilities(net, z["constant_waveforms"])
    result = {"deterministic": deterministic, "batch_max_allowance_ratio": max(ratios),
              "gain_max_allowance_ratio": gain, "constant_max_allowance_ratio": ratio(*constant)}
    assert deterministic and max(*ratios, gain, ratio(*constant)) <= 1, result
    return result


def evaluate(seed=SEED):
    gpu_setup()
    work = work_dir(seed)
    training = load_training(work)
    assert training["steps"] == STEPS and training["training_seconds"] <= T["REFERENCE_MAX_TRAINING_S"]
    assert not (work / "completion.json").exists()
    net, untrained = Picker(seed), Picker(seed)
    net.load_weights(str(work / "model.npz"))
    for split in ("val", "test"):
        x, labels = load_split(split)
        picks, initial_picks = [], []
        start = time.monotonic()
        for i in range(0, len(x), BATCH):
            picks.extend(decode_probabilities(probabilities(net, x[i:i+BATCH])))
        seconds = time.monotonic() - start
        if split == "test":
            for i in range(0, len(x), BATCH):
                initial_picks.extend(decode_probabilities(probabilities(untrained, x[i:i+BATCH])))
            write_json(work / "untrained_picks.json", {"seed": seed, "step": 0, "picks": initial_picks})
        write_json(work / f"{split}_picks.json", {"step": STEPS, "picks": picks})
        report = score_picks(picks, labels)
        report["inference_and_decode_seconds"] = seconds
        if split == "test":
            report["inference_audit"] = inference_audit(net, x, labels)
        write_json(work / f"{split}_metrics.json", report)
        print(split, json.dumps(report), flush=True)


def seal(seed=SEED):
    work = work_dir(seed)
    training = load_training(work)
    assert training["steps"] == STEPS and training["training_seconds"] <= T["REFERENCE_MAX_TRAINING_S"]
    for split in ("val", "test"):
        assert json.loads((work / f"{split}_metrics.json").read_text())["step"] == STEPS
    evidence = {p.name: digest(p) for p in sorted(work.glob("*.json")) if p.name != "completion.json"}
    for name in ("model.npz", "optimizer.npz"):
        (work / name).unlink()
    write_json(work / "completion.json", {
        "steps": STEPS, "evidence_sha256": evidence, "source_sha256": source_hashes(),
        "checkpoint_removed": True, "note": "Reports and decoded picks only; no trained weights shipped",
    })
    print(f"Sealed MLX seed {seed}; trained weights and optimizer state removed.", flush=True)
