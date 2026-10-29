"""Offline PyTorch reference experiment, imported only by make_goldens.

No weights from this experiment are an implementation input. Checkpoints are
local judge work products, never goldens. Chunking permits <5 minute commands;
resuming restores weights, Adam state and the frozen schedule position exactly.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 43043
REFERENCE_SEEDS = (43043, 43044, 43045)
STEPS = 6000
BATCH = 32
LEARNING_RATE = 0.001
FINAL_LEARNING_RATE = 0.0001
LABEL_SIGMA = 10.0
DATA = Path(__file__).resolve().parents[3] / "data" / "phasepicker"
WORK_ROOT = Path(__file__).resolve().parent / "reference_round3"
TOLERANCES = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests" / "tolerances.py"))


class Picker(nn.Module):
    """Small PhaseNet-style 1-D U-Net; ZNE input, noise/P/S output."""

    def __init__(self):
        super().__init__()
        widths = [8, 16, 32, 64, 96]
        self.stem = nn.Conv1d(3, widths[0], 7, padding=3)
        self.down = nn.ModuleList([
            nn.Conv1d(a, b, 7, stride=4, padding=3)
            for a, b in pairwise(widths)])
        self.up = nn.ModuleList([
            nn.Conv1d(a + b, b, 7, padding=3)
            for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.head = nn.Conv1d(widths[0], 3, 1)

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / torch.sqrt((x*x).mean(dim=(1, 2), keepdim=True)).clamp_min(1e-8)
        x = x.clamp(-10, 10)
        x = F.relu(self.stem(x))
        skips = [x]
        for layer in self.down:
            x = F.relu(layer(x))
            skips.append(x)
        for layer, skip in zip(self.up, skips[-2::-1]):
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = F.relu(layer(torch.cat([x, skip], dim=1)))
        return self.head(x)


def batch_data(waveforms, labels, indices, shifts, polarities):
    """Consume saved sampling, zero-padded shifts and polarity exactly."""
    x = waveforms[indices].copy()
    y = labels[indices].copy()
    for b, shift in enumerate(shifts):
        x[b] = np.roll(x[b], int(shift), axis=-1)
        if shift > 0:
            x[b, :, :shift] = 0
        elif shift < 0:
            x[b, :, shift:] = 0
        valid = y[b] >= 0
        y[b, valid] += shift
        y[b, (y[b] < 0) | (y[b] >= 3000)] = -1
    x *= polarities[:, None, None]
    t = np.arange(3000, dtype=np.float32)
    phase = np.exp(-0.5*((t[None, None, :] - y[:, :, None])/LABEL_SIGMA)**2)
    phase *= (y >= 0)[:, :, None]
    phase /= np.maximum(1, phase.sum(axis=1, keepdims=True))
    target = np.concatenate([1-phase.sum(axis=1, keepdims=True), phase], axis=1)
    return torch.from_numpy(x), torch.from_numpy(target.astype(np.float32))


def schedule_path(seed):
    suffix = "" if seed == SEED else f"_{seed}"
    return Path(__file__).resolve().parent / "golden" / f"training_schedule{suffix}.npz"


def work_dir(seed):
    assert seed in REFERENCE_SEEDS
    return WORK_ROOT / str(seed)


def train_chunk(seconds=50, seed=SEED):
    work = work_dir(seed)
    work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Picker().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    with np.load(schedule_path(seed), allow_pickle=False) as z:
        schedule = {k: z[k] for k in z.files}
    step, elapsed = 0, 0.0
    checkpoint = work / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        step, elapsed = saved["step"], saved["elapsed_s"]
        assert saved["seed"] == seed and saved["planned_steps"] == STEPS
    else:
        # A new reproduction run must not inherit old evaluation evidence.
        for name in ("val_metrics.json", "test_metrics.json", "val_picks.json", "test_picks.json", "untrained_picks.json", "completion.json"):
            (work / name).unlink(missing_ok=True)
    with np.load(DATA / "train.npz", allow_pickle=False) as z:
        waveforms = z["waveforms"]
        labels = np.column_stack([z["p_idx"], z["s_idx"]])
    print(f"reference: {device=}, start={step}/{STEPS}, training_seconds={elapsed:.2f}", flush=True)
    start = time.monotonic()
    last_loss = None
    while step < STEPS and time.monotonic() - start < seconds:
        x, y = batch_data(waveforms, labels, schedule["batch_indices"][step],
                          schedule["shifts"][step], schedule["polarities"][step])
        for group in opt.param_groups:
            group["lr"] = float(schedule["learning_rates"][step])
        opt.zero_grad(set_to_none=True)
        loss = -(y.to(device) * F.log_softmax(net(x.to(device)), dim=1)).sum(dim=1).mean()
        loss.backward()
        opt.step()
        step += 1
        last_loss = float(loss.detach().cpu())
        if step % 100 == 0:
            print(f"step={step}, loss={last_loss:.6f}, chunk_seconds={time.monotonic()-start:.2f}", flush=True)
    elapsed += time.monotonic() - start
    torch.save({"model": net.cpu().state_dict(), "optimizer": opt.state_dict(),
                "seed": seed, "planned_steps": STEPS, "step": step, "elapsed_s": elapsed}, checkpoint)
    report = {"seed": seed, "steps": step, "planned_steps": STEPS, "batch_size": BATCH,
              "training_seconds": elapsed, "device": device, "torch": torch.__version__,
              "threads": torch.get_num_threads(), "deterministic_algorithms": True,
              "training_dtype": "float32", "parameters": sum(p.numel() for p in net.parameters()),
              "last_loss": last_loss, "widths": [8, 16, 32, 64, 96],
              "learning_rate_initial": LEARNING_RATE, "learning_rate_final": FINAL_LEARNING_RATE,
              "learning_rate_schedule": "saved float32 cosine, inclusive endpoints",
              "schedule_sha256": hashlib.sha256(schedule_path(seed).read_bytes()).hexdigest(),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (work / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def evaluate(seed=SEED):
    """Held-out predictions from final weights; no fitting or test-label tuning."""
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    net = Picker()
    work = work_dir(seed)
    saved = torch.load(work / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == STEPS, "Finish the fixed training budget before held-out evaluation"
    training = json.loads((work / "training.json").read_text())
    assert hashlib.sha256(schedule_path(seed).read_bytes()).hexdigest() == training["schedule_sha256"]
    net.load_state_dict(saved["model"])
    net.eval()
    torch.manual_seed(seed)
    untrained = Picker().eval()

    def decode_probabilities(probabilities):
        rows = []
        for trace in probabilities:
            row = []
            for phase in (1, 2):
                active = trace[phase] >= TOLERANCES["PICK_THRESHOLD"]
                edges = np.diff(np.r_[False, active, False].astype(np.int8))
                row.append([int(a + np.argmax(trace[phase, a:b]))
                            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))])
            rows.append(row)
        return rows

    for split in ("val", "test"):
        with np.load(DATA / f"{split}.npz", allow_pickle=False) as z:
            x = z["waveforms"]
            labels = np.column_stack([z["p_idx"], z["s_idx"]])
        picks = []
        untrained_picks = []
        # Store decoded picks, not 108 MB of probabilities or trained weights.
        with torch.no_grad():
            for i in range(0, len(x), BATCH):
                batch = torch.from_numpy(x[i:i+BATCH])
                picks.extend(decode_probabilities(net(batch).softmax(dim=1).numpy()))
                if split == "test":
                    untrained_picks.extend(decode_probabilities(untrained(batch).softmax(dim=1).numpy()))
        (work / f"{split}_picks.json").write_text(json.dumps({"step": saved["step"], "picks": picks}))
        if split == "test":
            (work / "untrained_picks.json").write_text(json.dumps({"step": 0, "seed": seed, "picks": untrained_picks}))
        counts, residuals = [], []
        for phase in (0, 1):
            tp = fp = fn = 0
            errors = []
            for pred, truth in zip(picks, labels[:, phase]):
                d = np.abs(np.array(pred[phase], dtype=np.int64) - truth)
                match = truth >= 0 and len(d) > 0 and d.min() <= TOLERANCES["MATCH_TOLERANCE_S"] / 0.01
                tp += int(match)
                fp += len(d) - int(match)
                fn += int(truth >= 0) - int(match)
                if match:
                    errors.append(float(d.min()) * 0.01)
            counts.append([tp, fp, fn])
            residuals.append(float(np.median(errors)) if errors else None)
        c = np.array(counts)
        f1 = (2*c[:, 0]/(2*c[:, 0]+c[:, 1]+c[:, 2])).tolist()
        micro = float(2*c[:, 0].sum()/(2*c[:, 0]+c[:, 1]+c[:, 2]).sum())
        report = {"step": saved["step"], "counts_P_S": counts, "matched_median_s_P_S": residuals,
                  "f1_P_S": f1, "f1_micro": micro}
        if split == "test":
            report["inference_audit"] = inference_audit(net, x, labels)
        (work / f"{split}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(split, json.dumps(report), flush=True)


def inference_audit(net, waveforms, labels):
    """Check that the reference can meet declared inference numerical rules."""
    with torch.no_grad():
        def probabilities(x):
            return net(torch.from_numpy(x)).softmax(dim=1).numpy()

        def allowance_ratio(a, b):
            return float(np.max(np.abs(a.astype(np.float64)-b) /
                                (TOLERANCES["PARITY_ATOL"] + TOLERANCES["PARITY_RTOL"]*np.abs(b))))

        p32 = probabilities(waveforms[:32])
        deterministic = bool(np.array_equal(p32, probabilities(waveforms[:32])))
        absent_s = int(np.flatnonzero(labels[:, 1] == -1)[0])
        batch_ratios = []
        for i in (0, 1, absent_s):
            start = (i // BATCH) * BATCH
            block = probabilities(waveforms[start:start+BATCH])
            single = probabilities(waveforms[i:i+1])
            batch_ratios.append(allowance_ratio(single, block[i-start:i-start+1]))
        with np.load(Path(__file__).resolve().parent / "golden" / "normalization.npz", allow_pickle=False) as z:
            gains = z["gains"]
            constant = np.ascontiguousarray(z["constant_waveforms"].transpose(0, 2, 1))
        gain_ratio = allowance_ratio(probabilities(waveforms[:3] * gains), probabilities(waveforms[:3]))
        pc = probabilities(constant)
        constant_ratio = allowance_ratio(pc[0], pc[1])
        result = {"deterministic": deterministic, "batch_max_allowance_ratio": max(batch_ratios),
                  "gain_max_allowance_ratio": gain_ratio, "constant_max_allowance_ratio": constant_ratio}
        assert deterministic and max(*batch_ratios, gain_ratio, constant_ratio) <= 1, result
        return result


def seal(seed=SEED):
    """Keep auditable results, delete the trained checkpoint before submission."""
    work = work_dir(seed)
    training = json.loads((work / "training.json").read_text())
    assert training["steps"] == STEPS
    for split in ("val", "test"):
        assert json.loads((work / f"{split}_metrics.json").read_text())["step"] == STEPS
    checkpoint = work / "checkpoint.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == training["checkpoint_sha256"]
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(work.glob("*.json"))
             if p.name != "completion.json"}
    completion = {"steps": STEPS, "evidence_sha256": files,
                  "reference_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "checkpoint_removed": True,
                  "note": "No trained weights are distributed to implementations; only reports and decoded picks remain."}
    (work / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    checkpoint.unlink()
    print("Reference evidence sealed; trained checkpoint removed.", flush=True)
