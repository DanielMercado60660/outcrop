"""Independent Metal memory/replay benchmark, separate from the locked tests. Does not import MLX or goldens.

Default: the large benchmark grid size, with full storage and seven slots.
For a cheap complete curve use --nz 15 --nx 17 --nt 127 --pml 4
--budgets=-1,0,1,3,7,126. Run from the project directory after building.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np


def relative(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = float(np.max(np.abs(b)))
    difference = float(np.max(np.abs(a - b)))
    return difference / scale if scale else difference


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=project / ".target/release/outcrop-21")
    parser.add_argument("--output", type=Path, default=project / "bench/metal/results")
    parser.add_argument("--nz", type=int, default=224)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nt", type=int, default=385)
    parser.add_argument("--pml", type=int, default=8)
    parser.add_argument("--budgets", default="-1,7")
    args = parser.parse_args()
    if min(args.nz, args.nx) < 3 or args.nt < 2 or args.pml < 0:
        parser.error("need nz,nx>=3, nt>=2, pml>=0")
    budgets = list(dict.fromkeys([-1] + [int(x) for x in args.budgets.split(",")]))
    if any(b < -1 or b >= args.nt for b in budgets):
        parser.error("budgets must be -1 or in 0..nt-1")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(210021)
    z, x = np.meshgrid(np.arange(args.nz), np.arange(args.nx), indexing="ij")
    velocity = (1800 + 65 * np.sin(z / 19) * np.cos(x / 23)
                + rng.uniform(-20, 20, size=z.shape)).astype(np.float32)
    dt, freq = 0.001, 25.0
    phase = np.pi * freq * (np.arange(args.nt) * dt - 0.04)
    source = (0.03 * (1 - 2 * phase**2) * np.exp(-phase**2)).astype(np.float32)[None, None, :]
    arrays = {
        "velocity": velocity,
        "source_amplitudes": source,
        "source_locations": np.array([[[args.nz // 2, args.nx // 2]]], np.int64),
        "receiver_locations": np.array([[[args.nz // 2, args.nx // 2],
                                          [args.nz // 3, args.nx // 3],
                                          [2 * args.nz // 3, 2 * args.nx // 3]]], np.int64),
        "observed": np.zeros((1, 3, args.nt), np.float32),
    }
    scalar_args = {
        "grid_spacing": [8.0, 10.0], "dt": dt, "accuracy": 2,
        "pml_width": [args.pml] * 4, "pml_freq": freq, "max_vel": 2600.0,
    }
    rows, baseline = [], None
    with tempfile.TemporaryDirectory(dir=args.output) as td:
        root = Path(td)
        ind = root / "in"
        ind.mkdir()
        for name, value in arrays.items():
            np.save(ind / f"{name}.npy", value)
        for budget in budgets:
            outd = root / f"out_{budget}"
            outd.mkdir()
            (ind / "args.json").write_text(json.dumps(dict(scalar_args, checkpoint_budget=budget)))
            started = time.perf_counter()
            result = subprocess.run(
                ["/usr/bin/time", "-l", str(args.binary.resolve()), "misfit_and_grad", str(ind), str(outd)],
                capture_output=True, text=True, check=True,
            )
            elapsed = time.perf_counter() - started
            match = re.search(r"^\s*(\d+)\s+maximum resident set size\s*$", result.stderr, re.M)
            if not match:
                raise RuntimeError(f"Cannot parse macOS time -l peak RSS:\n{result.stderr}")
            out = {p.stem: np.load(p, allow_pickle=False) for p in outd.glob("*.npy")}
            np.savez(args.output / f"audit_{budget}.npz", **out)
            (args.output / f"time_{budget}.txt").write_text(result.stderr)
            errors = {name: 0.0 for name in ("objective", "gradient", "predicted")}
            if baseline is None:
                baseline = {name: out[name] for name in errors}
            else:
                errors = {name: relative(out[name], baseline[name]) for name in errors}
                if max(errors.values()) > 1e-6:
                    raise RuntimeError(f"Replay parity failed: budget={budget}, errors={errors}")
            row = dict(budget=budget, peak_rss_bytes=int(match[1]), elapsed_seconds=elapsed,
                       forward_steps=int(out["forward_steps"]), peak_checkpoints=int(out["peak_checkpoints"]),
                       checkpoint_bytes=int(out["checkpoint_bytes"]), **{f"{k}_relative_error": v for k, v in errors.items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.output / "measurements.json").write_text(json.dumps({
        "shape": [args.nz, args.nx], "nt": args.nt, "scalar_args": scalar_args,
        "measurement": "macOS time -l subprocess peak RSS in bytes, including runtime overhead",
        "rows": rows,
    }, indent=2))
    with (args.output / "measurements.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ordered = sorted(rows, key=lambda r: r["checkpoint_bytes"])
    extra = np.array([(r["forward_steps"] - args.nt) / args.nt for r in ordered])
    logical = np.array([r["checkpoint_bytes"] for r in ordered])
    measured = np.array([r["peak_rss_bytes"] for r in ordered])
    np.savez(args.output / "memory_recompute.npz", budgets=[r["budget"] for r in ordered],
             forward_steps=[r["forward_steps"] for r in ordered], checkpoint_bytes=logical,
             peak_rss_bytes=measured)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, memory, title in zip(axes, (logical, measured), ("Logical restart payload", "Measured subprocess peak RSS")):
        ax.plot(memory / 2**20, extra, "o-")
        for row, size, work in zip(ordered, memory, extra):
            ax.annotate(str(row["budget"]), (size / 2**20, work))
        ax.set(xlabel="Memory (MiB)", ylabel="Extra forward steps / nt", title=title)
        ax.grid(alpha=0.25)
    fig.savefig(args.output / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
