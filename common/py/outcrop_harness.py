"""Outcrop test harness: call an implementation path and load goldens.

Frozen by CONTRACT.md. Tests import this; do not copy it into projects.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PATHS = ["mlx", "metal"]


def project_dir() -> Path:
    """The project folder of the calling test (projects/NN-slug)."""
    for frame in inspect.stack():
        p = Path(frame.filename).resolve()
        if p.parent.name == "tests" and p.parent.parent.parent.name == "projects":
            return p.parent.parent
    env = os.environ.get("OUTCROP_PROJECT")
    if env:
        return Path(env).resolve()
    raise RuntimeError("cannot find project folder; run tests from projects/NN-slug/tests")


def _to_numpy(v):
    try:
        import mlx.core as mx
        if isinstance(v, mx.array):
            return np.array(v)
    except ImportError:
        pass
    return np.asarray(v)


_mlx_cache: dict[Path, object] = {}


def _load_mlx(pdir: Path):
    if pdir not in _mlx_cache:
        impl = pdir / "mlx" / "impl.py"
        spec = importlib.util.spec_from_file_location(f"outcrop_mlx_{pdir.name.replace('-', '_')}", impl)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(impl.parent))
        spec.loader.exec_module(mod)
        _mlx_cache[pdir] = mod
    return _mlx_cache[pdir]


def metal_binary(pdir: Path) -> Path:
    """Release binary outcrop-NN from the workspace target dir (override with OUTCROP_METAL_BIN)."""
    if os.environ.get("OUTCROP_METAL_BIN"):
        return Path(os.environ["OUTCROP_METAL_BIN"])
    nn = pdir.name.split("-")[0]
    target = Path(os.environ.get("CARGO_TARGET_DIR", pdir.parent.parent / "target"))
    return target / "release" / f"outcrop-{nn}"


def call(path: str, fn: str, **kwargs) -> dict[str, np.ndarray]:
    pdir = project_dir()
    if path == "mlx":
        out = getattr(_load_mlx(pdir), fn)(**kwargs)
        return {k: _to_numpy(v) for k, v in out.items()}
    if path == "metal":
        exe = metal_binary(pdir)
        with tempfile.TemporaryDirectory() as td:
            ind, outd = Path(td, "in"), Path(td, "out")
            ind.mkdir()
            outd.mkdir()
            scalars = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    np.save(ind / f"{k}.npy", np.ascontiguousarray(v))
                else:
                    scalars[k] = v
            (ind / "args.json").write_text(json.dumps(scalars))
            res = subprocess.run([str(exe), fn, str(ind), str(outd)], capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"{exe.name} {fn} failed ({res.returncode}):\n{res.stderr[-4000:]}")
            return {p.stem: np.load(p) for p in sorted(outd.glob("*.npy"))}
    raise ValueError(f"unknown path {path!r}")


def golden(name: str) -> dict[str, np.ndarray]:
    """Load judge/golden/<name>.npz as a dict of arrays."""
    pdir = project_dir()
    with np.load(pdir / "judge" / "golden" / f"{name}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def golden_meta() -> dict:
    return json.loads((project_dir() / "judge" / "golden" / "meta.json").read_text())


def rel_err(a, b, floor: float = 0.0) -> float:
    """max |a-b| / max(|b|, floor) over all entries."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    denom = max(float(np.max(np.abs(b))), floor)
    return float(np.max(np.abs(a - b)) / denom) if denom > 0 else float(np.max(np.abs(a - b)))
