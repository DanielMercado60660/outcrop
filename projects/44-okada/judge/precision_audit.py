"""Judge-only float32 feasibility measurements; never imported by tests.

The published decimals remain the oracle. The supplied independent
transcription (data/okada/reviewer/dc3d.py) is used only to size cancellation allowances and check FD steps.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ORDER = [0, 1, 2, 3, 6, 4, 7, 5, 8]


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("okada_reviewer", path)
    module = importlib.util.module_from_spec(spec)
    # Reference data is read-only, including its bytecode cache location.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def evaluate(ref, args, dtype):
    """Return (N,9) in the printed table's column order."""
    rows = []
    for x, y, z in args["coordinates"]:
        if args["source"] == "point":
            u = ref.dc3d0_surface(args["alpha"], x, y, args["depth"],
                                 args["dip_deg"], *args["slip"], dt=dtype)
        else:
            u = ref.dc3d(args["alpha"], x, y, z, args["depth"],
                         args["dip_deg"], *args["bounds"], *args["slip"], dt=dtype)
        rows.append(u[ORDER].astype(np.float64))
    return np.array(rows)


def corner_scale(ref, args):
    """Absolute pre-sum block magnitude per unit L1 slip/potency.

    Sum absolute UA/UB/z*UC contributions over all real/image corners, before
    the Chinnery signs and coordinate rotation cancel. The absolute rotation
    also avoids accidental cancellation in the scale itself. This is a
    conservative componentwise condition scale, not a second solver.
    """
    total = np.zeros(12)
    names = ("UA0", "UB0") if args["source"] == "point" else ("UA", "UB", "UC")
    originals = {name: getattr(ref, name) for name in names}

    def wrap(name):
        def record(c, v, *rest):
            block = originals[name](c, v, *rest)
            a = np.abs(block.astype(np.float64))
            if name == "UC":
                a *= abs(float(rest[0]))  # z*UC; z-derivatives are not exposed.
            if args["source"] != "point":
                for i in range(0, 12, 3):
                    a[i+1:i+3] = [abs(c.cd)*a[i+1] + abs(c.sd)*a[i+2],
                                  abs(c.sd)*a[i+1] + abs(c.cd)*a[i+2]]
            total[:] += a
            return block
        return record

    try:
        for name in names:
            setattr(ref, name, wrap(name))
        evaluate(ref, args, np.float64)
    finally:
        for name, original in originals.items():
            setattr(ref, name, original)
    return total[ORDER] / np.sum(np.abs(args["slip"]))


def float32_variants(ref, args):
    """Two valid trig rounding choices: native f32 and f64 then round to f32."""
    native = evaluate(ref, args, np.float32)
    original = ref.Con0.__init__

    def rounded_trig(c, alpha, dip, dtype):
        original(c, alpha, dip, dtype)
        if dtype is np.float32:
            angle = float(dip) * (2 * np.pi) / 360
            sd, cd = np.sin(angle), np.cos(angle)
            if abs(cd) < ref.EPS:
                sd, cd = np.sign(sd), 0.0
            c.sd, c.cd = np.float32(sd), np.float32(cd)
            c.sdsd, c.cdcd = c.sd*c.sd, c.cd*c.cd
            c.sdcd = c.sd*c.cd
            c.s2d, c.c2d = np.float32(2)*c.sdcd, c.cdcd-c.sdsd

    try:
        ref.Con0.__init__ = rounded_trig
        rounded = evaluate(ref, args, np.float32)
    finally:
        ref.Con0.__init__ = original
    return np.stack((native, rounded))


def difference(values, coordinates):
    """Centered differences using the actual saved float32 coordinate spacing."""
    dx = float(coordinates[0, 0]) - float(coordinates[1, 0])
    dy = float(coordinates[2, 1]) - float(coordinates[3, 1])
    return np.stack(((values[0, :3]-values[1, :3])/dx,
                     (values[2, :3]-values[3, :3])/dy), axis=-1)
