"""Package published checks and audit float32 allowances with the independent
precision-audit transcription (data/okada/reviewer/dc3d.py).

Run with the repository's pinned Python environment. No network or randomness.
The precision-audit outputs size tolerances only; they never replace the
published table/manual oracle values.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
from precision_audit import (
    corner_scale,
    difference,
    evaluate,
    float32_variants,
    load_reference,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[2] / "data" / "okada" / "test_values.json"
REVIEWER = DATA.parent / "reviewer" / "dc3d.py"
DATA_SHA256 = "82b619a369917e79136d166f48551933226a81e79efde141c009e74eb5cfd1fb"
REVIEWER_SHA256 = "f067b5cb1b19a66b58fd46df5cd1b1b98eebf07e47ce9362ee8fcdfe6be77f2a"
KEYS = ["ux", "uy", "uz", "duxdx", "duxdy", "duydx", "duydy", "duzdx", "duzdy"]


def load_tolerances():
    # Single source of thresholds; importing by file avoids sys.path changes.
    spec = importlib.util.spec_from_file_location("okada_tolerances", ROOT.parent / "tests" / "tolerances.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    t = load_tolerances()
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == DATA_SHA256, "Published source changed; review before regenerating."
    assert hashlib.sha256(REVIEWER.read_bytes()).hexdigest() == REVIEWER_SHA256, "Precision-audit source changed; review before regenerating."
    data = json.loads(DATA.read_text())
    cases = data["okada1985_table2"]["cases"]
    values = np.array([[float(c["outputs"][k]) for k in KEYS] for c in cases], dtype=np.float64)
    half_units = np.array([
        [0.0 if c["outputs"][k] == "0" else
         0.5 * 10.0 ** (int(c["outputs"][k].split("E")[1]) - 3) for k in KEYS]
        for c in cases
    ], dtype=np.float64)
    coords = np.array([[c["inputs"][k] for k in ("x", "y", "z")] for c in cases], dtype=np.float64)
    slip = np.array([[c["inputs"][k] for k in ("U1", "U2", "U3")] for c in cases], dtype=np.float64)
    bounds = np.array([[0, c["inputs"].get("L", 0), 0, c["inputs"].get("W", 0)] for c in cases], dtype=np.float64)
    args = [{"coordinates": coords[i:i+1], "source": c["inputs"]["source"],
             "depth": c["inputs"]["d"], "dip_deg": c["inputs"]["dip_deg"],
             "slip": slip[i], "bounds": bounds[i], "alpha": 2/3} for i, c in enumerate(cases)]
    manual = data["okada1992_table2"]["cases"][0]
    m = manual["inputs"]
    ma = {"coordinates": np.array([[m["X"], m["Y"], m["Z"]]], dtype=np.float64),
          "source": "finite_rectangle", "depth": m["DEPTH"], "dip_deg": m["DIP"],
          "bounds": np.array([m[k] for k in ("AL1", "AL2", "AW1", "AW2")], dtype=np.float64),
          "slip": np.array([m[k] for k in ("DISL1", "DISL2", "DISL3")], dtype=np.float64), "alpha": 2/3}
    manual_values = np.array([[manual["outputs_float"][k] for k in KEYS[:3]]], dtype=np.float64)

    ref = load_reference(REVIEWER)
    scales = np.array([corner_scale(ref, a) for a in args])
    truth = np.concatenate([evaluate(ref, a, np.float64) for a in args])
    variants = np.concatenate([float32_variants(ref, a) for a in args], axis=1)
    floors = t.TABLE_CANCELLATION_ULPS * t.FLOAT32_EPS * np.sum(abs(slip), axis=1)[:, None] * scales
    limits = np.where(values == 0, t.TABLE_ZERO_ABS, half_units + floors)
    assert np.all(abs(truth-values) <= np.where(values == 0, t.TABLE_ZERO_ABS, half_units))
    assert np.all(abs(variants-truth) <= floors)
    assert np.all(abs(variants-values) <= limits)
    manual_scale = corner_scale(ref, ma)
    manual_truth = evaluate(ref, ma, np.float64)
    manual_variants = float32_variants(ref, ma)
    manual_floor = t.MANUAL_CANCELLATION_ULPS*t.FLOAT32_EPS*np.sum(abs(ma["slip"]))*manual_scale[:3]
    manual_limit = t.MANUAL_PRINTED_HALF_UNIT + t.MANUAL_REFERENCE_ABS + manual_floor
    assert np.all(abs(manual_truth[:, :3]-manual_values) <= t.MANUAL_REFERENCE_ABS)
    assert np.all(abs(manual_variants[:, :, :3]-manual_truth[:, :3]) <= manual_floor)
    assert np.all(abs(manual_variants[:, :, :3]-manual_values) <= manual_limit)

    # Quantize coordinates exactly as implementation inputs before the audit.
    fd = np.repeat(coords[[0, 3], None, :], 4, axis=1)
    fd_audit = []
    fd_scales = []
    for fi, ci in enumerate((0, 3)):
        h = t.FD_STEPS[fi]
        fd[fi, 0, 0] += h
        fd[fi, 1, 0] -= h
        fd[fi, 2, 1] += h
        fd[fi, 3, 1] -= h
        c = fd[fi].astype(np.float32)
        pa = {**args[ci], "coordinates": c.astype(np.float64)}
        u64 = evaluate(ref, pa, np.float64)
        u32 = float32_variants(ref, pa)
        exact_fd = difference(u64, c)
        numeric_fd = np.stack([difference(u, c) for u in u32])
        deriv64 = truth[ci, 3:].reshape(3, 2)
        deriv32 = variants[:, ci, 3:].reshape(2, 3, 2)
        ps = np.array([corner_scale(ref, {**pa, "coordinates": row[None, :]}) for row in pa["coordinates"]])
        dx = float(c[0, 0])-float(c[1, 0])
        dy = float(c[2, 1])-float(c[3, 1])
        noise_scale = np.stack(((ps[0, :3]+ps[1, :3])/dx,
                                (ps[2, :3]+ps[3, :3])/dy), axis=-1)
        noise_scale += scales[ci, 3:].reshape(3, 2)
        fd_scales.append(noise_scale)
        fd_floor = t.FD_CANCELLATION_ULPS[fi]*t.FLOAT32_EPS*np.sum(abs(slip[ci]))*noise_scale
        fd_limit = t.FD_ABS[fi]+t.FD_REL*abs(numeric_fd)+fd_floor
        ratio = float(np.max(abs(deriv32-numeric_fd)/fd_limit))
        fd_audit.append({"case": cases[ci]["name"], "step": h,
                         "float64_truncation_max_abs": float(np.max(abs(exact_fd-deriv64))),
                         "float32_fd_noise_max_abs": float(np.max(abs(numeric_fd-exact_fd))),
                         "combined_noise_over_eps_slip_corner_scale": float(np.max(abs((deriv32-numeric_fd)-(deriv64-exact_fd))/(t.FLOAT32_EPS*np.sum(abs(slip[ci]))*noise_scale))),
                         "cancellation_floor_range": [float(fd_floor.min()), float(fd_floor.max())],
                         "worst_allowance_ratio_both_variants": ratio})
        assert ratio <= 1, fd_audit[-1]

    # Size allowances against concrete convention mistakes as well as correct
    # float32 variants. These measurements never generate test expectations.
    sensitivity = {}
    sensitivity["reversed_slip_min_row_ratio"] = float(np.min(np.max(abs(-truth-values)/limits, axis=1)))
    flipped = truth.copy()
    flipped[:, 3:] *= -1
    sensitivity["reversed_derivatives_min_row_ratio"] = float(np.min(np.max(abs(flipped-values)/limits, axis=1)))
    wrong_depth = []
    for ci in range(3, 6):
        a = args[ci]
        wrong = evaluate(ref, {**a, "depth": a["depth"]+bounds[ci, 3]*np.sin(np.deg2rad(a["dip_deg"]))}, np.float64)[0]
        wrong_depth.append(float(np.max(abs(wrong-values[ci])/limits[ci])))
    sensitivity["top_edge_depth_min_case2_ratio"] = min(wrong_depth)
    wrong_dip = [float(np.max(abs(evaluate(ref, {**args[ci], "dip_deg": 90}, np.float64)[0]-values[ci])/limits[ci])) for ci in range(9, 12)]
    sensitivity["negative_dip_replaced_with_positive_min_row_ratio"] = min(wrong_dip)
    sensitivity["missing_point_source_min_row_ratio"] = float(np.min(np.max(abs(values[:3])/limits[:3], axis=1)))
    wrong_point = [float(np.max(abs(evaluate(ref, {**args[ci], "source": "finite_rectangle", "bounds": [0, 1, 0, 1]}, np.float64)[0]-values[ci])/limits[ci])) for ci in range(3)]
    sensitivity["point_replaced_by_unit_rectangle_min_row_ratio"] = min(wrong_point)
    assert min(sensitivity.values()) > 1

    out = ROOT / "golden"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "checks.npz", names=np.array([c["name"] for c in cases]),
        source=np.array([c["inputs"]["source"] for c in cases]), coordinates=coords,
        depth=np.array([a["depth"] for a in args], dtype=np.float64),
        dip_deg=np.array([a["dip_deg"] for a in args], dtype=np.float64),
        slip=slip, bounds=bounds, values=values, half_units=half_units, corner_scale=scales,
        fd_coordinates=fd, fd_corner_scale=np.array(fd_scales), manual_coordinates=ma["coordinates"],
        manual_depth=np.array(ma["depth"], dtype=np.float64),
        manual_dip=np.array(ma["dip_deg"], dtype=np.float64),
        manual_bounds=ma["bounds"], manual_slip=ma["slip"], manual_values=manual_values,
        manual_corner_scale=manual_scale[:3])
    audit = {"table_float32_vs_float64_max_abs": float(np.max(abs(variants-truth))),
             "table_error_over_eps_slip_corner_scale": float(np.max(np.divide(abs(variants-truth), t.FLOAT32_EPS*np.sum(abs(slip), axis=1)[:, None]*scales, out=np.zeros_like(variants), where=scales > 0))),
             "table_cancellation_floor_range": [float(floors.min()), float(floors.max())],
             "table_max_nonzero_relative_allowance": float(np.max(limits[values != 0]/abs(values[values != 0]))),
             "table_worst_allowance_ratio": float(np.max(abs(variants-values)/limits)),
             "manual_float64_displacement": manual_truth[0, :3].tolist(),
             "manual_float32_vs_float64_max_abs": float(np.max(abs(manual_variants[:, :, :3]-manual_truth[:, :3]))),
             "manual_error_over_eps_slip_corner_scale": float(np.max(abs(manual_variants[:, :, :3]-manual_truth[:, :3])/(t.FLOAT32_EPS*np.sum(abs(ma["slip"]))*manual_scale[:3]))),
             "manual_printed_error_each_variant": (manual_variants[:, 0, :3]-manual_values[0]).tolist(),
             "manual_cancellation_floor": manual_floor.tolist(), "manual_total_abs_limit": manual_limit.tolist(),
             "finite_difference": fd_audit, "wrong_convention_error_over_allowance": sensitivity}
    meta = {"reference": "Okada (1985) Table 2 and NIED DC3D manual Appendix 2",
        "source_data": "data/okada/test_values.json",
        "determinism": {"random_data": False, "seed": None},
        "citations": [data["okada1985_table2"]["citation"], data["okada1992_table2"]["citation"]],
        "inputs": {"table_cases": [c["inputs"] for c in cases], "manual": m,
                   "alpha": 2/3, "finite_difference_steps": list(t.FD_STEPS),
                   "finite_difference_cases": [cases[0]["name"], cases[3]["name"]]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pytest": importlib.metadata.version("pytest"),
                     "reference": "published decimal transcription",
                     "source_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
                     "precision_audit_reference": "data/okada/reviewer/dc3d.py",
                     "precision_audit_sha256": hashlib.sha256(REVIEWER.read_bytes()).hexdigest()},
        "tolerances": {"status": "round-3 threshold proposal; requires reviewer acceptance before lock",
          "constants": {k: v for k, v in vars(t).items() if k.isupper()},
          "table": "Nonzeros: printed half-unit + 8*eps32*L1(slip)*corner_scale; zeros: 1e-6. Absolute cancellation, not relative to cancelled output.",
          "manual": "5e-5 printed half-unit + 5e-5 REAL*4 reference allowance + 4*eps32*L1(slip)*corner_scale. Componentwise absolute limits.",
          "corner_scale": "Sum absolute pre-Chinnery UA/UB/z*UC blocks, with absolute coordinate rotation, per unit L1 slip/potency. Saved independently of implementation.",
          "parity": "1e-4 relative plus 1e-6 absolute; unchanged, independently reviewed as feasible.",
          "finite_difference": "1e-3 relative plus absolute truncation/noise intervals (1e-6 point, 2e-6 rectangle); rectangle adds 2*eps32*L1(slip)*fd_corner_scale. h=0.01 point, h=0.05 rectangle; float64 truncation and both float32 trig variants audited."},
        "precision_audit": audit,
        "notes": "No random data, hence no seed. Numeric oracle arrays float64. Printed decimals supply no extra reference precision. Precision-audit reference outputs used only for allowance sizing. Column order: " + ", ".join(KEYS)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    assert values.shape == (12, 9) and np.isfinite(values).all()
    assert np.all(np.abs(values) < 0.1) and np.all(np.isfinite(fd))
    print(f"Table: {values.size} values, range {values.min():.7g} to {values.max():.7g}, zeros {(values == 0).sum()}")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
