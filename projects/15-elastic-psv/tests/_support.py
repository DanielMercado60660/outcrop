"""Pure NumPy test helpers. All synthetic arrays come from stored goldens."""
import numpy as np
from outcrop_harness import call, golden, rel_err

from . import tolerances as tol

CASES = ("heterogeneous", "free_surface", "zero",
         "plane_P_0", "plane_P_1", "plane_P_2",
         "plane_S_0", "plane_S_1", "plane_S_2", "lamb", "pml", "reflective")
COMPONENTS = ("vy", "vx", "p")
FIELDS = ("vp", "vs", "rho", "source_y", "source_x")


def arguments(g, gradient=False):
    out = {k: np.ascontiguousarray(g[k], dtype=np.float32) for k in FIELDS}
    out.update({k: np.ascontiguousarray(g[k], dtype=np.int64)
                for k in ("source_locations", "receiver_locations")})
    out.update(spacing=g["spacing"].tolist(), dt=float(g["dt"]),
               pml_width=g["pml_width"].tolist(), pml_freq=float(g["pml_freq"]),
               max_vel=float(g["max_vel"]))
    if gradient:
        out.update({k: np.ascontiguousarray(g[k], dtype=np.float32)
                    for k in ("observed_y", "observed_x")})
        out["data_scale"] = float(g["data_scale"])
    return out


def forward(path, name):
    g = golden(name)
    args = arguments(g)
    before = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    out = call(path, "elastic", **args)
    assert set(out) == set(COMPONENTS)
    for k in COMPONENTS:
        assert out[k].dtype == np.float32, (k, out[k].dtype)
        assert out[k].shape == g["expected_"+k].shape
        assert np.isfinite(out[k]).all(), k
    for k, v in before.items():
        np.testing.assert_array_equal(args[k], v, err_msg=f"input mutated: {k}")
    return g, out


def component_floor(case, g, component):
    if not case.startswith("plane_"):
        return 0.0
    absent = (case.startswith("plane_P") and component == "vx") or (
        case.startswith("plane_S") and component in ("vy", "p"))
    if not absent:
        return 0.0
    velocity = max(np.max(np.abs(g["expected_vy"])), np.max(np.abs(g["expected_vx"])))
    return velocity if component != "p" else velocity*np.max(g["rho"]*g["vp"])


def compare(a, b, threshold=tol.PARITY, floor=0.0):
    error = rel_err(a, b, floor=floor)
    # Truly zero reference channels (plane-wave transverse components, vacuum
    # pressure, zero-source case) must be exactly zero, avoiding an arbitrary
    # dimensionally inconsistent absolute floor.
    limit = threshold if np.any(b) or floor > 0 else tol.ZERO
    assert error <= limit, f"relative max error {error:.8g} > {limit:.8g}"


def gradient(path, case="gradient"):
    g = golden(case)
    args = arguments(g, gradient=True)
    before = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    out = call(path, "misfit_and_grad", **args)
    keys = {"objective"} | {"grad_"+k for k in FIELDS}
    assert set(out) == keys
    for k in keys:
        assert out[k].dtype == np.float32, (k, out[k].dtype)
        assert out[k].shape == g["expected_"+k].shape
        assert np.isfinite(out[k]).all(), k
    for k, v in before.items():
        np.testing.assert_array_equal(args[k], v, err_msg=f"input mutated: {k}")
    return g, args, out


def objective(out, g):
    # Accumulate in float64 so the centered check measures the float32
    # propagator, without additional float32 scalar reduction cancellation.
    return sum(np.sum((np.asarray(out["v"+c], np.float64) -
                       np.asarray(g["observed_"+c], np.float32).astype(np.float64))**2)
               for c in ("y", "x")) / (2*out["vy"].size*float(g["data_scale"])**2)


def arrival_times(traces, g):
    dt = float(g["dt"])
    times = (np.arange(traces.shape[-1])-.5)*dt
    arrivals = []
    for trace, window in zip(traces[0], g["arrival_windows"]):
        candidates = np.flatnonzero((times >= window[0]) & (times <= window[1]))
        idx = int(candidates[np.argmax(np.abs(trace[candidates]))])
        assert candidates[0] < idx < candidates[-1], "peak on search window boundary"
        a, b, c = np.abs(np.asarray(trace[idx-1:idx+2], np.float64))
        assert b > a and b > c, "Rayleigh peak must be resolved"
        shift = .5*(a-c)/(a-2*b+c)
        arrivals.append((idx+shift-.5)*dt)
    return np.asarray(arrivals)


def reflection_error(out, g):
    start = int(g["late_start"])
    residual = np.concatenate([(out[k]-g["open_"+k])[..., start:].reshape(2, -1)
                               for k in ("vy", "vx")], axis=1)
    return np.linalg.norm(residual, axis=1)/g["incident_norm"]
