"""Float64 scikit-fmm travel times and independently checked PINN derivatives.

Run from this project with the repository's pinned Python environment. References
are imported only here. No project-24 implementation existed when this judge was
written: use its named scikit-fmm oracle directly, with an explicit zero-time
single grid-node source and order=2.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import skfmm
import torch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "judge" / "golden"
spec = importlib.util.spec_from_file_location("locked_tolerances", ROOT / "tests" / "tolerances.py")
tol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tol)
SEED = 29012020


def unpack(p, dimension, width):
    """Shared *layout*, not a reference-library implementation."""
    at = 0
    arrays = []
    for shape in ((2 * dimension, width), (width,), (width, width),
                  (width,), (width,), (1,)):
        size = int(np.prod(shape))
        arrays.append(p[at:at + size].reshape(shape))
        at += size
    assert at == len(p)
    return arrays


def numpy_evaluate(p, s, r, origin, scale, vref, width):
    """Explicit analytic forward derivative, entirely float64."""
    d = s.shape[1]
    w1, b1, w2, b2, w3, b3 = unpack(p, d, width)
    x = np.concatenate(((s - origin) / scale, (r - origin) / scale), axis=1)
    a = np.tanh(x @ w1 + b1)
    b = np.tanh(a @ w2 + b2)
    f = b @ w3 + b3[0]
    df = (((1 - b * b) * w3) @ w2.T * (1 - a * a)) @ w1[d:].T / scale
    delta = r - s
    distance = np.linalg.norm(delta, axis=1)
    unit = np.divide(delta, distance[:, None], out=np.zeros_like(delta),
                     where=distance[:, None] != 0)
    factor = np.exp(f) / vref
    time = distance * factor
    gradient = factor[:, None] * (unit + distance[:, None] * df)
    gradient[distance == 0] = 0
    return time, gradient


def numpy_loss(p, s, r, velocity, origin, scale, vref, width):
    _, gradient = numpy_evaluate(p, s, r, origin, scale, vref, width)
    residual = velocity**2 * np.sum(gradient**2, axis=1) - 1
    return np.mean(residual**2)


def torch_loss_grad(p, s, r, velocity, origin, scale, vref, width):
    """Independent float64 autograd through spatial and parameter derivatives."""
    params = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    src = torch.tensor(s, dtype=torch.float64)
    rec = torch.tensor(r, dtype=torch.float64, requires_grad=True)
    vel = torch.tensor(velocity, dtype=torch.float64)
    org = torch.tensor(origin, dtype=torch.float64)
    w1, b1, w2, b2, w3, b3 = unpack(params, s.shape[1], width)
    x = torch.cat(((src - org) / scale, (rec - org) / scale), dim=1)
    a = torch.tanh(x @ w1 + b1)
    b = torch.tanh(a @ w2 + b2)
    time = torch.linalg.vector_norm(rec - src, dim=1) / vref * torch.exp(b @ w3 + b3[0])
    spatial, = torch.autograd.grad(time.sum(), rec, create_graph=True)
    residual = vel**2 * torch.sum(spatial**2, dim=1) - 1
    loss = torch.mean(residual**2)
    grad, = torch.autograd.grad(loss, params)
    return loss.item(), grad.detach().numpy(), spatial.detach().numpy()


def relative_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def quantized(a):
    """Judge evaluates exactly representable float32 API inputs in float64."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def initial_parameters(rng, d, width, nonzero_output=False):
    values = []
    for shape in ((2 * d, width), (width,), (width, width),
                  (width,), (width,), (1,)):
        if len(shape) == 2:
            a = rng.normal(0, 1 / np.sqrt(shape[0]), shape)
        else:
            a = rng.normal(0, 0.12, shape)
        values.append(a.ravel())
    p = np.concatenate(values)
    if not nonzero_output:
        p[-width - 1:] = 0  # Starts at distance / reference_velocity.
    return quantized(p)


def derivative_case(d, rng):
    width, count = 4, 17
    origin = quantized(np.array([-900.0, 120.0, -300.0])[:d])
    scale, vref = 2000.0, 3000.0
    s = quantized(origin + scale * rng.uniform(0.15, 0.35, (count, d)))
    r = quantized(origin + scale * rng.uniform(0.55, 0.85, (count, d)))
    v = quantized(rng.uniform(2400, 3700, count))
    p = initial_parameters(rng, d, width, nonzero_output=True)
    time, spatial = numpy_evaluate(p, s, r, origin, scale, vref, width)
    loss, gradient, torch_spatial = torch_loss_grad(p, s, r, v, origin, scale, vref, width)
    assert relative_l2(torch_spatial, spatial) < tol.ORACLE_FD_REL
    assert abs(loss - numpy_loss(p, s, r, v, origin, scale, vref, width)) / loss < tol.ORACLE_FD_REL

    # Coordinatewise finite differences for EVERY parameter, not just directions.
    perturb = tol.PARAMETER_FD_STEP * np.eye(len(p))
    plus = quantized(p + perturb)
    minus = quantized(p - perturb)
    fd = np.array([(numpy_loss(a, s, r, v, origin, scale, vref, width)
                    - numpy_loss(b, s, r, v, origin, scale, vref, width))
                   / (a[j] - b[j]) for j, (a, b) in enumerate(zip(plus, minus))])
    fd_error = relative_l2(fd, gradient)
    assert fd_error <= tol.ORACLE_FD_REL, fd_error

    rec_plus = np.repeat(r[None, :, :], d, axis=0)
    rec_minus = rec_plus.copy()
    for axis in range(d):
        rec_plus[axis, :, axis] += tol.COORDINATE_FD_STEP
        rec_minus[axis, :, axis] -= tol.COORDINATE_FD_STEP
    spatial_fd = np.column_stack([
        (numpy_evaluate(p, s, rec_plus[j], origin, scale, vref, width)[0]
         - numpy_evaluate(p, s, rec_minus[j], origin, scale, vref, width)[0])
        / (2 * tol.COORDINATE_FD_STEP) for j in range(d)])
    coord_error = relative_l2(spatial_fd, spatial)
    assert coord_error <= tol.ORACLE_FD_REL, coord_error

    # Audit the *chosen steps* in float32 arithmetic as well as float64; this is
    # not an implementation or training run. It catches cancellation-prone tests.
    sf, rf, vf, of = (x.astype(np.float32) for x in (s, r, v, origin))
    float32_fd = np.array([
        (float(numpy_loss(a.astype(np.float32), sf, rf, vf, of, scale, vref, width))
         - float(numpy_loss(b.astype(np.float32), sf, rf, vf, of, scale, vref, width)))
        / (a[j] - b[j]) for j, (a, b) in enumerate(zip(plus, minus))])
    float32_fd_error = relative_l2(float32_fd, gradient)
    assert float32_fd_error <= tol.GRADIENT_FD_REL, float32_fd_error

    name = f"derivatives_{d}d"
    np.savez_compressed(DEST / f"{name}.npz", parameters=p, sources=s, receivers=r,
                        velocities=v, origin=origin, length_scale=scale,
                        reference_velocity=vref, width=width, time=time,
                        receiver_gradient=spatial, loss=loss, gradient=gradient,
                        parameter_plus=plus, parameter_minus=minus,
                        parameter_fd=fd, receiver_plus=rec_plus,
                        receiver_minus=rec_minus, receiver_fd=spatial_fd)
    return {"name": name, "dimension": d, "pairs": count, "width": width,
            "parameters": len(p), "loss": loss, "parameter_fd_relative_l2": fd_error,
            "float32_parameter_fd_relative_l2": float32_fd_error,
            "coordinate_fd_relative_l2": coord_error,
            "time_range_seconds": [float(time.min()), float(time.max())]}


def velocity_model(points, origin, scale, name):
    """Analytic positive synthetic speed; x,z (or x,y,z) in physical metres."""
    u = (points - origin) / scale
    if name == "homogeneous_2d":
        return np.full(u.shape[:-1], 3000.0)
    # Smooth depth trend + an off-centre low-speed lens. All derivatives smooth.
    center = np.array([0.60, 0.45] if u.shape[-1] == 2 else [0.60, 0.45, 0.55])
    radius_squared = np.sum(((u - center) / 0.22)**2, axis=-1)
    return 2600.0 + 1000.0 * u[..., -1] - 550.0 * np.exp(-radius_squared)


def sample_grid(field, points, origin, spacing):
    """Multilinear FMM sampling at the exact float32-quantized query positions.

    2D queries lie exactly on nodes. 193^3 spacing is not binary representable,
    so quantization can move 3D queries by <0.1 mm; interpolation accounts for it.
    """
    coordinate = (points - origin) / spacing
    lower = np.floor(coordinate).astype(int)
    fraction = coordinate - lower
    result = np.zeros(len(points), dtype=np.float64)
    for corner in itertools.product((0, 1), repeat=points.shape[1]):
        corner = np.asarray(corner)
        weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
        result += weight * field[tuple((lower + corner).T)]
    return result


def travel_case(name, d, rng):
    # 129^3 failed the homogeneous p95 <=1% oracle sanity check (1.38%).
    # Refine the oracle rather than spending the network's 1% error budget on it.
    size = 513 if d == 2 else 193
    scale, vref, width = 2000.0, 3000.0, 32
    origin = quantized(np.array([-900.0, 120.0, -300.0])[:d])
    spacing = scale / (size - 1)
    fractions = np.array([[0.25, 0.25], [0.75, 0.25], [0.50, 0.75]]) if d == 2 else np.array(
        [[0.25, 0.25, 0.25], [0.75, 0.25, 0.50], [0.50, 0.75, 0.75]])
    source_indices = (fractions * (size - 1)).astype(int)
    source_locations = quantized(origin + source_indices * spacing)
    axes = [origin[j] + np.arange(size) * spacing for j in range(d)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    speed = velocity_model(grid, origin, scale, name)

    eval_sources, eval_receivers, expected, group, exact_times, homogeneous_fmm = [], [], [], [], [], []
    colloc_sources, colloc_receivers = [], []
    summaries = []
    baseline_errors = []
    for i, (index, source) in enumerate(zip(source_indices, source_locations)):
        phi = np.ones((size,) * d, dtype=np.float64)
        phi[tuple(index)] = 0.0
        time_grid = np.asarray(skfmm.travel_time(phi, speed, dx=[spacing] * d, order=2))
        assert time_grid[tuple(index)] == 0
        assert np.all(np.isfinite(time_grid)) and np.all(time_grid >= 0)

        # Interior grid nodes, >300 m from source: never include singular or
        # zero-time values in relative-error denominators. Unique per source.
        candidate = rng.integers(int(0.08 * (size - 1)), int(0.92 * (size - 1)), (3000, d))
        candidate = np.unique(candidate, axis=0)
        xyz = origin + candidate * spacing
        keep = np.linalg.norm(xyz - source, axis=1) >= 300
        candidate = candidate[keep]
        # np.unique sorts rows: shuffle in the judge to cover the whole domain.
        candidate = candidate[rng.permutation(len(candidate))[:384]]
        receivers = quantized(origin + candidate * spacing)
        ref = sample_grid(time_grid, receivers, origin, spacing)
        analytic = np.linalg.norm(receivers - source, axis=1) / vref
        errors = np.abs(analytic - ref) / ref
        baseline_errors.append(float(np.median(errors)))
        if name == "homogeneous_2d":
            homogeneous_reference = ref
        else:
            # Independently audit the SAME point-source stencil and query nodes
            # in a homogeneous medium, including the coarser 3D grid.
            homogeneous_grid = np.asarray(skfmm.travel_time(
                phi, np.full_like(speed, vref), dx=[spacing] * d, order=2))
            homogeneous_reference = sample_grid(homogeneous_grid, receivers, origin, spacing)
        homogeneous_errors = np.abs(homogeneous_reference - analytic) / analytic
        homogeneous_median = float(np.median(homogeneous_errors))
        homogeneous_p95 = float(np.percentile(homogeneous_errors, 95))
        assert homogeneous_median <= tol.FMM_HOMOGENEOUS_MEDIAN_REL, homogeneous_median
        assert homogeneous_p95 <= tol.FMM_HOMOGENEOUS_P95_REL, homogeneous_p95
        # Any path inside this convex box obeys these global speed bounds.
        distance = np.linalg.norm(receivers - source, axis=1)
        assert np.all(ref >= distance / speed.max() * (1 - tol.FMM_HOMOGENEOUS_P95_REL))
        assert np.all(ref <= distance / speed.min() * (1 + tol.FMM_HOMOGENEOUS_P95_REL))
        eval_sources.append(np.repeat(source[None], len(ref), axis=0))
        eval_receivers.append(receivers)
        expected.append(ref)
        exact_times.append(analytic)
        homogeneous_fmm.append(homogeneous_reference)
        group.append(np.full(len(ref), i, dtype=np.int32))

        # Continuous collocation positions are disjoint from held-out grid nodes.
        # Include points near the source so the factored network learns local speed.
        cr = origin + scale * rng.uniform(0.02, 0.98, (1536, d))
        cr[:256] = source + scale * rng.uniform(-0.025, 0.025, (256, d))
        cr = quantized(cr)
        assert np.all(np.linalg.norm(cr - source, axis=1) > 0)
        colloc_sources.append(np.repeat(source[None], len(cr), axis=0))
        colloc_receivers.append(cr)
        summaries.append({"source": source.tolist(), "source_grid_index": index.tolist(),
                          "evaluation_pairs": len(ref),
                          "time_range_seconds": [float(ref.min()), float(ref.max())],
                          "homogeneous_fmm_median_relative_error": homogeneous_median,
                          "homogeneous_fmm_p95_relative_error": homogeneous_p95,
                          "untrained_baseline_median_relative_error": float(np.median(errors))})
    if name != "homogeneous_2d":
        # Confirm the dataset cannot be passed just by the homogeneous ansatz.
        assert min(baseline_errors) > tol.MEDIAN_TIME_REL
    cs, cr = np.concatenate(colloc_sources), np.concatenate(colloc_receivers)
    cv = quantized(velocity_model(cr, origin, scale, name))
    initial = initial_parameters(rng, d, width)
    initial_loss = numpy_loss(initial, cs, cr, cv, origin, scale, vref, width)
    np.savez_compressed(DEST / f"{name}.npz", initial_parameters=initial,
                        collocation_sources=cs, collocation_receivers=cr,
                        collocation_velocities=cv, sources=np.concatenate(eval_sources),
                        receivers=np.concatenate(eval_receivers), time=np.concatenate(expected),
                        source_id=np.concatenate(group), source_locations=source_locations,
                        homogeneous_time=np.concatenate(exact_times), origin=origin,
                        homogeneous_fmm_time=np.concatenate(homogeneous_fmm),
                        length_scale=scale, reference_velocity=vref, width=width,
                        steps=10000, learning_rate=0.001, initial_loss=initial_loss)
    return {"name": name, "dimension": d, "oracle_grid_shape": [size] * d,
            "axis_order": ["x", "z"] if d == 2 else ["x", "y", "z"],
            "grid_origin_metres": origin.tolist(), "grid_spacing_metres": spacing,
            "width": width, "initial_loss": float(initial_loss),
            "collocation_pairs": len(cr), "source_cases": summaries,
            "velocity_range_m_per_s": [float(speed.min()), float(speed.max())],
            "velocity_formula": "3000" if name == "homogeneous_2d" else
            "u=(x-origin)/2000; v=2600+1000*u[-1]-550*exp(-sum(((u-center)/0.22)^2)); "
            + ("center=[0.60,0.45]" if d == 2 else "center=[0.60,0.45,0.55]")}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    derivatives = [derivative_case(d, rng) for d in (2, 3)]
    cases = [travel_case(name, d, rng) for name, d in
             (("homogeneous_2d", 2), ("smooth_2d", 2), ("smooth_3d", 3))]
    files = {p.name: {"bytes": p.stat().st_size,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(DEST.glob("*.npz"))}
    assert sum(f["bytes"] for f in files.values()) < 20_000_000
    meta = {
        "project": "29-eikonet", "round": 1, "tag": "M", "paths": ["mlx"],
        "seed": SEED, "rng": "numpy.random.PCG64", "reference_dtype": "float64",
        "api_inputs": "Quantized to float32 before float64 reference evaluation; arrays saved in npz.",
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-fmm", "torch", "pytest", "mlx", "mlx-metal")}},
        "oracle": "scikit-fmm.travel_time, order=2, nonperiodic box, phi=1 except one phi=0 source node; T(source)=0",
        "query_sampling": "Multilinear interpolation at float32-quantized receivers; exact nodes in 2D, <0.1 mm displacement from nodes in 3D.",
        "dependency_24": "Project 24 had no judge when this judge was written. Directly using its named scikit-fmm oracle; upstream source/stencil convention requires reconciliation before lock.",
        "paper": "Smith, Azizzadenesheli, Ross (2020), EikoNet, arXiv:2004.00361; small factored tanh variant, not paper architecture replication.",
        "training": "Physics-only; 3 fixed sources per medium; 4608 supplied collocation pairs; width 32, 10000 updates max, initial lr 0.001; held-out receivers never passed to train.",
        "metric": "Pointwise abs(T-Tfmm)/Tfmm, median <=0.01 and proposed p95 <=0.05 separately per source; positive times >300 m away.",
        "tolerances": {key: {"value": getattr(tol, key), "reason": reason}
                       for key, reason in tol.REASONS.items()},
        "derivatives": derivatives, "travel_cases": cases, "files": files,
        "limitations": ["Synthetic smooth isotropic media only; no discontinuous salt or public model.",
                        "Only three source locations per medium; no claim of unseen-source generalization.",
                        "No PINN was trained in building this judge; optimizer feasibility is established by the implementation, not the judge.",
                        "FMM is a discretized oracle; same-grid homogeneous 2D/3D error is audited; no 1e-4 continuum claim."]}
    (DEST / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"derivatives": derivatives, "travel_cases": cases,
                      "golden_bytes": sum(f["bytes"] for f in files.values())}, indent=2))


if __name__ == "__main__":
    main()
