"""Locked black-box parity, scheduling and gradient tests; no reference imports."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, golden_meta, rel_err

from .tolerances import TOLERANCES as TOL

CASES = ["small", "pml_multishot", "long", "zero_source"]
CURVE_INDICES = list(range(5))
FLOAT_FIELDS = ("objective", "gradient", "predicted")
INT_FIELDS = ("schedule", "forward_steps", "peak_checkpoints", "checkpoint_bytes")


@lru_cache(None)
def load_case(name):
    return golden(name)


def inputs(g, budget, velocity=None):
    """Marshal saved arrays and scalars exactly as the frozen harness expects."""
    return {
        "velocity": g["velocity"] if velocity is None else velocity,
        "source_amplitudes": g["source_amplitudes"],
        "source_locations": g["source_locations"],
        "receiver_locations": g["receiver_locations"],
        "observed": g["observed"],
        "grid_spacing": g["grid_spacing"].tolist(), "dt": float(g["dt"]),
        "accuracy": int(g["accuracy"]), "pml_width": g["pml_width"].tolist(),
        "pml_freq": float(g["pml_freq"]), "max_vel": float(g["max_vel"]),
        "checkpoint_budget": int(budget),
    }


def validate_schedule(out, g, budget):
    """Independently replay every event; reported counters alone cannot pass."""
    nt = g["source_amplitudes"].shape[-1]
    limit = nt - 1 if budget == -1 else budget
    current, next_adjoint, work, peak = 0, nt - 1, 0, 0
    live = {}
    first_forward_complete = False
    schedule = out["schedule"]
    assert schedule.ndim == 2 and schedule.shape[1] == 3
    assert schedule.shape[0] > 0
    for row_index, row in enumerate(schedule):
        op, time, slot = map(int, row)
        context = f"event {row_index}: {row.tolist()}, time={current}, live={live}"
        assert 0 <= time <= nt, context
        if op == 0:  # ADVANCE
            assert slot == -1 and current < time, context
            work += time - current
            current = time
            first_forward_complete |= current == nt
        elif op == 1:  # SAVE a complete restart
            assert current == time and 0 < time < nt, context
            assert 0 <= slot < limit and slot not in live, context
            live[slot] = time
            peak = max(peak, len(live))
            assert peak <= limit, context
        elif op == 2:  # RESTORE; initial state is free and immutable
            assert first_forward_complete, context
            assert (slot == -1 and time == 0) or (
                slot in live and live[slot] == time
            ), context
            current = time
        elif op == 3:  # DROP
            assert slot in live and live[slot] == time, context
            del live[slot]
        elif op == 4:  # REVERSE at the input state of the step
            assert first_forward_complete and slot == -1, context
            assert time == current == next_adjoint, context
            next_adjoint -= 1
        else:
            pytest.fail(f"unknown schedule opcode: {context}")
    assert next_adjoint == -1, "Every adjoint step must execute exactly once"
    assert not live, "Every checkpoint slot must be released"
    assert int(out["forward_steps"]) == work
    assert int(out["peak_checkpoints"]) == peak
    assert int(out["checkpoint_bytes"]) == peak * int(g["state_bytes"])
    if budget == -1 or budget == nt - 1:
        assert work == nt, "Full storage must perform no forward recomputation"
        assert peak == nt - 1
    else:
        index = int(np.flatnonzero(g["budgets"] == budget)[0])
        assert nt <= work <= int(g["forward_step_limits"][index])
    if budget == 0:
        assert peak == 0
        assert work == nt * (nt + 1) // 2


def validate_output(out, g, budget):
    assert set(out) == set(FLOAT_FIELDS + INT_FIELDS)
    shapes = {"objective": (), "gradient": g["velocity"].shape,
              "predicted": g["observed"].shape}
    for key, shape in shapes.items():
        value = out[key]
        assert isinstance(value, np.ndarray)
        assert value.shape == shape, (key, value.shape, shape)
        assert value.dtype == np.dtype("float32"), (key, value.dtype)
        assert np.all(np.isfinite(value)), key
    for key in INT_FIELDS:
        assert out[key].dtype == np.dtype("int64"), (key, out[key].dtype)
        if key != "schedule":
            assert out[key].shape == ()
            assert int(out[key]) >= 0
    assert float(out["objective"]) >= 0
    # Loss identity also rejects accidental means, hidden dt factors and a
    # receiver record concatenated from recomputed segments.
    residual = out["predicted"].astype(np.float64) - g["observed"].astype(np.float64)
    measured_loss = np.array(0.5 * np.sum(residual**2))
    assert rel_err(out["objective"], measured_loss) <= TOL["loss_from_receivers"]
    assert np.max(np.abs(out["predicted"][..., 0])) <= TOL["zero_absolute"]
    validate_schedule(out, g, budget)


@lru_cache(None)
def evaluate(path, name, budget, variant="base", direction=0):
    g = load_case(name)
    velocity = None if variant == "base" else g[f"fd_{variant}_velocity"][direction]
    out = call(path, "misfit_and_grad", **inputs(g, budget, velocity))
    validate_output(out, g, budget)
    return out


def compare_reference(out, g):
    for field in FLOAT_FIELDS:
        reference = g[f"reference_{field}"]
        if np.count_nonzero(reference) == 0:
            assert np.max(np.abs(out[field])) <= TOL["zero_absolute"], field
        else:
            assert rel_err(out[field], reference) <= TOL["reference"], field


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", CASES, ids=CASES)
def test_full_storage_matches_float64(path, name):
    compare_reference(evaluate(path, name, -1), load_case(name))


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("name", CASES, ids=CASES)
@pytest.mark.parametrize("budget_index", CURVE_INDICES,
                         ids=["slots0", "slots1", "slots3", "slots7", "slots-all"])
def test_checkpoint_matches_full_storage(path, name, budget_index):
    g = load_case(name)
    budget = int(g["budgets"][budget_index])
    out = evaluate(path, name, budget)
    full = evaluate(path, name, -1)
    # The project's 1e-6 criterion is deliberately independent of the looser
    # float32-vs-float64 discretisation reference comparison below.
    assert rel_err(out["gradient"], full["gradient"]) <= TOL["checkpoint_gradient"]
    for field in ("objective", "predicted"):
        assert rel_err(out[field], full[field]) <= TOL["checkpoint_forward"], field
    compare_reference(out, g)


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("budget", [-1, 0, 3], ids=["full", "slots0", "slots3"])
def test_centered_velocity_finite_differences(path, budget):
    g = load_case("small")
    base = evaluate(path, "small", budget)
    for k, displacement in enumerate(g["fd_displacements"]):
        plus = evaluate(path, "small", budget, "plus", k)
        minus = evaluate(path, "small", budget, "minus", k)
        difference = np.array((float(plus["objective"]) - float(minus["objective"])) / 2)
        tangent = np.array(np.sum(base["gradient"].astype(np.float64) * displacement))
        oracle_difference = g["fd_loss_differences"][k]
        assert oracle_difference != 0, "Judge selected a non-informative direction"
        # Per-direction relative errors; a strong source derivative cannot
        # hide an incorrect, weaker off-source propagation derivative.
        assert rel_err(difference, tangent) <= TOL["finite_difference"], k
        assert rel_err(tangent, oracle_difference) <= TOL["finite_difference"], k


@pytest.mark.parametrize("paths", [("mlx", "metal")], ids=["mlx-metal"])
@pytest.mark.parametrize("name", CASES, ids=CASES)
@pytest.mark.parametrize("budget_index", [-1] + CURVE_INDICES,
                         ids=["full", "slots0", "slots1", "slots3", "slots7", "slots-all"])
def test_mlx_metal_cross_check(paths, name, budget_index):
    g = load_case(name)
    budget = -1 if budget_index == -1 else int(g["budgets"][budget_index])
    left, right = (evaluate(path, name, budget) for path in paths)
    for field in FLOAT_FIELDS:
        assert rel_err(left[field], right[field]) <= TOL["cross_path"], field
    # Schedules may differ; each independently satisfies the same resource
    # limits and is validated by evaluate. Do not require identical plans.


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_memory_recompute_curve_is_plotted(path, tmp_path, record_property, monkeypatch):
    # Plot measured/reported implementation logs, not the oracle's example
    # curve. Keep test artifacts in pytest's temp area, never source roots.
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = load_case("long")
    outputs = [evaluate(path, "long", int(b)) for b in g["budgets"]]
    work = np.array([int(out["forward_steps"]) for out in outputs])
    payload = np.array([int(out["checkpoint_bytes"]) for out in outputs])
    peak = np.array([int(out["peak_checkpoints"]) for out in outputs])
    nt = g["source_amplitudes"].shape[-1]
    assert np.all(np.diff(work) <= 0)
    assert work[0] > work[1] > work[2] > work[-1] == nt
    assert payload[0] == 0 and payload[-1] == (nt - 1) * int(g["state_bytes"])
    assert np.all(np.diff(payload) >= 0)
    figure = tmp_path / f"memory_recompute_{path}.png"
    raw = tmp_path / f"memory_recompute_{path}.npz"
    np.savez(raw, budgets=g["budgets"], forward_steps=work,
             checkpoint_bytes=payload, peak_checkpoints=peak)
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.plot(payload / 1024, (work - nt) / nt, "o-", label=path)
    ax.set(xlabel="Logical saved restart payload (KiB)",
           ylabel="Extra forward steps / nt", title="Checkpoint memory versus recomputation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    record_property("memory_recompute_figure", str(figure))
    record_property("memory_recompute_data", str(raw))


@pytest.mark.parametrize("path", ["mlx"], ids=["mlx"])
def test_allocator_peak_reduces_with_checkpoints(path, tmp_path, record_property, monkeypatch):
    """Independent allocator evidence: a schedule log alone cannot pass.

    Use uncached calls on a larger saved fixture: a full-storage gradient
    plus invented checkpoint counters would have approximately the same
    measured peak in both calls and fail the ratio. Absolute peaks also
    include any GPU history improperly kept alive between calls.
    """
    import gc

    import mlx.core as mx

    # Warm imports and small per-step kernels before measuring. This is NOT
    # a warm large-fixture result, and neither measured call uses evaluate's
    # Python result cache. Harness conversion to NumPy forces lazy results.
    warm = call(path, "misfit_and_grad", **inputs(load_case("small"), 3))
    validate_output(warm, load_case("small"), 3)
    del warm
    g = load_case("memory")
    nt = g["source_amplitudes"].shape[-1]
    budget = int(g["budgets"][0])
    outputs, peaks = [], []
    for slots in (-1, budget):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()  # cached unused allocations are not trajectory data
        mx.reset_peak_memory()
        out = call(path, "misfit_and_grad", **inputs(g, slots))
        mx.synchronize()
        peaks.append(int(mx.get_peak_memory()))
        validate_output(out, g, slots)
        for field in FLOAT_FIELDS:
            assert np.max(np.abs(out[field])) > TOL["zero_absolute"], field
        outputs.append(out)  # NumPy only; cannot retain an MLX trajectory
    full, checkpointed = outputs
    logical_ratio = int(checkpointed["checkpoint_bytes"]) / int(full["checkpoint_bytes"])
    assert logical_ratio <= budget / (nt - 1)
    limit = (TOL["mlx_memory_workspace_fraction"]
             + TOL["mlx_memory_payload_factor"] * logical_ratio)
    # At least one distinct physical wavefield per time sample must be
    # present in the full-storage baseline. This also rejects a CPU-only
    # implementation or a zero/missing GPU allocator measurement. Adjacent
    # restart levels may share storage; no six-field lower bound is used.
    full_floor = (nt - 1) * g["velocity"].nbytes
    assert peaks[0] >= full_floor, (peaks, full_floor)
    assert peaks[1] > 0
    assert peaks[1] / peaks[0] <= limit < 1, (peaks, logical_ratio, limit)
    assert rel_err(checkpointed["gradient"], full["gradient"]) <= TOL["checkpoint_gradient"]
    for field in ("objective", "predicted"):
        assert rel_err(checkpointed[field], full[field]) <= TOL["checkpoint_forward"], field

    # In addition to the logical curve, preserve a real allocator/work
    # figure and raw measurements. The benchmark wave publishes figure.png.
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work = np.array([int(out["forward_steps"]) for out in outputs])
    raw = tmp_path / "mlx_allocator_memory_recompute.npz"
    figure = tmp_path / "mlx_allocator_memory_recompute.png"
    np.savez(raw, budgets=np.array([-1, budget]), peak_memory_bytes=np.array(peaks),
             forward_steps=work, peak_ratio_limit=np.array(limit),
             checkpoint_bytes=np.array([int(out["checkpoint_bytes"]) for out in outputs]))
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.plot(np.asarray(peaks) / 2**20, (work - nt) / nt, "o-")
    for peak, extra, label in zip(peaks, (work - nt) / nt, ("full", f"slots={budget}")):
        ax.annotate(label, (peak / 2**20, extra))
    ax.set(xlabel="Measured MLX allocator peak (MiB)", ylabel="Extra forward steps / nt",
           title="Measured MLX memory versus recomputation")
    ax.grid(alpha=0.25)
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    record_property("mlx_measured_peak_bytes", peaks)
    record_property("mlx_measured_peak_ratio_limit", limit)
    record_property("mlx_allocator_figure", str(figure))
    record_property("mlx_allocator_data", str(raw))


def test_oracle_manifest_and_float64_fd():
    """Detect altered/stale fixtures and verify the saved oracle self-checks."""
    meta = golden_meta()
    assert meta["tolerances"] == TOL
    assert set(meta["cases"]) == set(CASES)
    directory = Path(__file__).resolve().parents[1] / "judge" / "golden"
    assert set(meta["files"]) == {f"{name}.npz" for name in CASES + ["memory"]}
    for filename in meta["files"]:
        data = (directory / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["files"][filename]["sha256"]
        assert len(data) == meta["files"][filename]["bytes"]
    for name in CASES:
        g = load_case(name)
        for field in FLOAT_FIELDS:
            assert g[f"reference_{field}"].dtype == np.dtype("float64")
        assert meta["cases"][name]["compiled_adjoint_error"] <= TOL["oracle_adjoint"]
        for error in meta["cases"][name]["float32_reference_errors"].values():
            assert error <= TOL["reference"]
    g = load_case("small")
    tangent = np.einsum("kij,ij->k", g["fd_displacements"], g["reference_gradient"])
    for fd, dot in zip(g["fd_loss_differences"], tangent):
        assert fd != 0
        assert rel_err(np.array(fd), np.array(dot)) <= TOL["oracle_finite_difference"]
    memory = load_case("memory")
    info = meta["memory_fixture"]
    assert list(memory["velocity"].shape) == info["shape"]
    assert memory["velocity"].dtype == np.dtype("float32")
    assert info["full_peak_floor_bytes"] == (info["nt"] - 1) * memory["velocity"].nbytes
    ratio = int(memory["budgets"][0]) / (info["nt"] - 1)
    assert info["peak_ratio_limit"] == (TOL["mlx_memory_workspace_fraction"]
                                          + TOL["mlx_memory_payload_factor"] * ratio)
    assert info["peak_ratio_limit"] < 1
