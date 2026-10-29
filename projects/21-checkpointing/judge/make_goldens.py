"""Float64 full-storage acoustic gradient oracle. No implementation imports.

Run with the repository's pinned Python environment from this project directory. Deepwave's
eager PyTorch operator retains the entire trajectory; no checkpoint helper
is used. Its autodiff gradient is independently checked against Deepwave's
compiled discrete adjoint and centered velocity finite differences.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import warnings
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

import deepwave
import numpy as np
import torch
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "judge" / "golden"
SEED = 210021
spec = importlib.util.spec_from_file_location(
    "checkpoint_tolerances", ROOT / "tests" / "tolerances.py"
)
tol_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tol_module)
TOL = tol_module.TOLERANCES
FD_STEPS = np.asarray(tol_module.FD_STEPS_M_PER_S, dtype=np.float64)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Exactly the frozen harness's maximum-norm relative error."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b))) / scale if scale else float(np.max(np.abs(a)))


@lru_cache(None)
def replay_work(n: int, slots: int, reaches_end: bool) -> tuple[int, int]:
    """Exact minimum recursive-split work under the api.py counting rules.
    slots excludes the interval's fixed initial restart.

    On the first descent, reach the interval end for the original forward
    pass. On later left-subtree replays only reach its last INPUT state.
    A split k retains its restart while reversing the suffix with slots-1;
    release it and reverse the prefix with all slots restored. Searching all
    k gives an exact minimum, with ties broken toward the smallest k. There
    is no scheduling slack: implementations must use these base cases and the
    initial reach-end / subsequent last-input convention, not an unadapted
    textbook Revolve count.
    """
    if n == 1:
        return int(reaches_end), 0
    if slots == 0:
        return n * (n - 1) // 2 + (n if reaches_end else 0), 0
    if slots >= n - 1:
        return n if reaches_end else n - 1, 0
    candidates = [
        (k + replay_work(n - k, slots - 1, reaches_end)[0]
         + replay_work(k, slots, False)[0], k)
        for k in range(1, n)
    ]
    return min(candidates)


def reference_schedule(nt: int, budget: int) -> np.ndarray:
    """Construct an optimal witness for the exact integer work limits."""
    events = []
    position = 0

    def event(op: int, time: int, slot: int = -1) -> None:
        nonlocal position
        if op == 0 and position == time:
            return
        events.append([op, time, slot])
        if op in (0, 2):
            position = time

    def reverse(start: int, n: int, available: list[int], base: int,
                reaches_end: bool) -> None:
        slots = len(available)
        if slots == 0:
            if reaches_end:
                event(0, start + n)
            for t in range(start + n - 1, start - 1, -1):
                event(2, start, base)
                event(0, t)
                event(4, t)
        elif slots >= n - 1:
            used = []
            for offset in range(1, n):
                slot = available[offset - 1]
                event(0, start + offset)
                event(1, start + offset, slot)
                used.append((start + offset, slot))
            if reaches_end:
                event(0, start + n)
            for t, slot in reversed(used):
                event(2, t, slot)
                event(4, t)
                event(3, t, slot)
            event(2, start, base)
            event(4, start)
        else:
            split = replay_work(n, slots, reaches_end)[1]
            slot = available[0]
            event(0, start + split)
            event(1, start + split, slot)
            reverse(start + split, n - split, available[1:], slot, reaches_end)
            event(3, start + split, slot)
            event(2, start, base)
            reverse(start, split, available, base, False)

    reverse(0, nt, list(range(budget)), -1, True)
    return np.asarray(events, dtype=np.int64)


def audit_schedule(events: np.ndarray, nt: int, budget: int) -> tuple[int, int]:
    """Judge sanity check for the optimal witness; tests independently
    replay implementation logs rather than trusting these saved witnesses.
    """
    live = {}
    position, work, peak, next_reverse = 0, 0, 0, nt - 1
    reached_end = False
    for op, t, slot in events:
        if op == 0:
            assert slot == -1 and position < t <= nt
            work += t - position
            position = int(t)
            reached_end |= t == nt
        elif op == 1:
            assert t == position and 0 < t < nt
            assert 0 <= slot < budget and slot not in live
            live[int(slot)] = int(t)
            peak = max(peak, len(live))
        elif op == 2:
            assert reached_end
            assert (slot == -1 and t == 0) or live.get(int(slot)) == t
            position = int(t)
        elif op == 3:
            assert live.pop(int(slot)) == t
        elif op == 4:
            assert reached_end and slot == -1 and position == t == next_reverse
            next_reverse -= 1
        else:
            raise AssertionError(op)
    assert not live and next_reverse == -1
    assert work == replay_work(nt, budget, True)[0]
    return int(work), peak


def propagate(case: dict, velocity: np.ndarray, dtype: torch.dtype,
              backend: str = "eager", need_gradient: bool = True) -> dict:
    """Only reference-library call site; full-storage, fixed discretisation."""
    v = torch.tensor(velocity, dtype=dtype, requires_grad=need_gradient)
    src = torch.tensor(case["source_amplitudes"], dtype=dtype)
    observed = torch.tensor(case["observed"], dtype=dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*zero.*width.*")
        result = deepwave.scalar(
            v, case["grid_spacing"].tolist(), float(case["dt"]),
            source_amplitudes=src,
            source_locations=torch.tensor(case["source_locations"], dtype=torch.int64),
            receiver_locations=torch.tensor(case["receiver_locations"], dtype=torch.int64),
            accuracy=int(case["accuracy"]), pml_width=case["pml_width"].tolist(),
            pml_freq=float(case["pml_freq"]), max_vel=float(case["max_vel"]),
            model_gradient_sampling_interval=1, checkpoint_segments=None,
            storage_mode="device", storage_compression=False,
            freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False,
            backend=backend,
        )
    assert result.info.step_ratio == 1, "Oracle must never silently resample time"
    predicted = result.receiver_amplitudes
    objective = 0.5 * ((predicted - observed) ** 2).sum()
    answer = {"objective": objective.detach().numpy(),
              "predicted": predicted.detach().numpy()}
    if need_gradient:
        answer["gradient"] = torch.autograd.grad(objective, v)[0].detach().numpy()
    return answer


def make_case(rng: np.random.Generator, name: str, nz: int, nx: int,
              nt: int, shots: int, accuracy: int, pml: list[int]) -> dict:
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    velocity = (1750 + 8 * z + 30 * np.sin(x / 3)
                + rng.uniform(-15, 15, (nz, nx))).astype(np.float32)
    truth = (velocity + 70 * np.exp(-((z - nz / 2) ** 2 + (x - nx / 2) ** 2) / 20)
             - 35 * (x > nx // 2)).astype(np.float32)
    # Two sources per shot, asymmetric signs/amplitudes, shared physical model.
    sloc = np.array([[[2 + s, 2], [nz // 2, nx - 3 - s]]
                     for s in range(shots)], dtype=np.int64)
    rloc = np.array([[[2, 2 + s], [3, nx // 2], [nz - 3, nx - 3], [1, nx - 2]]
                     for s in range(shots)], dtype=np.int64)
    dt = 0.001
    t = np.arange(nt, dtype=np.float64) * dt
    src = np.empty((shots, 2, nt), dtype=np.float32)
    for s in range(shots):
        for k in range(2):
            phase = np.pi * (25 + 3 * k) * (t - 0.020 - 0.003 * s - 0.002 * k)
            src[s, k] = (0.03 * (1 + 0.2 * s) * (-0.7 if k else 1)
                         * (1 - 2 * phase**2) * np.exp(-phase**2))
    case = {
        "velocity": velocity, "source_amplitudes": src,
        "source_locations": sloc, "receiver_locations": rloc,
        "observed": np.zeros((shots, 4, nt), dtype=np.float32),
        "grid_spacing": np.array([8.0, 10.0], dtype=np.float64),
        "dt": np.array(dt), "accuracy": np.array(accuracy, dtype=np.int64),
        "pml_width": np.array(pml, dtype=np.int64), "pml_freq": np.array(25.0),
        "max_vel": np.array(2600.0), "truth_velocity": truth,
    }
    data = propagate(case, truth, torch.float64, need_gradient=False)["predicted"]
    case["observed"] = data.astype(np.float32)
    if name == "zero_source":
        # Nonzero fixed observations with exactly zero illumination: positive
        # loss but a mathematically zero velocity gradient, not 0/0 rel error.
        case["source_amplitudes"] = np.zeros_like(src)
    return case


def make_memory_case(rng: np.random.Generator) -> dict:
    """Saved inputs only: real MLX peak memory, not a float64 gradient judge.

    A full history contains at least one distinct physical-grid wavefield
    per time step (84 MiB), or ~574 MiB under dense six-field restart
    accounting. Seven dense restart slots occupy only ~10.5 MiB. The large
    separation makes working memory a small enough fraction to measure.
    Do not evaluate a full-storage float64 gradient on this fixture.
    """
    nz, nx, nt = 224, 256, 385
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    velocity = (1750 + 0.4 * z + 30 * np.sin(x / 13)
                + rng.uniform(-15, 15, (nz, nx))).astype(np.float32)
    t = np.arange(nt, dtype=np.float64) * 0.001
    phase = np.pi * 25 * (t - 0.04)
    source = (0.03 * (1 - 2 * phase**2) * np.exp(-phase**2)).astype(np.float32)
    pml = [8, 8, 8, 8]
    state_bytes = 6 * (nz + sum(pml[:2])) * (nx + sum(pml[2:])) * 4
    budgets = [7, nt - 1]
    case = {
        "velocity": velocity, "source_amplitudes": source[None, None, :],
        "source_locations": np.array([[[nz // 2, nx // 2]]], dtype=np.int64),
        "receiver_locations": np.array([[[nz // 2, nx // 2],
                                          [nz // 2 + 3, nx // 2 + 5],
                                          [nz // 2 - 7, nx // 2 - 2]]], dtype=np.int64),
        "observed": np.zeros((1, 3, nt), dtype=np.float32),
        "grid_spacing": np.array([8.0, 10.0], dtype=np.float64),
        "dt": np.array(0.001), "accuracy": np.array(2, dtype=np.int64),
        "pml_width": np.array(pml, dtype=np.int64), "pml_freq": np.array(25.0),
        "max_vel": np.array(2600.0),
        "budgets": np.array(budgets, dtype=np.int64),
        "forward_step_limits": np.array(
            [replay_work(nt, b, True)[0] for b in budgets], dtype=np.int64),
        "state_bytes": np.array(state_bytes, dtype=np.int64),
    }
    assert 1700 < velocity.min() < velocity.max() < 1900 < float(case["max_vel"])
    assert np.count_nonzero(source) > 0
    assert float(case["dt"]) * float(case["max_vel"]) / 8 < 0.5
    for budget, cost in zip(budgets, case["forward_step_limits"]):
        assert audit_schedule(reference_schedule(nt, budget), nt, budget)[0] == cost
    return case


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    pinned_versions = {}
    for line in (ROOT / "judge" / "requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            package, version = line.split("==")
            actual = importlib.metadata.version(package)
            assert actual == version, f"{package}: need {version}, found {actual}"
            pinned_versions[package] = actual
    # Freeze the installed dependency closure once; subsequent regenerations
    # check the committed lock rather than silently adopting newer packages.
    lock_path = ROOT / "judge" / "requirements.lock"
    if not lock_path.exists():
        resolved, pending = {}, list(pinned_versions)
        while pending:
            package = pending.pop()
            package = package.lower().replace("_", "-")
            if package in resolved:
                continue
            resolved[package] = importlib.metadata.version(package)
            for dependency in importlib.metadata.requires(package) or []:
                requirement = Requirement(dependency)
                if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                    pending.append(requirement.name)
        lock_path.write_text(
            "# Installed dependency closure, Python 3.12 / macOS; no optional extras.\n"
            + "".join(f"{package}=={version}\n" for package, version in sorted(resolved.items()))
        )
    locked_dependencies = {}
    for line in lock_path.read_text().splitlines():
        if line and not line.startswith("#"):
            package, version = line.split("==")
            assert importlib.metadata.version(package) == version, f"Dependency drift: {package}"
            locked_dependencies[package] = version
    deepwave_dir = Path(deepwave.__file__).resolve().parent
    deepwave_sources = [deepwave_dir / name for name in
                        ("scalar.py", "_common.py", "_engine.py", "_boundaries.py")]
    deepwave_sources += sorted(deepwave_dir.glob("libdeepwave_*"))
    rng = np.random.default_rng(SEED)
    configs = [
        ("small", 9, 11, 17, 1, 2, [0, 0, 0, 0]),
        ("pml_multishot", 13, 15, 67, 2, 4, [3, 4, 2, 3]),
        ("long", 15, 17, 127, 1, 2, [4, 3, 3, 4]),
        ("zero_source", 9, 11, 19, 1, 2, [2, 2, 2, 2]),
    ]
    metadata = {
        "project": "21-checkpointing", "seed": SEED,
        "oracle": "Deepwave scalar eager float64 full-trajectory PyTorch autodiff",
        "versions": pinned_versions,
        "locked_dependencies": locked_dependencies,
        "deepwave_build_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in deepwave_sources
        },
        "python": platform.python_version(),
        "precision": "Inputs rounded to float32 FIRST; oracle promotes these exact values to float64.",
        "operator": {"temporal_order": 2, "spatial_orders": [2, 4],
                     "boundary": "Deepwave default PML, bare zero-width faces when width=0",
                     "max_vel": 2600.0, "pml_freq": 25.0,
                     "step_ratio": 1, "model_gradient_sampling_interval": 1,
                     "checkpoint_segments": None, "backend": "eager",
                     "storage_mode": "device", "storage_compression": False,
                     "survey_pad": None, "location_interpolation": "integer grid points",
                     "objective": "0.5 * sum((predicted-observed)**2)",
                     "source": "u_tt=v^2*(laplacian(u)-f); -v(source)^2*dt^2*f[t] injection",
                     "receiver_sampling": "u[t], BEFORE injection/step t; initial data sample is zero"},
        "tolerances": TOL,
        "tolerance_reasons": {
            "checkpoint_gradient": "Explicit project 21 bound, vs SAME path's full-storage gradient.",
            "checkpoint_forward": "Replaying the same deterministic forward operator must preserve its outputs.",
            "reference": "Project-wide float32-vs-float64 parity at the SAME discrete operator; not continuum error.",
            "cross_path": "Independent float32 implementations may accumulate in different orders.",
            "finite_difference": "Project-wide centered finite differences; directional steps 0.625, 1.25, 2 m/s.",
            "oracle_finite_difference": "Judge's float64 FD truncation must be negligible vs 1e-3 allowance.",
            "oracle_adjoint": "Float64 eager autodiff checked against independently compiled Deepwave adjoint.",
            "loss_from_receivers": "Float32 reduction rounding, sum loss must match returned samples.",
            "zero_absolute": "No illumination means exactly zero field and velocity sensitivity."},
        "finite_difference": {"nominal_steps_m_per_s": FD_STEPS.tolist(),
                              "directions": "source cell, sign-aligned off-source field, and fixed-seed dense field",
                              "formula": "(J(v_plus)-J(v_minus))/2; compare grad dot ((v_plus-v_minus)/2)",
                              "note": "Actual float32-rounded plus/minus arrays are stored, not regenerated by tests."},
        "work_bound": {
            "meaning": "Exact minimum recursive-split cost; optimality under this API counting convention is required.",
            "definition": "F(1,c,e)=e; F(n,0,e)=n*(n-1)//2+e*n; F(n,c,e)=n-1+e if c>=n-1; otherwise min_{1<=k<n} [k+F(n-k,c-1,e)+F(k,c,0)].",
            "required_limit": "F(nt,budget,1); baseline -1 requires nt steps.",
            "convention": "First descent reaches nt; later left/prefix replays stop at their last input state.",
            "witness_tie_break": "Smallest k minimizing THIS recurrence; a classical Revolve planner may need its largest first optimal split when adapted.",
        },
        "memory": "Logical dense restart payload only, 6*nshot*nz_padded*nx_padded*4 bytes per checkpoint.",
        "cases": {},
        "collected_tests": 82,
        "test_groups": {"full_storage": 8, "checkpoint_parity_and_audits": 40,
                        "finite_differences": 6, "cross_path": 24,
                        "implementation_curve_plots": 2, "golden_integrity": 1,
                        "mlx_measured_memory": 1},
    }
    for config in configs:
        name, nz, nx, nt, shots, accuracy, pml = config
        case = make_case(rng, *config)
        ref = propagate(case, case["velocity"], torch.float64)
        native = propagate(case, case["velocity"], torch.float64, backend="compiled")
        single = propagate(case, case["velocity"], torch.float32)
        for field in ("predicted", "objective", "gradient"):
            assert relative(native[field], ref[field]) <= TOL["oracle_adjoint"], (name, field)
            assert relative(single[field], ref[field]) <= TOL["reference"], (name, field)
            assert np.all(np.isfinite(ref[field]))
            case[f"reference_{field}"] = ref[field]
        assert np.all(ref["predicted"][..., 0] == 0)
        assert float(ref["objective"]) > 0
        doubled = dict(case)
        doubled["source_amplitudes"] = 2 * case["source_amplitudes"]
        linearity = propagate(doubled, case["velocity"], torch.float64, need_gradient=False)
        assert relative(linearity["predicted"], 2 * ref["predicted"]) <= TOL["oracle_adjoint"]
        if name == "zero_source":
            assert np.count_nonzero(ref["gradient"]) == 0
            assert np.count_nonzero(ref["predicted"]) == 0
        else:
            assert np.max(np.abs(ref["gradient"])) > 0
            assert np.max(np.abs(ref["predicted"])) > 0
        budgets = [0, 1, 3, 7, nt - 1]
        witnesses, costs, peaks = [], [], []
        for budget in budgets:
            schedule = reference_schedule(nt, budget)
            work, peak = audit_schedule(schedule, nt, budget)
            witnesses.append(schedule)
            costs.append(work)
            peaks.append(peak)
        assert all(a >= b for a, b in pairwise(costs))
        assert costs[0] == nt * (nt + 1) // 2 and costs[-1] == nt
        case["budgets"] = np.array(budgets, dtype=np.int64)
        case["forward_step_limits"] = np.array(costs, dtype=np.int64)
        state_bytes = 6 * shots * (nz + sum(pml[:2])) * (nx + sum(pml[2:])) * 4
        case["state_bytes"] = np.array(state_bytes, dtype=np.int64)
        for b, witness in zip(budgets, witnesses):
            case[f"witness_budget_{b}"] = witness
        fd_error, fd32_errors = None, None
        if name == "small":
            directions = np.zeros((3, nz, nx), dtype=np.float64)
            directions[0, 2, 2] = 1  # source cell: catches missing injection derivative
            off_source = np.abs(ref["gradient"]).copy()
            for point in case["source_locations"].reshape(-1, 2):
                off_source[tuple(point)] = 0
            directions[1] = np.sign(ref["gradient"]) * (off_source > 0)
            directions[2] = rng.uniform(-1, 1, (nz, nx))
            plus = (case["velocity"][None] + FD_STEPS[:, None, None] * directions).astype(np.float32)
            minus = (case["velocity"][None] - FD_STEPS[:, None, None] * directions).astype(np.float32)
            displacements = (plus.astype(np.float64) - minus.astype(np.float64)) / 2
            fd_losses, fd_losses32 = [], []
            for vp, vm in zip(plus, minus):
                jp = propagate(case, vp, torch.float64, need_gradient=False)["objective"]
                jm = propagate(case, vm, torch.float64, need_gradient=False)["objective"]
                fd_losses.append([float(jp), float(jm)])
                jp32 = propagate(case, vp, torch.float32, need_gradient=False)["objective"]
                jm32 = propagate(case, vm, torch.float32, need_gradient=False)["objective"]
                fd_losses32.append([float(jp32), float(jm32)])
            fd_losses = np.asarray(fd_losses)
            differences = (fd_losses[:, 0] - fd_losses[:, 1]) / 2
            tangent = np.einsum("kij,ij->k", displacements, ref["gradient"])
            fd_error = max(relative(a, b) for a, b in zip(differences, tangent))
            print(f"{name} FD: differences={differences}, tangents={tangent}, "
                  f"per_direction={[relative(a, b) for a, b in zip(differences, tangent)]}")
            assert fd_error <= TOL["oracle_finite_difference"], fd_error
            fd_losses32 = np.asarray(fd_losses32)
            differences32 = (fd_losses32[:, 0] - fd_losses32[:, 1]) / 2
            tangent32 = np.einsum("kij,ij->k", displacements, single["gradient"])
            fd32_errors = [relative(a, b) for a, b in zip(differences32, tangent32)]
            assert max(fd32_errors) <= TOL["finite_difference"], fd32_errors
            print(f"{name} float32 FD errors={fd32_errors}")
            case.update(fd_plus_velocity=plus, fd_minus_velocity=minus,
                        fd_displacements=displacements,
                        fd_objectives=fd_losses, fd_loss_differences=differences,
                        fd_steps=FD_STEPS)
        np.savez_compressed(OUT / f"{name}.npz", **case)
        stats = {
            "shape": [nz, nx], "nt": nt, "shots": shots, "sources_per_shot": 2,
            "receivers_per_shot": 4, "accuracy": accuracy, "pml_width": pml,
            "velocity_range_m_per_s": [float(case["velocity"].min()), float(case["velocity"].max())],
            "source_range_U_per_m2": [float(case["source_amplitudes"].min()), float(case["source_amplitudes"].max())],
            "predicted_range_U": [float(ref["predicted"].min()), float(ref["predicted"].max())],
            "objective_U2": float(ref["objective"]),
            "gradient_range_U2_per_m_per_s": [float(ref["gradient"].min()), float(ref["gradient"].max())],
            "float32_reference_errors": {k: relative(single[k], ref[k]) for k in ref},
            "compiled_adjoint_error": relative(native["gradient"], ref["gradient"]),
            "fd_error": fd_error, "float32_fd_errors": fd32_errors, "budgets": budgets,
            "forward_step_limits": costs, "state_bytes": state_bytes,
        }
        metadata["cases"][name] = stats
        print(f"{name}: J={stats['objective_U2']:.9g}, "
              f"max|data|={np.max(np.abs(ref['predicted'])):.6g}, "
              f"max|grad|={np.max(np.abs(ref['gradient'])):.6g}, work={costs}")
        if name == "long":
            fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
            ax.plot(np.array(peaks) * state_bytes / 1024, np.array(costs) / nt, "o-")
            ax.set(xlabel="Logical saved restart payload (KiB)",
                   ylabel="Forward work / original forward pass",
                   title="Optimal recursive work (not measured implementation memory)")
            ax.grid(alpha=0.25)
            fig.savefig(OUT / "schedule_bound.png", dpi=140)
            plt.close(fig)
    memory_case = make_memory_case(rng)
    np.savez_compressed(OUT / "memory.npz", **memory_case)
    nt = memory_case["source_amplitudes"].shape[-1]
    payload_ratio = int(memory_case["budgets"][0]) / (nt - 1)
    peak_ratio_limit = (TOL["mlx_memory_workspace_fraction"]
                        + TOL["mlx_memory_payload_factor"] * payload_ratio)
    assert peak_ratio_limit < 1, "Allocator gate must reject unchanged full storage"
    metadata["tolerance_reasons"].update({
        "mlx_memory_workspace_fraction": "Allow 40% of full-storage peak for working arrays, adjoints, VJP and stencil temporaries.",
        "mlx_memory_payload_factor": "Allow 6x logical payload ratio for compact PML and shared adjacent wavefield levels; normalized restarts contain six fields.",
    })
    metadata["memory_fixture"] = {
        "name": "memory", "purpose": "Saved inputs for independent in-process MLX allocator peak comparison; no float64 gradient reference.",
        "shape": list(memory_case["velocity"].shape), "nt": nt, "shots": 1,
        "accuracy": 2, "pml_width": memory_case["pml_width"].tolist(),
        "velocity_range_m_per_s": [float(memory_case["velocity"].min()), float(memory_case["velocity"].max())],
        "observed": "Zero target; nonzero saved Ricker source, so field, objective and gradient must be nonzero.",
        "budgets": memory_case["budgets"].tolist(),
        "forward_step_limits": memory_case["forward_step_limits"].tolist(),
        "state_bytes": int(memory_case["state_bytes"]),
        "dense_full_restart_bytes": (nt - 1) * int(memory_case["state_bytes"]),
        "logical_checkpoint_to_full_ratio": payload_ratio,
        "peak_ratio_formula": "mlx_memory_workspace_fraction + mlx_memory_payload_factor * budget/(nt-1)",
        "peak_ratio_limit": peak_ratio_limit,
        "full_peak_floor_bytes": (nt - 1) * memory_case["velocity"].nbytes,
        "measurement": "Synchronize, GC, clear allocator cache, reset peak, uncached harness call, synchronize, read absolute MLX allocator peak. Full first, then budget 7; keep only NumPy outputs.",
        "scope": "MLX allocations only. Metal subprocess allocator/footprint measurement still required in bench and code review.",
    }
    print(f"memory: input-only grid={memory_case['velocity'].shape}, nt={nt}, "
          f"dense full={metadata['memory_fixture']['dense_full_restart_bytes']} bytes, "
          f"peak ratio limit={peak_ratio_limit:.6f}")
    metadata["files"] = {
        p.name: {"bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(OUT.glob("*.npz"))
    }
    assert sum(v["bytes"] for v in metadata["files"].values()) < 20_000_000
    (OUT / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(configs)} float64 goldens + 1 saved-input memory fixture; total NPZ bytes "
          f"{sum(v['bytes'] for v in metadata['files'].values())}")


if __name__ == "__main__":
    # Keep Matplotlib's font/config cache inside this project's authorized
    # write area and remove it on exit. No user-global cache writes needed.
    with TemporaryDirectory(prefix=".mpl-", dir=ROOT / "judge") as mpl_config:
        os.environ["MPLCONFIGDIR"] = mpl_config
        main()
