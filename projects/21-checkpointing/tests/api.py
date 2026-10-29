"""Shared MLX / Rust+Metal interface; the judge does not implement this API."""

import numpy as np


def misfit_and_grad(
    velocity: np.ndarray,
    source_amplitudes: np.ndarray,
    source_locations: np.ndarray,
    receiver_locations: np.ndarray,
    observed: np.ndarray,
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
    pml_width: list[int],
    pml_freq: float,
    max_vel: float,
    checkpoint_budget: int,
) -> dict[str, np.ndarray]:
    """Return an acoustic L2 objective, velocity gradient and replay audit.

    The discrete physical operator is Deepwave ``scalar`` (version pinned in
    judge/golden/meta.json): second order in time, ``accuracy`` in space,
    zero initial wavefields and PML memory variables, full model (no survey
    crop), no time resampling, no compression or gradient subsampling.
    GPU arithmetic and all floating outputs are float32. MLX differentiates
    by autodiff with recomputation; Metal uses the discrete hand adjoint.

    Inputs
    ------
    velocity : float32 [nz, nx], m/s
        Positive cell wavespeed, shared by all shots. Index [0,0] is the
        physical grid origin; z increases down, x increases right.
    source_amplitudes : float32 [nshot, nsource, nt], U/m**2
        Point RHS f in u_tt = v**2 * (Laplacian(u) - f), following Deepwave's
        sign and scaling: each step injects -v(source)**2 * dt**2 * f[t].
        There is no cell-area normalization. U is an arbitrary scalar field
        unit. The velocity dependence of injection MUST be differentiated.
    source_locations : int64 [nshot, nsource, 2], cell indices [z,x]
        Zero-based indices in the physical model, unique within a shot.
    receiver_locations : int64 [nshot, nreceiver, 2], cell indices [z,x]
        Zero-based, unique within a shot. Record u[t] BEFORE step t's source
        injection, for t=0,...,nt-1; sample zero is therefore exactly zero.
    observed : float32 [nshot, nreceiver, nt], U
        Fixed data. Objective J = 0.5 * SUM((predicted-observed)**2), in U**2.
        No normalization by time, shot count, receiver count, dt or cell area.
    grid_spacing : list[float] of length 2, [dz,dx], m
    dt : positive Python float, s
        Chosen to give Deepwave internal step_ratio=1 for fixed max_vel.
    accuracy : Python int, 2 or 4 in the locked fixtures
        Centered spatial finite-difference order, exactly Deepwave's stencil.
    pml_width : list[int] of length 4, cells
        [z_low,z_high,x_low,x_high]. Extend velocity by edge replication.
        A zero width uses Deepwave's default bare zero-width reflecting edge
        (zero exterior stencil), NOT its optional named boundary conditions.
        All PML auxiliary fields must be restored along with u[t],u[t-1].
    pml_freq : Python float, Hz
        Fixed PML design frequency, held constant when differentiating v.
    max_vel : Python float, m/s
        Fixed CFL/PML design wavespeed, held constant when differentiating v.
    checkpoint_budget : Python int
        -1: full-storage gradient baseline, with every restart saved. Use
        the SAME per-step VJP/adjoint and descending gradient accumulation
        order as checkpointed modes; whole-loop autodiff can change the
        accumulation order enough to violate the 1e-6 parity requirement.
        Otherwise 0..nt-1:
        maximum LIVE saved forward restart states, excluding immutable zero
        state t=0 and O(1) working/adjoint fields. A restart contains BOTH
        wavefield levels and ALL four PML variables (even for zero-width
        PML). No hidden history, disk offload, lossy state compression or
        differentiation tape retaining the full trajectory is permitted.
        Release trajectory storage before returning; no history may be
        cached across calls. MLX allocator peaks are independently measured
        on a saved large fixture, in addition to the logical byte audit.

    Returns (all values NumPy arrays, including zero-dimensional scalars)
    ------------------------------------------------------------------
    objective : float32 [], U**2
    gradient : float32 [nz,nx], U**2/(m/s)
        dJ/dvelocity for the ORIGINAL physical grid, including accumulation
        of derivatives through the edge-replicated PML model and sources.
    predicted : float32 [nshot,nreceiver,nt], U
        Initial forward-pass receiver record. Replays do not append data.
    schedule : int64 [nevent,3], dimensionless [opcode,time,slot]
        An audit of ACTUAL executed operations. Current forward time starts
        at 0. Slots are 0..budget-1 (0..nt-2 for full storage). Opcodes:
        0 ADVANCE: advance from current time to strictly larger ``time``;
          slot=-1. Counts time-current time forward steps, each step advancing
          ALL shots together. No advance beyond nt. The initial run must
          reach nt before any REVERSE or RESTORE.
        1 SAVE: save the complete current state at ``time`` (1..nt-1) in a
          currently empty slot. State 0 is implicit; working fields excluded.
        2 RESTORE: restore ``time`` from ``slot``, or (time=0,slot=-1) for
          the immutable initial state. Does not consume the saved state.
        3 DROP: release a live slot whose saved time equals ``time``.
        4 REVERSE: apply exactly one discrete adjoint step at forward state
          ``time``; slot=-1. Must occur at current forward time, in descending
          order nt-1,...,0, exactly once each. It does not move forward time.
        All slots must be dropped by the end. This state-at-input convention
        permits a one-step VJP from the current restart, with O(1) workspace.
    forward_steps : int64 [], number of batched forward time steps
        Includes the original nt steps and every replay. Excludes adjoint
        work. Full storage must equal nt; zero slots equal nt*(nt+1)//2.
    peak_checkpoints : int64 [], number of simultaneously live saved states
        Must equal the maximum occupancy reconstructed from schedule.
    checkpoint_bytes : int64 [], bytes of logically retained restart payload
        Exactly peak_checkpoints * state_bytes. state_bytes is defined as
        6 * nshot * (nz+z_low+z_high) * (nx+x_low+x_high) * 4, i.e. six
        dense float32 fields WITHOUT constant zero stencil halos. This is a
        normalized payload count, not process RSS or allocator peak memory;
        compact PML slabs may use less actual memory. Inputs, data, model,
        gradients, working fields and schedule itself are excluded.

    With nonnegative budgets, forward_steps must not exceed F(nt,budget,1)
    below. This is the exact minimum recursive-split cost under this API's
    convention, so an optimal schedule under these counting rules is
    required. It is NOT the usual textbook count without adaptation.

    Define F(n,c,e) for n>=1, c>=0, e in {0,1}. The interval's initial
    restart is already available and does not count against its c slots;
    e=1 means this descent must reach the interval end, while e=0 means
    only its last INPUT state is needed. Base cases, in precedence order:

        F(1,c,e) = e
        F(n,0,e) = n*(n-1)//2 + e*n
        F(n,c,e) = n-1+e                         if c >= n-1

    Otherwise minimize over integer splits 1 <= k < n:

        F(n,c,e) = min_k [k + F(n-k,c-1,e) + F(k,c,0)].

    Advance k steps, save that split, reverse the suffix while retaining
    it, drop it, restore the interval's initial restart, then reverse the
    prefix using all c slots. Any minimizing split is valid for THIS
    recurrence. The saved witness breaks ties toward the smallest k.
    Adapting a classical Revolve planner can require choosing its largest
    optimal first split to attain this count. A separate loss-only forward
    pass adds counted work and exceeds the bound; collect loss/data on the
    original descent instead. The initial run MUST reach nt, although
    u[nt] is unrecorded and REVERSE nt-1 has zero velocity sensitivity.
    Later prefix replays stop at their last input state, never one beyond.
    """
    raise NotImplementedError
