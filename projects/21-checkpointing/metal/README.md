# Rust and Metal checkpointing

The CLI implements `misfit_and_grad` as specified in `tests/api.py`. Rust
constructs fixed profiles, maps acquisition points, plans replay, and manages
GPU resources. Spatial operators, time updates, reverse updates, reductions,
state copies, and gradient folding execute in `kernels/checkpoint.metal`.
There is no trajectory on the host and no read of any golden fixture.

The physical model is extended by edge replication, then surrounded by
`accuracy/2` constant zero stencil guards. The guards are not part of the
normalized restart payload. Each restart contains two dense wavefields and
one dense `float4` containing `(psi_z,psi_x,zeta_z,zeta_x)`, including all
shots. All fields are saved even if a PML width is zero.

For each axis q, with centered operators D1 and D2, fixed profiles a and b,
and c=D1(b), the forward equations are

```text
Tq       = (1+bq)*D2q(u) + cq*D1q(u) + D1q(aq*psi_q)
Sq       = (1+bq)*Tq + aq*zeta_q
psi_q'   = bq*D1q(u) + aq*psi_q
zeta_q'  = bq*Tq + aq*zeta_q
C        = v^2*dt^2
u'       = 2*u - previous + C*(Sz+Sx) - C*f
previous'= u
```

Receivers sample `u` before the step, so sample zero is exactly zero.
The objective is half the sum of squared residuals, without normalization.
Order 2 uses D1 coefficients `[1/2]` and D2 `[1]`; order 4 uses
`[2/3,-1/12]` and `[4/3,-1/12]`, respectively. The two spacings are
independent. The source has no cell area factor.

With outgoing cotangents l=`bar(u')` and m=`bar(memory')`, the local reverse
kernel forms

```text
ls          = C*l
bar(Tq)     = (1+bq)*ls + bq*bar(zeta_q')
bar(D1q u)  = cq*bar(Tq) + bq*bar(psi_q')
grad_v     += 2*v*dt^2*(Sz+Sx-f)*l
```

The gather kernel applies D2 transpose (symmetric) and D1 transpose
(negative D1), with the profile factors on the correct side of the
operators. It also computes the old memory cotangents, `bar(previous)=-l`,
and `bar(u)=2*l+bar(previous')+spatial_terms+residual[t]`. Guard cotangents
are projected to zero before neighboring gathers. Extended velocity
gradients accumulate in descending time order and are folded to physical
cells using one thread per physical cell. Each thread visits all replicated
faces, corners, and shots in a fixed order, using compensated float32 sums.
There are no floating atomics.

`planner.rs` uses binomial marginal costs to solve exactly F(n,c,e) from the
API with the smallest minimizing split. Full-storage and zero-slot plans
are generated directly. Intermediate budgets need no cost tables: each
split uses a binary search over saturated binomial capacities. Planning
uses O(nt) metadata including the event log and O(nt log^2(nt)) worst-case
integer work; a heap stack avoids deep recursion near full capacity.
Its emitted events are executed in order. State copies
really occur at SAVE/RESTORE, DROP releases its GPU buffers, and the
returned audit is appended only after each successful operation. Only the
initial descent records receiver data. The first descent reaches `nt`;
subsequent prefix descents stop at their last input state. The adjoint
recomputes its local spatial terms directly from the current input restart,
without advancing or saving any forward state. Full storage uses this same
per-step adjoint and accumulation order.

Logical retained bytes use checked int64 multiplication, independently of
the uint32 limits on Metal array indices. Zero saved states report zero
bytes without computing an unused single-state payload.

At most c saved restarts plus two working states, cotangents, and fixed
workspace are live. No rhs history or differentiation tape is stored.
Command buffers are synchronous and bounded to 32 forward steps. The local
`device.rs` wrapper drains Objective-C autoreleases after every submission,
so completed command buffers cannot retain released checkpoint resources.
No trajectory, buffer, or compiled pipeline is cached between calls.

`bench/metal/measure.py` is a standalone benchmark driver; it is not run as part of the tests and had not been executed when this was written.
It generates independent synthetic inputs, measures each Metal subprocess
using macOS `/usr/bin/time -l`, checks replay parity, and saves raw counters,
peak resident memory, elapsed time, and a memory/recomputation plot. RSS
includes compilation/runtime overhead and unified GPU memory; it is a
process measurement, separate from the API's normalized logical bytes.
