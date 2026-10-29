# Acoustic forward solve and discrete adjoint

`src/main.rs` implements the frozen `misfit_and_grad` CLI. It validates shapes,
builds fixed float32 PML profiles and point-index lists, schedules Metal kernels,
and writes three float32 arrays. Propagation, objective reduction, reverse
propagation, and gradient folding run in `kernels/fwi.metal`, compiled with
`Gpu::compile` (fast math disabled). There are no additional dependencies.

The spatial and C-PML equations are those in `tests/DISCRETE_OPERATOR.md`.
Profiles use the supplied `max_vel`, independent of velocity. Both guard layers
are zero in every field. The derivative of the profile product samples `a*psi`
at neighboring nodes; the separate `c*Du` term is evaluated throughout the
active grid, including original-model nodes next to the PML.

The time recurrence uses a stored first difference to reduce float32
cancellation:

```text
d^k       = u^k - u^(k-1)
F^k       = Sz^k + Sx^k - f^k
C         = V^2 dt^2
d^(k+1)   = d^k + C F^k
u^(k+1)   = u^k + d^(k+1)
```

Eliminating `d` gives exactly the specified centered second-order recurrence;
`u^0=d^0=0` implements the zero initial conditions. Receiver gathering precedes
the update. The explicit `fma` in the first-difference update rounds its multiply
and add once. No extended-precision arithmetic or internal substeps are used.

The hand-written reverse pass transposes this factored sequence. For one step,
let `U`, `D`, `Pq`, and `Zq` be the incoming cotangents of the **new** wavefield,
first difference, psi memory, and zeta memory. For each axis q:

```text
l          = U + D
s          = C*l
tq         = (1+bq)*s + bq*Zq
pq         = cq*tq + bq*Pq
bar(u_old) = U + sum_q[Dqq^T((1+bq)*tq) + Dq^T(pq)] + receiver_residual/M
bar(d_old) = l
bar(psi_q) = aq*(Pq + Dq^T(tq))
bar(zeta_q)= aq*(Zq + s)
bar(V)    += (2*V*dt^2)*F^k*l
```

Here `Dq^T=-Dq` and `Dqq^T=Dqq`, with the guard projection applied to the
cotangents before and after the stencil gathers. `adjoint_local` forms `tq`
and `pq` and accumulates the direct velocity derivative.
`adjoint_gather` evaluates the transpose stencils and the old-memory branches.
It adds the normalized residual for the recorded **current** `u^k` only after
reversing the step. This timing also makes the final source sample contribute
zero gradient, since `u^nt` is not recorded. No continuous-adjoint approximation
or backward PML damping is substituted for this transpose.

Only `F^k` is saved at each time: the remaining state Jacobians are independent
of forward state. Saved forcing includes the source term, so the velocity
derivative includes source injection. The history occupies
`4*nt*shots*nzp*nxp` bytes. This implementation targets the locked small grids;
a full Marmousi benchmark would benefit from checkpointing or shot batching.

Grid-to-point lists sum coincident source injections and receiver cotangents
without atomics. The final kernel sums the full clipped-coordinate preimage
of each original model node across shots, including exterior PML edges and
corners. Guard velocities do not contribute. Objective and gradient-fold
reductions use compensated float32 sums. All output shapes and dtypes follow
the locked API, including a zero-dimensional float32 objective.
