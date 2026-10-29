# Project 50 discrete operator, version 1

This is the normative companion to `tests/api.py`. It defines the particular
discrete acoustic CFS-PML problem graded here, including its auxiliary state.
Equivalent pressure equations with differently scaled or staggered memories
do not define this API's six-channel state. No reference-library internals are
needed to implement this document. The judge uses pinned Deepwave CPU float64
and separately audits a NumPy expression of these equations against it.

The scalar two-memory C-PML formulation follows Pasalic and McGarry (2010),
*Convolutional perfectly matched layer for isotropic and anisotropic acoustic
wave equations*. The profile, node placement, discrete product rule and guard
closure below are explicit numerical choices. The stencil and PML law agree
with project 13's `tests/SCHEME.md` and project 16's
`tests/DISCRETE_OPERATOR.md`; this document additionally defines resumable state,
unequal axis spacings, periodic closure and the sandbox objective scale.

## Coordinates, guards and derivatives

Let `w=pml_width`, `g=accuracy/2`, and `h_d=grid_spacing[d]` for axes `d=y,x`.
The API's state includes `w` PML nodes on every side of the physical model,
but excludes guards. For calculation, add `g` guard nodes on every side, giving
axis lengths `N_d=n_d+2*w+2*g`. Physical node `j` is stored at `j+w+g`.
All state fields occupy collocated integer nodes, with no half-cell shifts.
The evolution region excludes only the guards: `g <= i_d < N_d-g` on BOTH
axes. On return remove the guards and retain the entire PML.

Extend the model by nearest-edge replication:
`V[i_y,i_x]=velocity[clip(i_y-w-g,0,ny-1),clip(i_x-w-g,0,nx-1)]`.
There is no spatial derivative of V in the wave equation. Time is centered
second order, with fixed `dt` and no CFL substeps or time resampling.

For an array F and either axis d use the following DISTINCT centered first
and second derivatives (do not compose D1 to make D2):

```text
D1_d F[i] = sum_(j=1..g) c_j*(F[i+j*e_d]-F[i-j*e_d])/h_d
D2_d F[i] = (d_0*F[i]+sum_(j=1..g) d_j*(F[i+j*e_d]+F[i-j*e_d]))/h_d^2
```

| accuracy | c_1,...,c_g | d_0,d_1,...,d_g |
|---|---|---|
| 2 | 1/2 | -2, 1 |
| 4 | 2/3, -1/12 | -5/2, 4/3, -1/12 |
| 6 | 3/4, -3/20, 1/60 | -49/18, 3/2, -3/20, 1/90 |
| 8 | 4/5, -1/5, 4/105, -1/280 | -205/72, 8/5, -1/5, 8/315, -1/560 |

## CFS-PML profiles and outer closure (`boundary='pml'`)

For each axis independently, with spacing h, allocated length N and w>0,
evaluate at every integer node `i=0,...,N-1`:

```text
ell = g+w                          # first physical model node
r   = N-1-g-w                      # last physical model node
qL  = clip((ell-i)/w,0,1)
qR  = clip((i-r)/w,0,1)
q   = max(qL,qR)
R   = 0.001                        # design reflection amplitude
p   = 2                            # damping polynomial power
kappa = 1                          # everywhere; no kappa memory/scaling
sigma0 = -(p+1)*max_velocity*ln(R)/(2*w*h)   # 1/s
sigma  = sigma0*(qL^p+qR^p)                 # 1/s
alpha  = pi*pml_freq*(1-q)                  # 1/s
a[i] = exp(-(sigma+alpha)*dt)       if q>0 else 0
b[i] = sigma/(sigma+alpha)*expm1(-(sigma+alpha)*dt)
```

If sigma+alpha=0, set b=0. At the physical interface q=0 BOTH a and b
are exactly zero. Distance is measured from the first/last physical NODE,
not from a face half a cell away. The outermost evolving PML node has q=1.
Extend a,b into the guards with q clipped at 1, then compute `c=D1(b)` with
the same stencil and spacing; set c=0 at the g guard nodes on each end.
Broadcast each one-dimensional profile across the other spatial axis.
Profiles are constant in time. `max_velocity` is the passed bound, never
recomputed from the painted model; all profiles stay fixed during FWI.

All pressure and memory GUARDS stay zero, including corners. No field is
evolved beyond the outer PML node: stencil reads there see these zero guards.
There is no reflection/image fill, free surface, or extra damping beyond
that edge. Profiles remain nonzero in guards so D1(b) near that edge has
the specified values; they multiply zero guard memories.

Force psi_y and zeta_y to zero in the entire slab `ell_y <= i_y <= r_y`,
including its overlap with x PML; force psi_x and zeta_x to zero in
`ell_x <= i_x <= r_x`, including its overlap with y PML. Do this on input
and after every update. Thus these memories are supported only in their own
axis's exterior PML. Pressure need not vanish anywhere in the evolution region.

## Stored state and update

At the start of local sample n, store exactly
`[u^n,u^(n-1),psi_y^n,psi_x^n,zeta_y^n,zeta_x^n]`. These superscripts define
the time levels, irrespective of any reference-library variable names.
Pressure has units AU, psi AU/m, zeta AU/m²; a,b are dimensionless.

First record receiver_amplitudes[:,n] by gathering u^n at physical integer
receiver nodes. Then compute the following over the entire evolution region,
using ONLY old fields on the right-hand sides:

```text
G_d^n = D1_d(u^n)
T_d^n = (1+b_d)*D2_d(u^n) + c_d*G_d^n + D1_d(a_d*psi_d^n)
S_d^n = (1+b_d)*T_d^n + a_d*zeta_d^n
psi_d^(n+1)  = b_d*G_d^n + a_d*psi_d^n
zeta_d^(n+1) = b_d*T_d^n + a_d*zeta_d^n
u^(n+1) = 2*u^n-u^(n-1) + V^2*dt^2*(S_y^n+S_x^n-f^n)
```

f^n is zero except at the physical source nodes, where it equals the supplied
source_amplitudes[:,n]. Sources are unique in this contract. Injection is
`-V^2*dt^2*f^n`, without cell-area normalization or memory injection.
`D1(a*psi)` is the derivative of the product. c can be nonzero up to g
physical nodes inside the PML interface, so do not restrict the T/S equations
to q>0. At corners add both axis contributions. Project interior memories
and guards to zero as specified above. Save old u^n as the new previous field.
After nt updates return `[u^nt,u^(nt-1),psi_y^nt,psi_x^nt,zeta_y^nt,zeta_x^nt]`.
Continuing with that state and the same settings is the same recurrence;
changing velocity is allowed and does not rescale the memories.

Render the physical u^(n+1) AFTER each update into the API's RGBA32Float
target. Receiver traces and texture samples therefore differ by one step.

## Periodic refinement mode (`boundary='periodic'`)

w=0. Set a=b=c=0 and all memories to zero. Pressure stencil indices wrap
modulo ny/nx, including corners; hidden pressure guards are refreshed by this
wrap each step instead of being zero. This is ordinary acoustic leapfrog on
a torus. The mode is required only for the analytic standing-wave refinement
test. Fourth-order space with dt proportional to spacing gives second-order
total convergence dominated by time error; it does not test spatial order.

## FWI derivative and objective scale

`J=0.5*sum((receiver_amplitudes-observed)^2)` over all nr*nt samples for the
single shot. This is a SUM, whereas project 16 uses a MEAN: J and its gradient
are nr*nt times that mean formulation. Differentiate the exact finite sequence
above with respect to physical velocity. Hold the initial six-channel state,
sources, acquisition, dt, grid spacing and all PML profiles fixed. Include
the velocity multiplier in source injection. Differentiate through edge
replication, summing every replicated PML cell's contribution into its
original physical edge/corner cell.

The Metal adjoint is the transpose of these discrete operations, including
record-before-update gather, interior/guard projections and old-memory
branches. If lambda^(n+1) is the cotangent of updated u^(n+1), the direct
extended-velocity contribution is
`lambda^(n+1)*2*V*dt^2*(S_y^n+S_x^n-f^n)` at each step. Inject the residual
into the cotangent of each recorded u^n, without normalization. Return dJ/dV
in AU²/(m/s), gathered back to the physical model by the extension transpose.
