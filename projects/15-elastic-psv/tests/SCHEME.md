# Locked discrete P-SV scheme

This document is part of `tests/api.py` for both Tag B paths. It specifies
the complete forward calculation, including termination of the finite grid;
implementers do not need Deepwave source. The reference is the fingerprinted
CPU float64 Deepwave `elastic`, `accuracy=2`, `surface_method="ivm"`, with
`mpml_fraction=0`. Implementations use float32. Algebraically equivalent
operation ordering is allowed within `tests/tolerances.py`.

## Coordinates and material extension

Axes are `[y,x]`, positive down/right. Stresses are tensile positive. The
input normal-stress node `[i,j]` is at `(i*dy,j*dx)`. Let PML widths be
`[T,B,L,R]`. The computational grid has `Ny=ny+T+B`, `Nx=nx+L+R` normal
nodes, indexed `I=0..Ny-1`, `J=0..Nx-1`. Input `[i,j]` maps to `[i+T,j+L]`.
Its physical coordinates are `((I-T)*dy,(J-L)*dx)`.

Before staggering, extend the material by edge replication:
`q_pad[I,J]=q[clamp(I-T,0,ny-1),clamp(J-L,0,nx-1)]`. Equivalently convert
to Lamé parameters and nodal buoyancy first, then replicate those arrays.
There is no domain crop, interpolation, smoothing or additional material
cell. Materials are common to all shots; fields/memories are independent
per shot. At normal nodes set

```text
lambda = rho*(vp*vp - 2*vs*vs)
mu     = rho*vs*vs
b      = 1/rho in solid, exactly 0 in vacuum
```

All vacuum values `vp,vs,rho,lambda,mu,b` are exactly zero. Positive solid
values are far from underflow; no epsilon regularization is part of this
scheme. The staggered fields and coefficients (same array index `[I,J]`)
are:

| Field/coefficient | Offset from normal node in cells |
|---|---|
| sigma_yy, sigma_xx, lambda, mu | (0,0) |
| vy, by | (+0.5,0) |
| vx, bx | (0,+0.5) |
| sigma_xy, mu_yx | (+0.5,+0.5) |

Here `sigma_xy=sigma_yx` is a single field. On the padded grid, compute

```text
by[I,J] = 2/(rho_pad[I,J]+rho_pad[I+1,J])       for I < Ny-1
bx[I,J] = 2/(rho_pad[I,J]+rho_pad[I,J+1])       for J < Nx-1
mu_yx[I,J] = 4/(1/mu[I,J] + 1/mu[I+1,J]
                 + 1/mu[I,J+1] + 1/mu[I+1,J+1])  for I < Ny-1, J < Nx-1
```

`by` or `bx` is zero if its density sum is zero. `mu_yx` is zero if ANY of
its four mu values is zero. Evaluate reciprocals only on active entries to
avoid `0*inf` or NaN. These are inverse arithmetic density means and a
four-node harmonic shear mean, respectively, including in the PML.

## Outer-edge termination (with or without PML)

Every wavefield value outside the padded computational grid is zero at all
times. Missing neighbors in differences below are zero; there is no wrap,
replication of fields, or stress/velocity image condition. In addition,
staggered coefficients whose half-node extends past the last normal node
are **zero**, not replicated:

```text
by[Ny-1,:] = 0
bx[:,Nx-1] = 0
mu_yx[Ny-1,:] = 0
mu_yx[:,Nx-1] = 0
```

Consequently `vy[Ny-1,:]`, `vx[:,Nx-1]`, and `sigma_xy` on the last row and
last column remain exactly zero from the zero initial state. Normal
stresses on those nodes still follow their update equations. At the low
edge, for example, `(f[0,J]-f[-1,J])/dy = f[0,J]/dy`; the outside field is
zero. The library's internal one-cell low-side storage halo (`fd_pad=[1,0]`
per axis) is not an extra physical or replicated material cell in these
equations. These edge rules are essential to the `reflective` fixture,
which has zero PML widths. This is a specified discrete termination, not a
claim that it realizes a particular continuum rigid/free boundary.

## Spatial operators and C-PML

For any field array at the offsets listed above, define

```text
D_y+ f[I,J] = (f[I+1,J]-f[I,J])/dy
D_y- f[I,J] = (f[I,J]-f[I-1,J])/dy
D_x+ f[I,J] = (f[I,J+1]-f[I,J])/dx
D_x- f[I,J] = (f[I,J]-f[I,J-1])/dx
```

All stencil coefficients are 1. These differences are centered at the
destination staggered location and second order in the interior.

Construct two profiles per axis, one at integer nodes and one at +0.5
nodes. For a generic axis with input size `n`, padded index `k`, low/high
widths `wlo,whi`, spacing `h`, and offset `o=0` or `0.5`, let `u=k+o`.
The PML begins at the **edge normal nodes of the input**, `elo=wlo`,
`ehi=wlo+n-1`; it does not begin half a cell outside those nodes. Define

```text
slo = clamp((elo-u)/wlo, 0, 1) if wlo>0 else 0
shi = clamp((u-ehi)/whi, 0, 1) if whi>0 else 0
sigma_lo = -3*max_vel*ln(0.001)*slo^2/(2*wlo*h) if wlo>0 else 0
sigma_hi = -3*max_vel*ln(0.001)*shi^2/(2*whi*h) if whi>0 else 0
sigma = sigma_lo + sigma_hi
s = max(slo,shi)
alpha = pi*pml_freq*(1-s)
```

Each face has its own physical width. `s` is clamped before forming sigma
and alpha. There is no cross-axis damping (`mpml_fraction=0`) and
`kappa=1`. For `s>0`,

```text
a = exp(-(sigma+alpha)*dt)
b = sigma/(sigma+alpha) * expm1(-(sigma+alpha)*dt)
```

For `s=0`, set **both a and b to zero**, including at an edge node of the
input model. `0.001` is the PML design reflection parameter, not a test
tolerance. Polynomial degree is 2; alpha's scale is `pi*pml_freq`, not
`2*pi*pml_freq`. `max_vel` and `pml_freq` are explicit fixed API inputs.

For each derivative occurrence `D` below maintain its own memory `psi`,
initially zero, at the destination index. Once per time step use

```text
psi_new = a*psi_old + b*D
C(D) = D + psi_new
```

Use the profile of the derivative's axis at the derivative's destination
coordinate along that axis. Equivalently:

| Derivative | Axis profile | Independent memory |
|---|---|---|
| D_y+ sigma_yy | y half | psi_syy_y |
| D_x- sigma_xy | x integer | psi_sxy_x |
| D_y- sigma_xy | y integer | psi_sxy_y |
| D_x+ sigma_xx | x half | psi_sxx_x |
| D_y- vy | y integer | psi_vy_y |
| D_x- vx | x integer | psi_vx_x |
| D_y+ vx | y half | psi_vx_y |
| D_x+ vy | x half | psi_vy_x |

The normal-stress updates share the same corrected `D_y- vy` and `D_x- vx`
values; do not advance those memories twice. At corners, correct the two
axis derivatives separately before summing. No PML damping is applied to
the source term itself.

## Time step, forcing and recording

At the start of step `n`, velocities are `v^(n-1/2)`, stresses `sigma^n`.
All are initially zero (`v^(-1/2)=0`, `sigma^0=0`), as are all eight
memories. `n=0..nt-1`. No time subdivision, resampling or tapering occurs.

1. Record `vy^(n-1/2)`, `vx^(n-1/2)` and
   `p^n=-(sigma_yy^n+sigma_xx^n)/2` at receiver indices shifted by `[T,L]`.
   The two velocity channels occupy different physical staggered nodes;
   no collocation interpolation occurs. Pressure is the in-plane mean
   compressive stress, not a three-dimensional thermodynamic pressure.
2. From old stresses compute the four corrected stress derivatives,
   updating their four memories. Update velocities simultaneously:

   ```text
   vy_new = vy_old + dt*by*(C(D_y+ sigma_yy) + C(D_x- sigma_xy))
   vx_new = vx_old + dt*bx*(C(D_y- sigma_xy) + C(D_x+ sigma_xx))
   ```

3. At each source index shifted by `[T,L]`, add
   `dt*by*source_y[shot,source,n]` and
   `dt*bx*source_x[shot,source,n]` to the respective new velocities.
   Sum contributions if sources coincide. Sources are force **densities**
   in N/m^3, sampled at `n*dt`; no cell-area normalization is added.
4. From these source-injected new velocities compute the four corrected
   velocity derivatives, updating their four memories. Set
   `ey=C(D_y- vy_new)`, `ex=C(D_x- vx_new)`,
   `g=C(D_y+ vx_new)+C(D_x+ vy_new)` and update stresses simultaneously:

   ```text
   w = lambda*(ey+ex)
   sigma_yy_new = sigma_yy_old + dt*(w + 2*mu*ey)
   sigma_xx_new = sigma_xx_old + dt*(w + 2*mu*ex)
   sigma_xy_new = sigma_xy_old + dt*mu_yx*g
   ```

Outputs contain `nt` samples at `(n-0.5)*dt` for velocities and `n*dt` for
pressure. There is no final extra recorded sample. SI units and output
shapes/dtypes are given in `api.py`.

## Free surface and gradients

The flat IVM surface uses input rows 0 and 1 with `vp=vs=rho=0`, solid
beginning at row 2, and `T=0`. The physical interface is at `y=1.5*dy`.
`vy[1,:]` has buoyancy `2/rho[2,:]`; `vx[1,:]` has zero buoyancy. Vacuum
normal stresses and any shear node touching vacuum remain exactly zero.
Thus a vertical force at index row 1 loads the surface directly. Do not
apply an additional stress image or parameter-modified surface rule.

`misfit_and_grad` differentiates the complete discrete calculation above
and the objective in `api.py`. This includes source buoyancy scaling,
four-node harmonic averaging and the replicated material extension.
Reverse contributions from replicated PML cells sum into their original
edge material cells. All shots contribute to a shared material gradient.
Geometry, profiles (including `max_vel`), observations, time/space steps
and `data_scale` are fixed. Vacuum is a fixed mask and its three material
gradients are exactly zero. Both paths provide all five gradients; this
scope is intentional groundwork for elastic FWI.

## Independent physical checks and scope

Same-grid parity pins spatial stencils and the rules above. The analytic
P/S force-sheet check uses `v=F(t-distance/c)/(2*rho*c)` and grid force
density `F(t)/dy`, with `dt` halved alongside `dy=dx` at fixed CFL. Its
expected order is the combined space/time order **2**. This analytic
check propagates along y on square cells; x stencils, rectangular cells
and two-dimensional coupling are covered by discrete parity and Lamb.

Lamb's vertical line-load case checks Rayleigh arrival, not the full
analytic transient Green tensor or amplitude. For `beta=(vs/vp)^2`, solve
`q^3-8*q^2+(24-16*beta)*q-16*(1-beta)=0` for `0<q<1`, then
`cR=vs*sqrt(q)`. The far-field vertical velocity pulse is proportional to
the Hilbert transform of the source derivative; for the saved Ricker
source its peak is at `source_peak+offset/cR`. Arrival windows and analytic
targets are saved before implementation; absolute and differential
travel-time errors must be within 1%.
