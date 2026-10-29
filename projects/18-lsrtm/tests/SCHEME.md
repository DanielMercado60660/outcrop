# Locked project 18 acoustic Born scheme (round 3)

This file defines the entire propagator used by `tests/api.py`. It is local
to project 18 and is the implementer's source of truth. No reference-library
access is needed. Only fourth-order space, second-order time, a positive PML
width and the no-substepping domain below are supported. All shots propagate
independently; the objective sums them. `judge/scheme_reference.py` is an
independent float64 NumPy transcription with an exact discrete transpose;
golden generation checks it against the reference outputs before accepting
either golden bundle. `golden/meta.json` records both files' hashes.

## Grid, extension, boundaries and sampling

Physical arrays have shape `(nz,nx)`, ordered `(z,x)`, with z positive down,
x positive right, origin `(0,0)` m. Let `h_z=spacing[0]`, `h_x=spacing[1]`,
`k=dt`, `p=accuracy/2=2`, and `W=pml_width`. The padded length on each axis is
`M=N+2*(W+p)`. Physical index zero is padded index `W+p`. Extend v and
`dv=mask*model` by clamping **both** coordinates to the nearest physical node,
including PML corners and the outer p guard nodes. Call this extension E.
All wavefields and memories have the padded shape, with initial value zero.
Update only the Cartesian product `p <= i_z < M_z-p`, `p <= i_x < M_x-p`;
the outer p guards stay zero for every wavefield and memory. The velocity and
dv guards retain the clamped values but cannot drive a guard wavefield.

Integer source and receiver indices shift by `(W+p,W+p)`. At the beginning
of sample n record the current scattered field `s^n` at receivers. Thus
sample zero is zero; source sample n changes `u^(n+1)` and `s^(n+1)`.
No interpolation, source regeneration, cell-area factors, normalization,
free surface, attenuation or cropping is applied. Coincident sources sum;
the adjoint of receiver sampling sums coincident receiver contributions.

## Centered derivatives and axis-specific PML profiles

On each axis j, with its own spacing h, define

```
D1_j q[i] = ((2/3)*(q[i+1]-q[i-1])
             -(1/12)*(q[i+2]-q[i-2])) / h
D2_j q[i] = ((4/3)*((q[i+1]-q[i])+(q[i-1]-q[i]))
             -(1/12)*((q[i+2]-q[i])+(q[i-2]-q[i]))) / h^2
```

For wavefield derivatives, evaluate only at updated nodes; derivative outputs
are zero in the guards. In the memory derivative `D1_j(a_j*psi_j)`, use the
entire old memory array, including the PML slab right up to its interface.
Never truncate the stencil to the region where its destination lies.

Build each axis's one-dimensional profiles at **all** padded indices,
including guards. Let `l=W+p`, `r=M-1-W-p` (the first/last physical node),
and `vmax=max(v)` fixed for every call. For that axis,

```
qL = clip((l-i)/W, 0, 1)
qH = clip((i-r)/W, 0, 1)
q = max(qL,qH)
sigma = 3*vmax*ln(1000)/(2*W*h) * (qL^2+qH^2)
alpha = pi*pml_freq*(1-q)
if q > 0:
    a = exp(-(sigma+alpha)*k)
    b = sigma/(sigma+alpha) * expm1(-(sigma+alpha)*k)
else:
    a = b = 0
db = D1_j(b)
```

Use zero b when its denominator is zero. In this supported domain the two
face profiles do not overlap. Compute db with the same fourth-order centered
stencil, **not** an analytic derivative; set db=0 in that axis's p guards.
Broadcast the profiles along the other axis and over shots. Both `sigma`'s
normalization `W*h` and the derivative's h are per-axis: the layered grid has
`[h_z,h_x]=[10,12]` m. No square-cell approximation is allowed.

## One time step: full PML expression at every updated node

For each field q (background u and scattered s), independently maintain
`psi_j`, `zeta_j` for j in {z,x}. Using exclusively old fields and memories,
compute **at every updated node**, even where a=b=0,

```
g_j = D1_j(q)
T_j = (1+b_j)*D2_j(q) + db_j*g_j + D1_j(a_j*psi_j)
H_j = (1+b_j)*T_j + a_j*zeta_j
psi_j_new  = b_j*g_j + a_j*psi_j
zeta_j_new = b_j*T_j + a_j*zeta_j
H(q) = H_z + H_x
```

Within p physical nodes next to a PML, a and b are zero **but db and the
stencil of a*psi can be nonzero**. Retain **both** `db_j*g_j` and
`D1_j(a_j*psi_j)` there. Replacing H_j by D2_j at non-PML destinations is
incorrect for this scheme. Memories themselves remain zero where a=b=0;
their neighboring stencil contributions do not.

Simultaneously advance

```
u_new = 2*u - u_old + k^2*v^2*H(u)
s_new = 2*s - s_old + k^2*(v^2*H(s) + 2*v*dv*H(u))
```

At each source node, sum `-k^2*v^2*source_amplitudes[...,n]` into u_new and
`-k^2*2*v*dv*source_amplitudes[...,n]` into s_new. The source units and
negative sign are intentional; no area division is applied. Install new
memories, rotate previous/current fields, and continue. PML profiles remain
fixed when differentiating; vmax is not differentiated. The scattered
memories are the tangents of the background memories, so the same linear
memory recurrence acts on s.

## CFL and substepping rule

The conservative step ratio is

```
R = max(1, ceil(abs(dt)*vmax*sqrt(1/h_z^2+1/h_x^2)/0.6))
```

Every supported call must have R=1, as every fixture does. Thus k=dt is both
the integration interval and recorded-data interval. Inputs with R>1 are
outside this API's supported domain; no source resampling, time taper,
frequency taper, padding or downsampling is specified or required.

## Exact discrete transpose and model fold-back

`A model = R_sample * s` using `dv=E(mask*model)`. Its transpose is the
reverse of the full recurrence above, with the ordinary Euclidean inner
product (no dt, dz, dx weights). It includes transposes of variable velocity
multiplication, all memory recurrences, source injection and receiver sampling.
It is not obtained by simply running a damped wave equation backward.

For an explicit reverse step, cache the background driving array
`B_n=2*k^2*v*(H(u_n)-f_n)` on the updated grid; f_n is source sample n
scattered into the padded grid and zero elsewhere. Let L be the sensitivity
of s_new, P_j of psi_j_new, Z_j of zeta_j_new, and J the sensitivity of the
new previous field (which was the old s). Before adding receiver data at n,
reverse the step as follows:

```
gradient_padded += sum_shots(B_n*L)
bar_s = 2*L + J
bar_s_old = -L
for j in {z,x}:
    V = k^2*v^2*L
    U_j = (1+b_j)*V + b_j*Z_j
    bar_s += D2_j^T((1+b_j)*U_j) + D1_j^T(db_j*U_j+b_j*P_j)
    bar_psi_j  = a_j*(P_j + D1_j^T(U_j))
    bar_zeta_j = a_j*(V+Z_j)
bar_s += R_sample^T(data[...,n])
```

Initialize all reverse states to zero after the final step. The derivative
matrices on zero-guard arrays satisfy `D1^T=-D1`, `D2^T=D2`; guard outputs
are discarded on the reverse pass too. Iterate n backward. Finally apply
`mask * E^T(gradient_padded)`: **sum**, rather than crop or average, every
replicated PML/guard sensitivity into its original physical edge/corner node.
For `misfit_and_grad`, supply `A model-observed` as the reverse data. For RTM,
supply observed. This defines the sign, scaling and boundary gradient fully.

## Solver and width convention

The fixed-count, two-pass reorthogonalized CGLS algorithm is specified in
`api.py`; it uses precisely this A and A.T. Width measurements are specified
in `metrics.py`: the full **69-point** depth FFT has no padding and implies
a periodic extension; window selection happens only after the envelope.
Changing padding, transform length or window order changes the locked metric.
