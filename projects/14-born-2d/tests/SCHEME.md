# Discrete scalar Born scheme

This is the complete contract for the cases in `api.py`, including the two
supported orders and the specified no-substepping domain. Arrays and locations
describe grid nodes. Every shot is independent.

## Grid, boundary and initial state

Let `h=dx`, `k=dt`, `p=accuracy/2`, `W=pml_width`, and let `N` be
the model length along an axis. For PML, extend the model by W nodes outside
each face and add p further guard nodes outside those. The padded length is
`M=N+2*(W+p)` and model index 0 is padded index `W+p`. Extend both v and dv
to the entire padded grid by clamping each coordinate to the nearest model
edge (including corners). Model-shaped initial wavefields are zero outside
the model. All auxiliary fields initially vanish everywhere. PML guard nodes
are permanently zero for all wavefields and auxiliary fields. Update only
indices `p <= i < M-p`. PML nodes along an axis are `p <= i < p+W` or
`p+W+N <= i < M-p`; other updated nodes are interior for that axis.

For periodic boundary W=0: use the model itself and evaluate all wavefield
stencils with indices modulo N on each axis (equivalent to wrapping p guard
nodes at every step); no PML auxiliary field is active. Initial `wavefield_0`
is u^0, `wavefield_m1` is u^-1, and the corresponding scattered arrays are
s^0 and s^-1. At the beginning of iteration n record u^n and s^n at integer
receiver nodes; there is no spatial interpolation, area scaling, or time
interpolation. Then compute the step and inject sources.

## Derivatives and profiles

For either axis define centered D1 q and D2 q, dividing by h and h²:

* accuracy 2: D1 uses `(q[i+1]-q[i-1])/2`; D2 uses
  `q[i+1]-2*q[i]+q[i-1]`.
* accuracy 4: D1 uses `(2/3)*(q[i+1]-q[i-1]) - (1/12)*(q[i+2]-q[i-2])`;
  D2 uses `(4/3)*(q[i+1]+q[i-1]-2*q[i]) - (1/12)*(q[i+2]+q[i-2]-2*q[i])`.

For each PML axis independently, build profiles on all padded indices i,
including guard nodes. The two interface positions are `l=p+W` and
`r=M-1-p-W`, i.e. the first and last model nodes. For W>0 set
`fraction=max(clamp((l-i)/W,0,1), clamp((i-r)/W,0,1))`,
`sigma=3*max_vel*ln(1000)/(2*W*h)*fraction²`,
`alpha=pi*pml_freq*(1-fraction)`. If fraction > 0, set
`a=exp(-(sigma+alpha)*k)` and
`b=sigma/(sigma+alpha)*expm1(-(sigma+alpha)*k)`; if the denominator
is zero take b=0. Where fraction=0 set a=b=0. Compute `db=D1(b)` using
the same centered stencil; set db=0 in the p guard positions. Do not use an
analytic derivative for db. Broadcast each one-dimensional profile along the
other axis. With W=0 use a=b=db=0.

## One step

For each field q, maintain separate psi_j and zeta_j for j in {y,x}.
For the background q=u and scattered q=s, independently, using the *old*
auxiliary fields, compute on each updated node:

* If the node lies in that axis's PML, `d=D1_j(q)`,
  `T_j=(1+b_j)*D2_j(q)+db_j*d+D1_j(a_j*psi_j)`,
  `H_j=(1+b_j)*T_j+a_j*zeta_j`,
  `psi_j_new=b_j*d+a_j*psi_j`, `zeta_j_new=b_j*T_j+a_j*zeta_j`.
* Otherwise `H_j=D2_j(q)` and both auxiliary fields for that axis remain
  zero. Ignore db at the interface model node, even though centered
  differentiation of b can make it nonzero there. The derivative of `a*psi`
  uses the old, zero-extended auxiliary array, including at its PML slab edges.

Let `H(q)=H_y+H_x`, with each field's own memories. Simultaneously compute

```
u_new = 2*u - u_old + k²*v²*H(u)
s_new = 2*s - s_old + k²*(v²*H(s) + 2*v*dv*H(u))
```

At each source node add `-k²*v²*f[n]` to u_new and
`-k²*2*v*dv*f[n]` to s_new. Sum contributions at coincident source nodes.
Then rotate u_old,u and s_old,s and install the new memories. The profiles
are held fixed when differentiating: max_vel is an independent supplied
scalar. The scattered memories are the tangent of the background memories,
so the same linear PML operator acts on s. When computing the dv gradient,
the transpose of clamped model extension sums all PML copies back into each
edge or corner model node. Initial scattered fields are fixed.

The conservative CFL step ratio would be
`max(1, ceil(abs(dt)*max_vel*sqrt(2)/(0.6*dx)))`. This API's supported
inputs require that ratio to equal 1; all fixtures meet it. Thus dt is the
internal step and no temporal upsampling, taper, filtering, or downsampling
is part of this contract. Cases requiring a larger ratio are outside the API's
specified domain; this avoids imposing unspecified source resampling.
