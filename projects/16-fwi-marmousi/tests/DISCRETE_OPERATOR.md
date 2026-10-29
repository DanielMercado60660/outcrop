# Project 16 discrete acoustic operator, version 1

This is the normative companion to `tests/api.py`. It specifies the numerical
problem for this lock, independently of library implementation details. It is
the same scalar C-PML problem evaluated by the campaign's pinned Deepwave
oracle. A different absorbing boundary defines a different discrete problem;
the 1e-4 parity threshold applies to these equations. Implementations should
derive their stepping and adjoint from these equations and the cited papers.
There is no requirement to read, import, or transcribe Deepwave source.

The acoustic least-squares formulation follows Tarantola (1984). The two-memory
scalar C-PML construction follows Pasalic and McGarry (2010), *Convolutional
perfectly matched layer for isotropic and anisotropic acoustic wave equations*.
The polynomial profile, placement, endpoint closure and discrete product rule
below are explicit numerical choices for this project, rather than claims that
the paper uniquely prescribes them. Reference-library source was inspected by
the judge builder to audit these choices; no library source is vendored here.
`judge/make_goldens.py` also checks a separately expressed NumPy evaluation of
these equations against Deepwave. This document fully replaces the former
unspecified instruction to inherit project 13's PML.

## Grid, extension and stencil

All coordinates are nodal integer indices, ordered (z,x). Let `h=grid_spacing`,
`w=pml_width`, `g=accuracy/2=2`, and `P=w+g`. Along an axis of input length `n`,
the allocated length is `N=n+2P`. The original nodes occupy indices
`P,...,P+n-1`; a source or receiver index `j` becomes `j+P`.
The evolving region is `g <= i < N-g` on BOTH axes, including the exterior PML.
The outermost `g` nodes on every face are guards: u and all memory fields there
are zero at every time, including corners. There is no free surface or mirror
boundary. Shots evolve independently with the same velocity.

The extended velocity is
`V[i,j]=velocity[clip(i-P,0,nz-1),clip(j-P,0,nx-1)]`.
This rule includes corners. Guard velocities are immaterial because guards
never evolve. Keep all input-model cells in the differentiated domain.

For either axis q, with `e_q` its unit index increment, use

```text
Dq F[i]  = { (2/3)(F[i+e_q]-F[i-e_q])
             -(1/12)(F[i+2e_q]-F[i-2e_q]) } / h
Dqq F[i] = { (4/3)(F[i+e_q]+F[i-e_q]-2F[i])
             -(1/12)(F[i+2e_q]+F[i-2e_q]-2F[i]) } / h^2
```

These are distinct fourth-order first and second derivative stencils: do NOT
replace Dqq by Dq composed with Dq. Stencils read guard values. Profile
derivatives below use the same Dq with zero derivative values assigned to the
outer two indices (not periodic rolls or one-sided stencils).

## One-dimensional PML profiles (separately on z and x)

For each allocated axis index `i=0,...,N-1`, set

```text
ell = g+w                      # first original model node
r   = N-1-g-w                  # last original model node
tL  = clip((ell-i)/w,0,1)
tR  = clip((i-r)/w,0,1)
t   = max(tL,tR)
R   = 0.001                    # design reflection amplitude
p   = 2                        # polynomial degree
L   = w*h                      # physical thickness, metres
sigma0 = -(1+p)*max_vel*ln(R)/(2*L)       # 1/s
sigma  = sigma0*(tL^p+tR^p)              # 1/s
alpha  = pi*pml_freq*(1-t)                # 1/s
a[i]   = exp(-(sigma+alpha)*dt) if t>0 else 0
b[i]   = sigma/(sigma+alpha)*(exp(-(sigma+alpha)*dt)-1)
```

If sigma+alpha is zero, set b=0. The locked cases have w=10; w=0 is not
required. `expm1` can evaluate the parenthesized difference accurately.
In particular at the model interface t=0, BOTH a and b are zero, not a=1.
Profiles extend into the guards with t clipped at 1; do not zero a,b in guards.
They multiply zero guard memories. Set `c=Dq b` on indices `2,...,N-3`, with
c=0 at the outer two indices. Broadcast each axis profile across shots and the
other axis. All profiles are fixed when differentiating velocity: max_vel is
the supplied scalar, never a maximum recomputed from the model.

For the locked h=10 m, w=10, max_vel=2600 m/s, dt=0.001 s, pml_freq=15 Hz,
sigma0 is approximately 269.402456 /s. At t=1, a is approximately 0.763836 and
b approximately -0.236164; at t=0.1, sigma is approximately 2.694025 /s and
alpha approximately 42.411501 /s. These are explanatory values, not additional
test tolerances. Interface placement is at a model NODE, not a half cell.

## Time recurrence and recording

Initialize `u^-1=u^0=0`, `psi_z^0=psi_x^0=zeta_z^0=zeta_x^0=0`.
For each sample `k=0,...,nt-1`, record `predicted[...,k]` by gathering u^k at
the integer receiver nodes BEFORE injection or evolution. For q in {z,x},
compute on the evolving region, always using OLD memory values:

```text
Tq^k     = (1+bq)*Dqq u^k + cq*Dq u^k + Dq(aq*psi_q^k)
Sq^k     = (1+bq)*Tq^k + aq*zeta_q^k
psi_q^(k+1)  = bq*Dq u^k + aq*psi_q^k
zeta_q^(k+1) = bq*Tq^k + aq*zeta_q^k
u^(k+1) = 2*u^k-u^(k-1) + V^2*dt^2*(Sz^k+Sx^k-f^k)
```

Here f^k is the sum of source_amplitudes[...,k] injected at each source node,
zero elsewhere. Injection is additive for coincident sources. There is no
division by cell area and no injection into a memory field. Dq(aq*psi_q)
differentiates the product, not aq times a differentiated memory. `cq*Dq u`
is a separate term and cannot be dropped. At corners, sum BOTH axis
contributions. Where a=b=c=0 the same equations reduce to ordinary acoustic
leapfrog stepping. c can be nonzero in the first two original-model nodes
adjacent to a PML: do not restrict the PML equations to t>0 nodes. It is safe to
evaluate the displayed recurrence throughout the evolving region. Set all
new guards to zero after each step. No time/frequency taper, resampling,
substepping, survey cropping, attenuation or regularization is present. The
locked 1 ms timestep is below the fourth-order stability bound at max_vel.

## Objective and derivative

Let M=shots*receivers*nt. `J=sum((predicted-observed)^2)/(2*M)`, with the sum
over every trace sample. Differentiate the exact finite sequence above with
respect to the supplied velocity, holding acquisition and profiles fixed.
The discrete adjoint is the transpose of this sequence, including the two
memory recurrences, guard-zero projection, point scatter/gather and edge
replication. Inject residual/M into the cotangent of each RECORDED u^k.
It is not the adjoint of a different continuous absorbing-boundary equation.

If lambda^(k+1) denotes the reverse-mode cotangent of the updated u^(k+1),
the direct extended-velocity contribution at that step is
`lambda^(k+1)*2*V*dt^2*(Sz^k+Sx^k-f^k)`, summed over shots and times.
This includes differentiation of the source multiplier. The adjoint of each
Dq or Dqq is its matrix transpose with the same guards and projection;
reverse the displayed products in order, including the OLD-memory branches.
Sum the resulting extended gradient into the original cell whose clipped
indices were used for V. Boundary cells therefore collect ALL replicated PML
copies, and corners collect both-axis copies. No derivative of max_vel,
sigma, alpha, a, b, or c is included. Return dJ/dvelocity in u^2/(m/s),
not dJ/dslowness or an unnormalized sum over shots.

## Scope and shared dependency

The tests retain their reviewed 1e-4 same-discretization parity, 0.999 cosine,
1e-3 centered-FD and 0.70 model-error-ratio criteria. This document specifies
the missing numerical choices without relaxing those thresholds. It is local
because this round permits edits only in project 16. Campaign owners should
review and promote a consistent shared spec for 13, 14, 16 and 21 before
sharing kernels; this file does not silently change another project's API.
