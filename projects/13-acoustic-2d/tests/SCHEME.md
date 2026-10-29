# Discrete scalar scheme

This specifies the Deepwave scalar scheme for the supplied fixtures. Only
stable inputs with CFL step factor one are in scope; no time resampling is
required. Let h=spacing, k=dt, and p=order/2. Pad the velocity by the PML
widths and then by p halo cells, replicating edge values throughout. The
wavefields and memory fields have this padded shape, initially zero. Their
outer p halo cells stay zero. Model indices shift by p plus the low-side PML
width. Derivatives are computed only outside the halo and set to zero there.

For each axis, D1 f = sum_j c_j(f[i+j]-f[i-j])/h and D2 f =
(d_0 f[i]+sum_j d_j(f[i+j]+f[i-j]))/h². The coefficients are:

| order | c_1,...,c_p | d_0,d_1,...,d_p |
|---|---|---|
| 2 | 1/2 | -2, 1 |
| 4 | 2/3, -1/12 | -5/2, 4/3, -1/12 |
| 6 | 3/4, -3/20, 1/60 | -49/18, 3/2, -3/20, 1/90 |
| 8 | 4/5, -1/5, 4/105, -1/280 | -205/72, 8/5, -1/5, 8/315, -1/560 |

For each axis, let N be the full padded length and wL,wH its PML widths.
The interface coordinates are iL=p+wL and iH=N-1-p-wH. At index i define
qL=clamp((iL-i)/wL,0,1), qH=clamp((i-iH)/wH,0,1), taking q=0 for a zero
width. Set q=max(qL,qH), sigma=sum over faces of
3 max_vel ln(1000) q_face²/(2 w_face h), omitting zero-width faces, and
alpha=pi pml_freq (1-q). At q>0, a=exp(-(sigma+alpha)k),
b=sigma/(sigma+alpha) * expm1(-(sigma+alpha)k), with b=0 for zero denominator.
At q=0 set a=b=0. Compute db=D1(b), set to zero in the halo. Profiles extend
into the halo and are constant in time and independent of shot.

At each sample n, first record u_n at every receiver, then calculate from the
old fields for each axis d:

    g_d = D1_d(u_n)
    T_d = (1+b_d) D2_d(u_n) + db_d g_d + D1_d(a_d psi_d)
    psi'_d = b_d g_d + a_d psi_d
    zeta'_d = b_d T_d + a_d zeta_d
    u_(n+1) = 2u_n-u_(n-1) + v² k² sum_d[(1+b_d)T_d+a_d zeta_d]

Then add -v² k² source[n] at each integer source location, summing coincident
sources, replace memory fields by primed values, and advance. There is no
interpolation or extra area scaling. Receivers include the zero initial field;
the last injection affects the unreturned next field. Each PML memory field
is zero in the interior along its own axis. No model perturbation or gradient
is part of this forward-only API.
