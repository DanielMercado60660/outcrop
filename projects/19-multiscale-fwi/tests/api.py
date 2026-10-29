"""Frozen project 19 API. Only the MLX path is required.

Arrays are NumPy float32, except integer grid coordinates (int64). The two
functions return dicts of named float32 NumPy arrays. No reference library,
golden-file access, generated survey/data, or custom result objects belong in
an implementation. Derivatives on MLX must use autodiff through propagation,
source scaling, filtering, and the velocity parameterization.
"""


def misfit_and_grad(*, parameters, basis, source, sources, receivers,
                    observed, dx, dt, cutoff):
    """Evaluate the 2D acoustic, frequency-filtered waveform objective.

    Arguments
    ---------
    parameters : float32 ndarray (P,)
        Positive velocity coefficients, m/s. Velocity is exactly
        v[z,x] = sum_p parameters[p]*basis[p,z,x]. No background term.
    basis : float32 ndarray (P,Nz,Nx)
        Dimensionless, fixed, nonnegative spatial partition of unity.
        P=2 in the locked fixtures. Inversion is over these coefficients,
        rather than independent grid-cell velocities.
    source : float32 ndarray (S,Q,T)
        Samples f at t=n*dt in laplacian(u)-u_tt/v^2=f, units u/m^2;
        arbitrary source normalization, identical on both paths. Inject
        -v(source)^2*dt^2*f into the next wavefield, without a cell-area
        factor. Differentiate this velocity-dependent scaling too.
    sources : int64 ndarray (S,Q,2)
        Integer [z,x] grid indices, origin (0,0) at the first physical
        cell, z positive down, x positive right. No subcell interpolation.
    receivers : int64 ndarray (S,R,2)
        Integer [z,x] indices in the same physical grid. S independent shots.
    observed : float32 ndarray (S,R,T)
        Scalar wavefield u, same arbitrary units as prediction. Deepwave
        receiver convention: record current u[n] before update/injection of
        f[n], so sample zero is exactly zero. No pressure/time derivative.
    dx : Python float
        Isotropic cell spacing, m (20 in fixtures).
    dt : Python float
        Sampling interval, s (0.002); zero initial u and previous u.
    cutoff : Python float
        Positive cutoff fc in Hz. With f_k=k/(T*dt), apply the zero-phase
        periodic real-FFT filter H_k=1/sqrt(1+(f_k/fc)^8), length T,
        to both prediction and observed. No padding, taper, causal IIR,
        or resampling. Standard unnormalized rfft / 1/T irfft convention.

    Propagation is Deepwave scalar's matched discretization: fourth-order
    central spatial differences, second-order time, 12-cell CPML on each
    face, edge-replicated velocity into CPML, pml_freq=12 Hz and FIXED
    max_vel=3000 m/s. No free surface, survey cropping, time taper,
    time/frequency padding or gradient subsampling. Locked dt is stable
    without internal time subdivision. Include derivatives of replicated
    edge velocities; CPML profiles use fixed max_vel and pml_freq.

    Returns
    -------
    dict with exactly:
        predicted : float32 ndarray (S,R,T), unfiltered u, units as observed.
        objective : float32 ndarray (), dimensionless scalar
            J = 0.5*sum(H(predicted-observed)^2)/sum(H(observed)^2),
            summed over every shot, receiver and time. No regularization,
            mean factor, gain fitting or additional normalization.
        gradient : float32 ndarray (P,), dJ/dparameters, units s/m.
            Derivative of J itself, with no clipping or preconditioning.
    """
    raise NotImplementedError


def invert(*, initial_parameters, basis, source, sources, receivers,
           observed, dx, dt, cutoffs, steps_per_stage):
    """Run fixed-budget frequency continuation using the objective above.

    initial_parameters : float32 ndarray (P,), m/s, initial coefficients.
    basis, source, sources, receivers, observed, dx, dt : as misfit_and_grad.
        Return values obey the same grid, source, filter and dtype conventions.
    cutoffs : list of Python floats (K,), Hz, stage order as supplied.
        [3,5,9,18] is multiscale; [18,18,18,18] is the single-scale
        control. Repeated cutoffs still reset the stage stall flag.
    steps_per_stage : Python int, fixed 35 in fixtures.
        Both schedules have the same 140 available updates.

    Protocol (the locked control experiment, not a tunable optimizer):
    At every stage reset stalled=False. At a nonstalled iteration evaluate
    J and gradient g. If J<=1e-10, stall for the rest of this stage.
    Otherwise d=-80*g/max(max(abs(g)),1e-30) m/s. For k=0,...,15, try
    q=clip(p+2**(-k)*d,1500,2900) componentwise in m/s. Accept the first
    candidate with J(q)<=J(p)+1e-4*dot(g,q-p). If none is accepted,
    stall for the rest of this stage. A stalled iteration preserves p
    and its preupdate J; append repeated states to preserve the budget.
    Carry p into the next stage; no truth, oracle endpoint, noise,
    random initialization, regularization or optimizer state reset of p.

    Returns a dict with exactly:
        parameters : float32 ndarray (P,), m/s, final coefficients.
        trajectory : float32 ndarray (K*steps_per_stage+1,P), m/s,
            initial coefficients followed by every postupdate state.
        objective_history : float32 ndarray (K*steps_per_stage,),
            dimensionless PREUPDATE J at that iteration's stage cutoff.
            Values across stage boundaries need not decrease.
    """
    raise NotImplementedError
