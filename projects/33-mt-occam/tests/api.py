"""Project 33 public API. Only the MLX implementation is required.

All arrays supplied to implementations are contiguous NumPy float32 arrays;
all returned arrays must be NumPy float32 (the harness converts MLX arrays).
No reference libraries belong in implementations. Coordinates are depth positive
down from the surface, models are ordered shallow to deep, and the final layer
is an infinite halfspace. Frequency order must be preserved, including duplicates.
"""


def forward_mt(model, thickness, frequencies):
    """Predict isotropic, nonmagnetic, quasistatic 1D plane-wave MT data.

    Args:
        model: (L,) float32 log10 resistivity in ohm m; L >= 1.
        thickness: (L-1,) float32 positive finite-layer thicknesses, metres.
        frequencies: (F,) float32 positive frequencies in Hz; arbitrary order.

    Returns:
        dict(response=(2,F) float32). Row 0 is log10 apparent resistivity
        (ohm m), row 1 is phase in radians, in (0, pi/2). With exp(+i omega t),
        choose surface Z=Ex/Hy in quadrant I: rhoa=abs(Z)**2/(mu0*omega),
        phase=atan2(Im Z, Re Z). mu0=4*pi*1e-7 H/m exactly for this contract.
        Displacement current is neglected. A homogeneous halfspace gives
        response[0]=model[0], response[1]=pi/4 at every frequency.
    """
    raise NotImplementedError


def objective_and_grad(model, thickness, frequencies, observed, sigma, alpha):
    """Return a scalar regularized objective and its model gradient.

    Args:
        model, thickness, frequencies: as in forward_mt, float32 arrays.
        observed: (2,F) float32 observations in forward_mt row units.
        sigma: (2,F) float32 strictly positive independent standard deviations
            in the same units. Both rows and all frequencies are weighted.
        alpha: Python float >=0, dimensionless roughness multiplier.

    Let N=2F, r=(response-observed)/sigma, Dm=model[1:]-model[:-1].
    phi = 0.5*sum(r*r)/N + 0.5*alpha*sum(Dm*Dm). D is an unweighted
    first difference in layer index, includes the halfspace interface, and has
    no boundary penalty or reference-model penalty. Empty D for L=1 is zero.

    Returns:
        dict(objective=scalar shape () float32, gradient=(L,) float32).
        Gradient is dphi/dmodel (log10 resistivity, not linear resistivity
        or natural log). MLX must obtain this derivative through autodiff.
    """
    raise NotImplementedError


def invert_occam(initial_model, thickness, frequencies, observed, sigma,
                 target_rms, max_iterations):
    """Find the smoothest bounded 1D model that fits the data to target RMS.

    Args:
        initial_model: (L,) float32 starting log10 resistivity, in [0,4].
        thickness, frequencies, observed, sigma: as in objective_and_grad.
        target_rms: Python float, desired sqrt(mean(r**2)), fixed at 1 here.
        max_iterations: Python int, outer nonlinear iteration budget (150).

    Minimize R=0.5*sum((model[1:]-model[:-1])**2), subject to RMS <=
    target_rms and 0 <= model <= 4. Use Occam regularization/target search
    or an equivalent constrained smoothness solver. There is no prescribed
    solver trajectory. The initial model must not be treated as truth.

    Returns:
        dict(model=(L,) float32, response=(2,F) float32,
             rms=scalar shape () float32, roughness=scalar shape () float32).
        response must be forward_mt(model); rms and roughness must describe
        this returned model. No returned diagnostic is trusted without
        independent recomputation by the tests.
    """
    raise NotImplementedError
