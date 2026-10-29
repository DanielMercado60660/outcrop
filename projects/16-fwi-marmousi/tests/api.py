"""Shared MLX/autodiff and Metal/hand-adjoint contract for project 16.

Only one operation is required. Optimization lives in the test driver, so an
implementation never receives the true model or generates acquisition data.
The normative discrete equations, including every PML coefficient and update,
are in tests/DISCRETE_OPERATOR.md (relative to the project root). This API and
that document together define the operation; no reference source is needed.
"""


def misfit_and_grad(velocity, source_amplitudes, source_locations,
                    receiver_locations, observed, grid_spacing, dt,
                    accuracy, pml_width, pml_freq, max_vel):
    """Return receiver data, least-squares objective and its velocity derivative.

    All floating array inputs/outputs are contiguous NumPy float32 arrays.
    Implementations compute in float32. Integer locations are NumPy int64.
    Geometry is Cartesian (z,x), z increasing down, origin at model[0,0].

    Args:
        velocity: [nz,nx], positive isotropic acoustic speed in m/s. This is
            the differentiation variable (NOT slowness or squared slowness).
        source_amplitudes: [shots,sources,nt], sampled forcing f in units
            u/m^2, where u is an arbitrary scalar wavefield unit. Solve
            (1/v^2) u_tt - Laplacian(u) = -f: a sample injects
            -v^2*dt^2*f into the next wavefield, without cell-area scaling.
        source_locations: [shots,sources,2], integer (z,x) grid indices,
            inside the supplied unpadded model; additive point injections.
        receiver_locations: [shots,receivers,2], integer (z,x) indices.
            Record current u before advancing/injecting each source sample,
            as Deepwave scalar does; data[... ,0] is zero with zero ICs.
        observed: [shots,receivers,nt], target u, already generated and saved
            by the judge. No trace weighting, normalization, filtering or mute.
        grid_spacing: Python float, equal dz=dx in metres.
        dt: Python float, seconds; no internal substepping for locked cases.
        accuracy: Python int, spatial centered Laplacian order (4 in tests).
            Time stepping is centered second order, initially zero u/u_t.
        pml_width: Python int, exterior absorbing width in cells on all four
            sides (no free surface). Use the discrete C-PML equations in
            DISCRETE_OPERATOR.md, extending velocity by edge replication.
            Differentiate through that extension, summing edge/corner copies.
        pml_freq: Python float, Hz, C-PML design frequency in the written law.
        max_vel: Python float, m/s, fixed CFL/PML speed bound,
            held constant when differentiating, including at perturbed models.

    Returns:
        Dict with exactly these arrays:
        predicted: float32 [shots,receivers,nt], scalar wavefield u.
        objective: float32 scalar shape (), 0.5*mean((predicted-observed)^2),
            units u^2, averaged over ALL shots, receivers and time samples.
        gradient: float32 [nz,nx], d objective / d velocity, u^2/(m/s),
            including velocity dependence of source injection and PML extension.

    Match Deepwave scalar with full model (survey_pad=None), integer sampling,
    model_gradient_sampling_interval=1, no time/frequency taper or regularizer.
    For this lock only accuracy=4 and equal grid spacing are required. There
    are two zero guard cells beyond each exterior PML. No guard cell evolves.
    DISCRETE_OPERATOR.md defines the operator locally; a shared definition
    with prerequisite 13 is future work. MLX uses autodiff; Metal uses a
    hand-written discrete adjoint of those equations.
    """
    raise NotImplementedError
