"""Public contract for 2-D scalar Born modeling."""


def born(v, dv, source_amplitudes, source_locations, receiver_locations,
         wavefield_0, wavefield_m1, wavefield_sc_0, wavefield_sc_m1,
         dx, dt, accuracy, boundary, pml_width, pml_freq, max_vel):
    """Propagate background and linearized scattered acoustic wavefields.

    v, dv: float32 arrays [ny, nx], velocity and velocity perturbation in
    m/s. Axis 0 is depth y, axis 1 is horizontal x; origin is (0, 0).
    dv is the direction of differentiation with respect to v, not squared
    slowness. source_amplitudes: float32 [shots, sources, nt], the forcing f
    in u_tt = v**2 * (Laplacian(u) - f); point forcing is injected without
    cell-area normalization. source_locations and receiver_locations: int64
    [shots, points, 2], zero-based (y, x) grid indices. Coincident source and
    receiver locations across shots are allowed.

    wavefield_0, wavefield_m1, wavefield_sc_0, wavefield_sc_m1: float32
    [shots, ny, nx], values at t=0 and t=-dt, respectively. Initial PML
    memory variables are zero. dx is square grid spacing in metres, dt is
    seconds, accuracy is the centered spatial stencil order (2 or 4),
    boundary is 'pml' or 'periodic' on all faces, pml_width is the number
    of cells on each face (zero for periodic), pml_freq is PML frequency in
    Hz, and max_vel in m/s fixes the PML and CFL setup. The full discrete
    equations, PML padding, profiles, source terms, and boundary rules are
    specified in tests/SCHEME.md, which is part of this API. Supported inputs
    must have CFL step ratio 1 as defined there; the supplied cases do. No
    source taper or interpolation is used.

    Return dict with float32 arrays 'background' and 'scattered', each
    [shots, receivers, nt], sampled at t = 0, dt, ..., (nt-1)*dt before the
    step's source injection, matching Deepwave scalar_born. Wavefield and
    source units are consistent with the stated PDE; amplitudes are arbitrary.
    """
    raise NotImplementedError


def misfit_and_grad(v, dv, source_amplitudes, source_locations,
                    receiver_locations, wavefield_0, wavefield_m1,
                    wavefield_sc_0, wavefield_sc_m1, dx, dt, accuracy,
                    boundary, pml_width, pml_freq, max_vel, observed):
    """Return half squared scattered-data residual and its gradient in dv.

    Modeling arguments follow born. observed is float32 [shots, receivers,
    nt], in the same units and ordering as scattered. Return float32 arrays
    'objective' (scalar shape ()) and 'gradient' [ny, nx], the derivative of
    0.5 * sum((scattered - observed)**2) with respect to dv in m/s. Initial
    scattered wavefields are held fixed when differentiating.
    """
    raise NotImplementedError
