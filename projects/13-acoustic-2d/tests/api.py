"""Contract for a Deepwave-compatible two-dimensional scalar propagator."""


def scalar(velocity, source, source_locations, receiver_locations, spacing,
           dt, order, pml_width, pml_freq, max_vel):
    """Propagate a scalar wave and return receiver time series.

    ``velocity``: float32 [nz, nx], wave speed in m/s, axis 0 is z and
    axis 1 is x, origin at index (0, 0). Grid spacing is uniform in both axes.
    ``source``: float32 [shots, sources, nt], samples of the right-hand side
    f in c^-2 u_tt - Laplacian(u) = -f, in units of u/m^2. A grid point
    source approximating a spatial delta must include the inverse cell area.
    ``source_locations``: int64 [shots, sources, 2], integer (z, x) indices.
    ``receiver_locations``: int64 [shots, receivers, 2], integer (z, x)
    indices. All indices are relative to the supplied velocity model, without
    PML; all locations are inside that model.
    ``spacing``: float, cell size in m. ``dt``: float, sample interval in s.
    ``order``: int, spatial finite-difference accuracy 2, 4, 6, or 8;
    time stepping is second order. ``pml_width``: list of four ints, number
    of PML cells on the low-z, high-z, low-x, high-x sides, respectively,
    added outside the velocity model (edge velocities replicated).
    ``pml_freq``: float, PML design frequency in Hz. ``max_vel``: float,
    fixed maximum velocity in m/s for Deepwave's CFL and PML calculation.

    Initial wavefields and PML memory variables are zero. Sources inject
    -c^2 dt^2 f at the current step, and receivers record the wavefield at
    times k*dt, starting with the zero initial field. The complete discrete
    equations and supported time steps are specified in tests/SCHEME.md.
    Return {'receiver_amplitudes': float32 [shots, receivers, nt]} in units
    of u. Source and receiver coordinates are grid-cell centers.
    """
    raise NotImplementedError
