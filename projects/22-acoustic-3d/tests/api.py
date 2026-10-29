"""Implementation contract for the 3D acoustic propagation benchmark."""


def propagate(velocity, initial, previous, spacing, dt, steps):
    """Advance a scalar wavefield on a 3D periodic grid.

    Solve u_tt = velocity**2 * (u_xx + u_yy + u_zz), with the centered
    second-order three-point spatial stencil on each axis and centered
    second-order time stepping. No source or damping is applied. All six
    faces are periodic, wrapping to the opposite face. For each step use
    u_next = 2*u_current - u_previous + (velocity*dt)**2 * Laplacian(u_current).

    Args:
        velocity: float32 NumPy array (nz, ny, nx), wavespeed in m/s, positive.
        initial: float32 NumPy array (nz, ny, nx), pressure in arbitrary units
            at t=0, with grid point (0, 0, 0) at the origin.
        previous: float32 NumPy array (nz, ny, nx), pressure in the same units
            at t=-dt. Grid indexing is (z, y, x); axes increase positively.
        spacing: positive Python float, uniform grid spacing in meters on all axes.
        dt: positive Python float, time step in seconds; input satisfies CFL.
        steps: positive Python int, number of updates.

    Returns:
        Dict with float32 NumPy arrays of shape (nz, ny, nx): 'final' is
        pressure at t=steps*dt and 'last' is pressure at t=(steps-1)*dt.
        Both use the same pressure units and axis order as initial.
    """
    raise NotImplementedError
