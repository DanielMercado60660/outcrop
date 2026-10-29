"""Contract for the one-dimensional acoustic wave solver."""

def propagate(u0, u_prev, c, dx, dt, steps):
    """Propagate constant-speed acoustic displacement with Mur boundaries.

    u0 and u_prev are float32 arrays of shape (n,), displacement in arbitrary
    units at t=0 and t=-dt, on x_i=i*dx in meters. c is speed in m/s, dx is
    spacing in m, dt is time step in s, with 0 < c*dt/dx <= 1. steps is a
    nonnegative integer. Return {'wavefield': float32 array of shape
    (steps+1,n)}, including u0 as row 0. Interior points use centered
    second-order differences in space and time. At each step endpoints use
    first-order Mur: new[0]=old[1]+q*(new[1]-old[0]) and
    new[-1]=old[-2]+q*(new[-2]-old[-1]), q=(c*dt-dx)/(c*dt+dx).
    The supplied previous field is used directly, without initialization.
    """
    raise NotImplementedError
