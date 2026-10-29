def propagate(epsilon_r, electric, hx, hz, dx, dz, dt, steps):
    """Advance the lossless 2-D Yee TE fields with periodic boundaries.

    All arrays are float32, shape (nx, nz), with axes x,z and origin (0,0).
    epsilon_r is relative permittivity sampled at electric nodes; permeability
    is vacuum permeability everywhere. electric is E_y in V/m at integer
    time n=0. hx and hz are Z0*H_x and Z0*H_z in V/m at time -dt/2,
    respectively at (i*dx,(j+1/2)*dz) and ((i+1/2)*dx,j*dz).
    dx,dz are positive spacings in metres, dt is seconds, steps is a
    nonnegative integer. Z0=sqrt(mu0/epsilon0), c=299792458 m/s.

    Each step first updates hx += c*dt*forward_z(electric),
    hz -= c*dt*forward_x(electric), then electric += c*dt/epsilon_r *
    (backward_z(hx)-backward_x(hz)). For any array a,
    forward_z(a)[i,j]=(a[i,(j+1)%nz]-a[i,j])/dz and
    backward_z(a)[i,j]=(a[i,j]-a[i,(j-1)%nz])/dz; forward_x and
    backward_x use the same definitions on i with nx and dx. Indices wrap
    periodically. Return dict electric, hx, hz, float32 arrays
    of the input shape, at times steps*dt and (steps-1/2)*dt respectively.
    """
    raise NotImplementedError
