"""Frozen numerical interface shared by MLX and Rust/Metal.

All quantities are dimensionless Boussinesq quantities: length is layer depth,
time is thermal diffusion time, temperature is imposed bottom-to-top contrast,
and velocity is thermal diffusivity / layer depth. The domain is the unit square.
Arrays use [z, x], with z increasing upward from bottom (row 0) to top (row N).
All array inputs and outputs are NumPy float32, C contiguous; no autodiff API is
required for this forward simulation project. See each function for the exact
discrete scheme. Implementations may use any solver that reproduces that scheme.
"""


def stokes(temperature, rayleigh):
    """Solve constant-viscosity infinite-Prandtl Stokes flow on a nodal grid.

    Parameters
    ----------
    temperature : float32 ndarray, shape (N+1, N+1), N >= 4
        Dimensionless temperature on x=i/N, z=j/N. Fixed-temperature horizontal
        walls and insulating vertical walls are the physical thermal conditions;
        the Stokes solve also accepts manufactured temperatures.
    rayleigh : float
        Nonnegative dimensionless Rayleigh number. Buoyancy is +Ra*T in the
        upward z direction; viscosity and thermal diffusivity are unity.

    Returns
    -------
    dict with psi, u, w : float32 ndarrays, each shape (N+1, N+1)
        Streamfunction, horizontal velocity, upward velocity. With h=1/N,
        solve L_h q = Ra*D_x T, then L_h psi=q, using the five-point Laplacian
        on interior nodes and q=psi=0 on all walls. D_x T is the centered
        (T[j,i+1]-T[j,i-1])/(2h) on interior nodes. This is the discrete
        stress-free, impermeable biharmonic problem. There is no pressure gauge.
        Interior velocities are u=D_z psi, w=-D_x psi. Extend psi oddly through
        every wall when taking centered derivatives: u[0,:]=psi[1,:]/h,
        u[N,:]=-psi[N-1,:]/h, w[:,0]=-psi[:,1]/h,
        w[:,N]=psi[:,N-1]/h. Set normal velocities u[:,0/N] and w[0/N,:]
        exactly to zero. This fixes corner velocities to zero.
    """
    raise NotImplementedError


def convection(temperature, rayleigh, dt, steps):
    """Advance coupled Stokes/advection-diffusion with explicit midpoint RK2.

    Parameters
    ----------
    temperature : float32 ndarray, shape (N+1, N+1), N >= 4
        Initial temperature at the same nodes/units as stokes(). Bottom row is
        exactly 1 and top row exactly 0. Vertical walls have homogeneous Neumann
        conditions, represented by reflected ghosts T[j,-1]=T[j,1] and
        T[j,N+1]=T[j,N-1]; do not copy the first interior value to the wall.
    rayleigh : float
        Nonnegative Rayleigh number, as in stokes().
    dt : float
        Positive dimensionless step size. Caller supplies a stable step; there
        is no adaptive stepping, clipping, filtering, or internal rescaling.
    steps : int
        Nonnegative number of steps, including zero for pure diagnostics.

    Returns
    -------
    dict with temperature, u, w : float32 ndarrays, shape (N+1, N+1)
        Final temperature and the Stokes velocity recomputed from that final
        temperature. Define F(T)=L_h T-u(T)*D_x T-w(T)*D_z T on rows 1..N-1.
        L_h is the centered five-point Laplacian (including reflected x ghosts);
        centered first derivatives use the same ghosts, so D_x T=0 on x walls.
        F=0 on bottom/top rows. Each step computes V=T+(dt/2)*F(T), then
        T_new=T+dt*F(V), solving Stokes separately at BOTH stages. Temperature
        bottom/top rows stay 1/0 at every stage. This is midpoint RK2, not Heun.
    nu_top, nu_bottom, vrms : float32 ndarrays, shape ()
        Heat flux and RMS speed from the final fields. At each x node,
        top flux=(4*T[N-1]-T[N-2]-3*T[N])/(2h),
        bottom flux=(3*T[0]-4*T[1]+T[2])/(2h). Integrate each by composite
        trapezoidal quadrature in x. vrms=sqrt(integral(u**2+w**2) dx dz),
        using tensor-product composite trapezoidal quadrature, area=1.
        Nu=1 for conduction. These definitions apply even away from steady state.
    """
    raise NotImplementedError
