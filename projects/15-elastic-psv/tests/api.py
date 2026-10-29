"""Frozen float32 NumPy interface for both Tag B implementation paths.

The complete discrete equations and boundary rules are in tests/SCHEME.md.
That document is part of this API and is fingerprinted in golden/meta.json.
"""


def elastic(vp, vs, rho, source_y, source_x, source_locations,
            receiver_locations, spacing, dt, pml_width, pml_freq, max_vel):
    """Propagate zero-initial-state, isotropic 2-D plane-strain P-SV waves.

    All arrays are C contiguous NumPy arrays; floating inputs/outputs are
    float32, locations are int64. No input may be mutated. Axis order is
    [y, x], y positive down, x positive right, origin at normal-stress node
    [0, 0]. SI units throughout. No gravity or attenuation.

    Parameters
    ----------
    vp, vs : (ny, nx), float32, m/s
        P and S speeds at normal-stress nodes (i*dy, j*dx).
    rho : (ny, nx), float32, kg/m^3
        Density. Solid cells have rho>0, vs>0, vp^2>2*vs^2. Vacuum cells
        have vp=vs=rho=0 EXACTLY. Convert to lambda=rho*(vp^2-2*vs^2),
        mu=rho*vs^2, buoyancy=1/rho in solid and zero in vacuum.
        Use Deepwave's improved-vacuum method (IVM), not PM or stress images.
        Shear modulus at (i+.5,j+.5) is the harmonic mean of four adjacent
        mu values, zero if any is zero. Velocity buoyancy is 2/(rho_i+rho_j)
        across its two neighboring nodes, zero if both are vacuum. Thus an
        interface velocity has twice the solid buoyancy. Flat free surfaces
        use two vacuum rows at the top and pml_width[0]=0; the physical
        interface is halfway between the last vacuum and first solid row.
    source_y, source_x : (nshot, nsource, nt), float32, N/m^3
        Collocated INDEX lists for vertical and horizontal body-force density
        components, sampled at t=n*dt (2-D line-source interpretation per
        unit out-of-plane thickness). These are force DENSITIES, not forces
        divided internally by cell area. Both components always supplied;
        unused components contain zero. Add dt*staggered_buoyancy*source to
        velocity after its stress-divergence update, before the stress update.
    source_locations : (nshot, nsource, 2), int64, grid indices
    receiver_locations : (nshot, nreceiver, 2), int64, grid indices
        Integer [i,j] selects vy at ((i+.5)*dy,j*dx), vx at
        (i*dy,(j+.5)*dx), and pressure at (i*dy,j*dx). Components are not
        interpolated to a common physical point. All indices must be >=0
        and less than [ny-1,nx-1]. Multiple shots share the material model.
    spacing : list[float], length 2, [dy, dx], meters
    dt : float, seconds
        Time step. All locked inputs satisfy CFL without substepping.
    pml_width : list[int], length 4
        C-PML thickness in cells OUTSIDE the input model, ordered top,
        bottom, left, right. Replicate edge material into PML, then stagger
        coefficients. Fields outside this padded grid are zero. At its last
        row buoyancy_y=0; at its last column buoyancy_x=0; at its last row
        AND last column mu_yx=0. Missing low-edge neighbors in differences
        are zero. There is no extra replicated low-side halo. These rules
        apply also when a face has zero PML width. See tests/SCHEME.md.
        A zero top width is used only with vacuum in the
        free-surface cases. Zero widths on solid are used as a reflective
        negative control. Match Deepwave elastic CFS-C-PML profiles, kappa=1,
        mpml_fraction=0, including integer and half-cell profile sampling.
        Fractional depth s=0 is at the input model's edge node, half-node
        profiles sample at +0.5 cell, and s is clamped to [0,1]. On each
        axis, sigma sums the two faces and alpha uses their larger s.
        For fractional depth s in a face of physical width L, use degree 2,
        sigma=-3*max_vel*log(0.001)*s^2/(2*L), alpha=pi*pml_freq*(1-s).
        Its derivative memory update is psi_new=a*psi+b*D, followed by
        D_corrected=D+psi_new, with a=exp(-(sigma+alpha)*dt) and
        b=sigma/(sigma+alpha)*expm1(-(sigma+alpha)*dt). Both memory
        coefficients are zero outside active PML, including s=0. These are
        profile design parameters, independent of the test error tolerances.
    pml_freq : float, Hz
        Explicit C-PML design frequency; never infer it from the source.
    max_vel : float, m/s
        Explicit fixed speed for C-PML construction and CFL. Do not
        differentiate this value or infer it from the material arrays.

    Returns
    -------
    dict with exactly vy, vx, p : (nshot, nreceiver, nt), float32
        Velocities in m/s recorded BEFORE each velocity update, at times
        (n-.5)*dt, starting from zero. Pressure in Pa recorded at n*dt,
        p=-(sigma_yy+sigma_xx)/2 (in-plane mean stress, compressive positive;
        this is not 3-D thermodynamic pressure). Stresses are tensile positive.

    Scheme is fixed: centered second-order staggered spatial differences,
    second-order leapfrog time stepping, initially zero velocity, stress,
    and all C-PML memory. No resampling, tapering, cropping of the input
    domain, source interpolation, attenuation, or internal time subdivision.
    The complete update order and eight derivative memories are specified
    in tests/SCHEME.md. Same-grid agreement with the pinned build is required. See
    judge/golden/meta.json for its version and source fingerprint.
    """
    raise NotImplementedError


def misfit_and_grad(vp, vs, rho, source_y, source_x, source_locations,
                    receiver_locations, spacing, dt, pml_width, pml_freq,
                    max_vel, observed_y, observed_x, data_scale):
    """Return the velocity least-squares objective and its exact discrete gradient.

    Forward arguments and conventions are exactly those of elastic().
    observed_y, observed_x: float32 (nshot,nreceiver,nt) arrays in m/s,
    fixed observations at the forward velocity sampling times.
    data_scale: positive Python float in m/s, held fixed in differentiation.
    Let N=nshot*nreceiver*nt. The dimensionless scalar objective is
    J=sum((vy-observed_y)^2+(vx-observed_x)^2)/(2*N*data_scale^2).
    All samples including zeros enter the sum; no dt or volume weighting.

    Return exactly:
      objective: float32 NumPy scalar, shape (), dimensionless;
      grad_vp, grad_vs: float32 (ny,nx), units s/m;
      grad_rho: float32 (ny,nx), units m^3/kg;
      grad_source_y, grad_source_x: float32 (nshot,nsource,nt), m^3/N.
    Gradients include conversion to Lame parameters, harmonic shear averaging,
    arithmetic density averaging, force scaling, and edge replication into
    C-PML. pml_freq, max_vel, geometry, dt and observations are held fixed.
    Differentiate the uncompressed discrete calculation, every time step.
    Vacuum cells are fixed and their three material gradients are exactly
    zero. Tests use positive solid cells for finite differences. Return the
    mathematical derivative, with no optimizer/preconditioner scaling.
    """
    raise NotImplementedError
