"""Headless project 50 API. Metal only.

All arrays are C contiguous. A single shot is used. y increases down the screen,
x increases right; physical cell (0,0) is at (y,x)=(0,0) metres. Pressure is a
scalar in arbitrary amplitude units (AU), constant density, positive source RHS
following Deepwave scalar: u_tt = v**2 * (Laplacian(u) - f). Source injection is
-v[y,x]**2 * dt**2 * source_amplitudes[s,t], without a cell-area divisor.
All models are float32 at the API, including the gradient. Location arrays are
int64 cell indices. Python scalar parameters keep the values passed in JSON.
No reference libraries, host wave simulation, or host rasterization in the
implementation. The texture must be a real GPU render target readable headlessly.

The normative, self-contained discrete operator is tests/SCHEME.md, relative
to the project root. It specifies every stencil coefficient, CFS-PML profile,
memory recurrence/time level, interior projection and outer-edge closure.
Implementations do not need to read Deepwave source or infer its state layout.
"""


def run_frames(*, velocity, state, source_amplitudes, source_locations,
               receiver_locations, grid_spacing, dt, accuracy, pml_width,
               pml_freq, max_velocity, boundary, render_scale):
    """Advance and render EVERY input time sample on the GPU.

    Parameters
    ----------
    velocity : float32 ndarray (ny,nx)
        Positive acoustic speed, m/s. Extend by nearest physical edge into PML.
    state : float32 ndarray (6,ny+2*pml_width,nx+2*pml_width)
        Input state at n=0: [u^0,u^-1,psi_y^0,psi_x^0,zeta_y^0,zeta_x^0].
        No FD halo, no shot axis. u is AU, psi is AU/m, zeta is AU/m².
        psi^n and zeta^n are OLD memories consumed alongside u^n by the
        recurrence in SCHEME.md; all six channels advance together. The
        returned channels are [u^nt,u^(nt-1),psi_y^nt,psi_x^nt,zeta_y^nt,
        zeta_x^nt]. Zero all channels to start at rest. Each memory is forced
        to zero throughout the physical interior along its own axis, on
        input and output. The hidden accuracy/2 guard cells beyond the PML
        stay zero for all fields (periodic mode wraps pressure instead).
    source_amplitudes : float32 ndarray (ns,nt)
        RHS f, AU/m², sampled at n*dt. nt>=1. ns>=1, unique source locations;
        all-zero amplitudes represent a source-free interval.
    source_locations : int64 ndarray (ns,2)
        Physical [y,x] cell indices, no PML offset, no interpolation.
    receiver_locations : int64 ndarray (nr,2)
        Physical [y,x] cell indices, nr>=1. Also the texture probe positions.
    grid_spacing : list of two floats
        [dy,dx] in metres.
    dt : float
        Seconds, fixed and CFL-safe for max_velocity; no internal substepping.
    accuracy : int
        Spatial central FD order, one of 2,4,6,8; time is second-order leapfrog.
    pml_width : int
        Cells on EACH side, outside the physical grid (0 for periodic).
    pml_freq : float
        Hz, fixed CFS-PML frequency; alpha=pi*pml_freq*(1-q).
        SCHEME.md defines q, sigma (R=0.001, power=2), kappa=1 and a,b.
    max_velocity : float
        m/s, fixed upper bound for CFL/PML setup, independent of model gradient.
    boundary : str
        'pml' selects the explicit CFS-PML in SCHEME.md; 'periodic' wraps
        both axes with pml_width=0 and zero memories. Full model extent,
        survey_pad=None, origin=0. Periodic mode is required for refinement.
    render_scale : float
        Positive AU, fixed symmetric display gain; no per-frame autoscaling.

    Returns
    -------
    dict with these float32 ndarrays:
        state : (6,ny+2*pml_width,nx+2*pml_width), state after nt updates.
        receiver_amplitudes : (nr,nt), AU; gather u_n BEFORE injection/update,
            matching Deepwave sampling at n=0,...,nt-1.
        rgba : (ny,nx,4), readback of a Metal RGBA32Float (MTLPixelFormat
            RGBA32Float / rgba32float) render target at u_nt, top-left origin.
            q=clip(u/render_scale,-1,1); RGBA=(max(q,0),0,max(-q,0),1).
            Physical domain only, no PML; no gamma/lighting/interpolation.
            Render and store all channels at float32 precision, with blending,
            sRGB conversion and multisampling disabled. This headless target
            has no intermediate unorm8 or float16 quantization; an app may
            separately convert it to its display drawable format.
        texture_samples : (nt,nr,4), same RGBA read from the rendered texture
            AFTER each update, sampled at receiver_locations; tests check all
            frames so returning only a final rendering does not suffice.

    Render parity propagates the project-wide pressure error budget through the
    colormap gain 1/render_scale, plus float32 colormap rounding (see
    tests/tolerances.py). The pressure peak is over the final physical field
    for rgba, and all post-update probe pressures for texture_samples.
    Green/alpha constants, range and red/blue exclusivity are checked exactly.

    Allocate pipelines/buffers once per invocation; retain state and textures on
    GPU through the loop, render each frame, synchronize before return. The
    performance judge times the entire call (including process startup and
    transport) on 1000x1000 physical cells for 240 frames, without subtracting
    overhead. Do not return internal timing as proof of throughput.
    """
    raise NotImplementedError


def misfit_and_grad(*, velocity, state, source_amplitudes, source_locations,
                    receiver_locations, grid_spacing, dt, accuracy, pml_width,
                    pml_freq, max_velocity, boundary, observed):
    """Live FWI L2 objective and exact discrete velocity gradient.

    velocity (float32 (ny,nx), m/s), state (float32 (6,ny+2p,nx+2p), AU
    and derivative memories), source_amplitudes (float32 (ns,nt), AU/m²),
    source_locations (int64 (ns,2), [y,x] physical cells), receiver_locations
    (int64 (nr,2), physical cells), grid_spacing (list [dy,dx], metres), dt
    (float, seconds), accuracy (int, 2/4/6/8), pml_width (int cells per side),
    pml_freq (float Hz), max_velocity (float m/s), and boundary (str 'pml' or
    'periodic') have exactly run_frames' conventions and recurrence. The state
    and source are held fixed while differentiating. All tests use zero state.
    observed is float32 (nr,nt), AU, sampled BEFORE each update like receivers.

    Return a dict of float32 arrays: objective (scalar shape (), AU²) equals
    0.5*sum((predicted-observed)**2), without normalization/regularization;
    this is project 16's mean objective multiplied by nr*nt for one shot.
    gradient (ny,nx), AU²/(m/s), is d(objective)/d(velocity), including velocity
    dependence of source injection and edge extension into PML. PML profile,
    max_velocity, dt and all other inputs remain fixed. Metal uses the exact
    discrete adjoint; an optional MLX implementation must use autodiff.
    """
    raise NotImplementedError
