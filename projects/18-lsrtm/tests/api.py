"""API shared by MLX and outcrop-18 (both paths).

Grid nodes are (z,x), origin (0,0) metres, z positive down and x positive right.
No density, free surface, attenuation, filtering, data normalization, quadrature
weights, model smoothing, illumination compensation, or regularization is used.

Let A m be the discrete acoustic Born scattered receiver field for velocity
perturbation dv = mask*m. This is the derivative of scalar(v) with respect to
velocity, not squared slowness or a dimensionless reflection coefficient.
The discrete convention is second order in time and ``accuracy`` order in space,
with centered Cartesian stencils and CFS-PML on all four faces.
Spacing and PML damping are axis-specific for rectangular [dz,dx] cells.
The complete locked propagator is tests/SCHEME.md: profiles, stencils,
full PML expressions at ALL updated nodes (including the p physical nodes
next to each PML), source injection, CFL rule and exact discrete transpose.
Use that local specification when implementing; it is verified independently
in float64 NumPy during golden generation. Reference-library access is not
required. The oracle build and scheme hashes are in golden/meta.json.
Background velocity and dv are edge-replicated into the PML/FD halo; the
transpose folds all replicated dv contributions back to physical edge cells.
The main survey has an all-one mask; the small problem tests a masked interior.
There is no survey cropping. Initial fields
and PML memories are zero. max_vel=max(v), gradient sampling interval=1,
freq_taper_frac=time_pad_frac=0, time_taper=False, location_interpolation='none',
storage_mode='device', storage_compression=False. The supported domain requires
R=max(1,ceil(dt*max(v)*sqrt(dz**-2+dx**-2)/0.6))=1; other inputs are outside
this API's domain. There is no internal time resampling. Point-source injection has
the Deepwave sign/scale -v**2*dt**2*source_amplitudes (no cell-area division).
Receivers record the current field before each step, starting at time zero.

All public floating arrays and results are NumPy float32. Integer node arrays
are int64 and contain physical-grid indices excluding PML. Implementations may
use float32 internally; accumulated Euclidean sums include all shots, receivers
and samples, with no dt or cell-area factors. Every return is a dict of arrays;
scalar outputs are zero-dimensional float32 arrays. No reference-library imports
are permitted in implementations. The generator uses float64 independently.
"""

import numpy as np


def born_model(
    v: np.ndarray, model: np.ndarray, mask: np.ndarray,
    source_amplitudes: np.ndarray, source_locations: np.ndarray,
    receiver_locations: np.ndarray, spacing: list[float], dt: float,
    pml_width: int, pml_freq: float, accuracy: int,
) -> dict[str, np.ndarray]:
    """Apply the masked discrete 2D acoustic Born operator A.

    Args:
        v: float32 (nz,nx), positive background velocity, m/s.
        model: float32 (nz,nx), signed velocity perturbation, m/s.
        mask: float32 (nz,nx), 0/1 spatial support; A uses dv=mask*model.
            The inversion region may include physical-grid borders; dv padding
            is edge-replicated, and its transpose must fold edge contributions.
        source_amplitudes: float32 (nshot,nsrc,nt), prescribed source term in
            arbitrary pressure units/m**2; t[n]=n*dt. No wavelet regeneration.
        source_locations: int64 (nshot,nsrc,2), integer (z,x) node indices.
        receiver_locations: int64 (nshot,nrec,2), integer (z,x) node indices.
        spacing: two Python floats [dz,dx], metres between grid nodes.
        dt: positive Python float, seconds per sample and integration step.
        pml_width: positive int, extra PML nodes at every physical-grid face.
        pml_freq: positive float, Hz, PML reference frequency.
        accuracy: int, centered spatial FD order; this judge fixes it to 4.

    Returns:
        {"predicted": float32 (nshot,nrec,nt)}, scattered pressure in arbitrary
        pressure units, same shot/time/receiver ordering as the supplied survey.
        Background/direct arrivals are excluded. See module discretization.
    """
    raise NotImplementedError


def misfit_and_grad(
    v: np.ndarray, model: np.ndarray, mask: np.ndarray, observed: np.ndarray,
    source_amplitudes: np.ndarray, source_locations: np.ndarray,
    receiver_locations: np.ndarray, spacing: list[float], dt: float,
    pml_width: int, pml_freq: float, accuracy: int,
) -> dict[str, np.ndarray]:
    """Return Phi(m)=0.5*sum((A m-observed)**2) and its exact discrete gradient.

    Args:
        v: float32 (nz,nx), fixed background velocity, m/s.
        model: float32 (nz,nx), velocity perturbation to differentiate, m/s.
        mask: float32 (nz,nx), 0/1 support, including edges when set;
            dv=mask*model and gradient is zero wherever mask is zero.
        observed: float32 (nshot,nrec,nt), scattered pressure data in arbitrary
            pressure units. No data weights, averaging, damping, or noise model.
        source_amplitudes: float32 (nshot,nsrc,nt), supplied source term,
            arbitrary pressure units/m**2, sampled at t[n]=n*dt.
        source_locations: int64 (nshot,nsrc,2), physical-grid (z,x) indices.
        receiver_locations: int64 (nshot,nrec,2), physical-grid (z,x) indices.
        spacing: Python list [dz,dx], metres.
        dt: Python float, seconds, stable integration and sampling interval.
        pml_width: int, added PML nodes at each of the four faces.
        pml_freq: float, Hz, PML reference frequency.
        accuracy: int, spatial FD order (4 in this judge).

    Returns:
        {"objective": float32 scalar (), "gradient": float32 (nz,nx),
         "predicted": float32 (nshot,nrec,nt)}.
        Objective units are pressure**2; gradient units pressure**2/(m/s).
        Gradient is A.T@(A m-observed) in the unweighted Euclidean inner
        product. MLX uses autodiff; Metal implements the discrete adjoint.
        Conventions and boundaries are exactly those of born_model.
    """
    raise NotImplementedError


def lsrtm(
    v: np.ndarray, mask: np.ndarray, observed: np.ndarray,
    source_amplitudes: np.ndarray, source_locations: np.ndarray,
    receiver_locations: np.ndarray, spacing: list[float], dt: float,
    pml_width: int, pml_freq: float, accuracy: int, iterations: int,
) -> dict[str, np.ndarray]:
    """Solve min_m 0.5||A m-observed||_2**2 by fixed-count, zero-start CGLS.

    Args:
        v: float32 (nz,nx), background velocity in m/s, held fixed.
        mask: float32 (nz,nx), 0/1 support, including edges when set.
            Only masked pixels may vary; all other image pixels remain zero.
        observed: float32 (nshot,nrec,nt), scattered pressure data in arbitrary
            pressure units; no direct wave, data weights or normalization.
        source_amplitudes: float32 (nshot,nsrc,nt), supplied source term in
            arbitrary pressure units/m**2, at t[n]=n*dt.
        source_locations: int64 (nshot,nsrc,2), physical-grid (z,x) indices.
        receiver_locations: int64 (nshot,nrec,2), physical-grid (z,x) indices.
        spacing: Python list [dz,dx], metres.
        dt: Python float, seconds, stable integration/sampling interval.
        pml_width: int, added PML nodes on every face.
        pml_freq: float, Hz, PML reference frequency.
        accuracy: int, spatial FD order (4 in this judge).
        iterations: nonnegative int, exact number of CGLS updates (40 for the
            main pass test). No early stopping, preconditioning or damping.

    Algorithm (CGLS with two-pass full A-conjugate reorthogonalization;
    all products are ordinary Euclidean sums):
        m=0; r=d; basis=[] (saved normalized p,q pairs).
        Repeat iterations times:
            p=A.T@r; q=A@p;
            Twice, in insertion order, for each (p_i,q_i) in basis:
                beta=sum(q_i*q); p=p-beta*p_i; q=q-beta*q_i.
            scale=sqrt(sum(q*q)); p=p/scale; q=q/scale;
            alpha=sum(r*q); m=m+alpha*p; append (p,q) to basis;
            r=d-A@m.
        This gives the CGLS iterates in exact arithmetic. Reorthogonalization
        avoids the severe loss of conjugacy seen in float32 ordinary CGLS on
        this problem, preserving the project-wide 1e-4 image parity criterion.
        Recompute the true residual after each update as written. For zero
        observed data return all-zero arrays, including history. Otherwise
        history[0]=1. No test exercises a nonzero problem after exact breakdown.

    Returns:
        {"image": float32 (nz,nx), velocity perturbation in m/s;
         "rtm": float32 (nz,nx), unscaled plain-RTM image A.T@observed;
         "predicted": float32 (nshot,nrec,nt), A@image, pressure units;
         "residual_history": float32 (iterations+1,), ||d-A@m_k||_2/||d||_2}.
        The RTM array uses the discrete Born imaging condition, with units
        pressure**2/(m/s); its global scale is immaterial for envelope FWHM.
        Resolution is measured by tests/metrics.py on full-column analytic-
        signal envelopes, with no padding, before selecting depth windows.
        An optimally
        scaled RTM is the first CGLS step and supplies the residual baseline.
        All conventions are exactly those of born_model and misfit_and_grad.
    """
    raise NotImplementedError
