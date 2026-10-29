"""Project 17 public API, identical for MLX and Rust/Metal.

Both functions use the SAME discrete Born operator J_v. Axis order is (z,x),
z positive down, x right, origin at the upper-left physical grid node. Grid
indices exclude the exterior PML. No free surface. All wavefields initially
zero. Spatial accuracy is fixed at fourth order (five-point 1-D second
derivatives summed in 2-D), time accuracy at second order. Use Deepwave scalar
Born conventions, including source injection -v**2*dt**2*f and its velocity
derivative. J_v maps a velocity perturbation in m/s to the scattered scalar
wavefield sampled at receivers. No direct/background arrivals are included.

PML matches Deepwave scalar_born (CFS-PML, edge-replicated v and scatter,
width on each exterior side, zero outer FD halo). The transpose includes the
adjoint of padding: contributions in replicated scatter cells fold back into
the physical edge cells. max_vel=max(velocity), survey_pad=None (full domain),
model_gradient_sampling_interval=1, no compression, interpolation, time or
frequency tapers. Tests use stable dt requiring no internal substeps. Receiver
sample timing must match Deepwave, including its initial sample convention.

Implementations use float32; locations are int64. J_v^T is the Euclidean
transpose over ALL shots, receivers and samples: no dt, cell area, shot count
or trace count normalization. Do not mute, whiten, smooth, normalize by
illumination, take an absolute value, or otherwise postprocess the image.
Only the derivative with respect to scatter is required; velocity is fixed.
"""

import numpy as np


def rtm(
    velocity: np.ndarray,
    source_amplitudes: np.ndarray,
    source_locations: np.ndarray,
    receiver_locations: np.ndarray,
    data: np.ndarray,
    grid_spacing: list[float],
    dt: float,
    pml_width: int,
    pml_freq: float,
) -> dict[str, np.ndarray]:
    """Migrate scattered data by applying the discrete Born adjoint.

    Args:
        velocity: float32 (nz,nx), positive background speed, m/s.
        source_amplitudes: float32 (nshot,nsrc,nt), scalar PDE source f.
            Scalar wavefield units are arbitrary U; f has units U/m**2;
            injection is -v**2*dt**2*f, without cell-area scaling.
        source_locations: int64 (nshot,nsrc,2), integer (z,x) node indices.
            Each shot has independent, simultaneously firing sources.
        receiver_locations: int64 (nshot,nrec,2), integer (z,x) indices.
            Each shot has nrec distinct receiver nodes; order is preserved.
        data: float32 (nshot,nrec,nt), scattered wavefield, U, t=k*dt.
            May contain arbitrary signed values; already stripped of direct
            arrivals. Source and data time axes have the same length.
        grid_spacing: Python list [dz,dx], positive node spacing, metres.
        dt: Python float, positive sample interval, seconds.
        pml_width: Python int, number of added PML cells on EVERY side.
        pml_freq: Python float, positive PML design frequency, Hz.

    Returns:
        {"image": float32 (nz,nx)}: signed J_v^T data, units U**2/(m/s).
        This is minus the gradient of 0.5*sum((J_v scatter-data)**2) at
        scatter=0. Shot contributions are summed. Includes all physical
        cells and excludes added PML/FD halo. See module for discretization.
    """
    raise NotImplementedError


def born_misfit_and_grad(
    velocity: np.ndarray,
    scatter: np.ndarray,
    source_amplitudes: np.ndarray,
    source_locations: np.ndarray,
    receiver_locations: np.ndarray,
    data: np.ndarray,
    grid_spacing: list[float],
    dt: float,
    pml_width: int,
    pml_freq: float,
) -> dict[str, np.ndarray]:
    """Return Born prediction, least-squares objective and scatter gradient.

    Args:
        velocity: float32 (nz,nx), fixed background speed, m/s.
        scatter: float32 (nz,nx), signed additive velocity perturbation,
            m/s, the linearization direction dv, NOT squared slowness or
            dimensionless reflectivity. The forward map is linear in this.
        source_amplitudes: float32 (nshot,nsrc,nt), source, U/m**2.
        source_locations: int64 (nshot,nsrc,2), (z,x) node indices.
        receiver_locations: int64 (nshot,nrec,2), (z,x) node indices.
        data: float32 (nshot,nrec,nt), observed scattered wavefield, U.
        grid_spacing: Python list [dz,dx], metres.
        dt: Python float, time sample interval, seconds.
        pml_width: Python int, added PML cells on every side.
        pml_freq: Python float, PML design frequency, Hz.
        All conventions and constraints are identical to rtm above.

    Returns:
        {"predicted": float32 (nshot,nrec,nt), U,
         "loss": float32 scalar ndarray (), U**2,
         "gradient": float32 (nz,nx), U**2/(m/s)}.
        predicted = J_v scatter; loss = 0.5*sum((predicted-data)**2);
        gradient = J_v^T (predicted-data), with no normalization.
        MLX uses autodiff; Metal uses the discrete adjoint. No derivatives
        with respect to background velocity are requested by this API.
    """
    raise NotImplementedError
