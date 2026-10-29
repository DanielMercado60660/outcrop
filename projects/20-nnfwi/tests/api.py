"""Frozen NumPy boundary for project 20 (MLX only).

No reference libraries may be imported by an implementation. All floating inputs
and outputs use float32; grid indices use int64. Arrays are C contiguous. Every
function returns a dict of NumPy arrays, including rank-zero objective arrays.
The argument list also permits the harness's Metal transport if a Metal path is
added later; currently only MLX is tested.
"""
from __future__ import annotations

import numpy as np


def misfit_and_grad(
    *, parameters: np.ndarray, latent: np.ndarray,
    initial_velocity: np.ndarray, observed: np.ndarray,
    source_amplitudes: np.ndarray, source_locations: np.ndarray,
    receiver_locations: np.ndarray, spacing: float, dt: float,
    data_scale: float, velocity_scale: float, coarse_shape: list[int],
    hidden_size: int, parameterization: str,
) -> dict[str, np.ndarray]:
    """Evaluate the normalized acoustic L2 objective and its parameter gradient.

    Parameters
    ----------
    parameters : float32 (P,)
        Dimensionless trainable vector, row-major layout. For ``grid``,
        P=Nz*Nx and q=reshape(parameters, (Nz,Nx)). For ``neural``, let
        L=len(latent), H=hidden_size, C=coarse_shape[0]*coarse_shape[1];
        P=H*L+H+C*H+C. Consecutive blocks are W1(H,L), b1(H), W2(C,H), b2(C).
        h=tanh(W1@latent/sqrt(L)+b1); coarse=W2@h/sqrt(H)+b2, reshaped
        to coarse_shape. Bilinearly resize to (Nz,Nx), with align_corners=True,
        to obtain q. Thus output node (i,j) samples the coarse image at
        (i*(Cz-1)/(Nz-1), j*(Cx-1)/(Nx-1)); no padding or antialiasing.
        In both modes v=initial_velocity+velocity_scale*tanh(q).
    latent : float32 (L,)
        Fixed dimensionless network input; ignored in grid mode. Never train it.
    initial_velocity : float32 (Nz,Nx), m/s
        Positive starting/background velocity, physical cells only. Axis 0 is
        depth z increasing down; axis 1 is horizontal x increasing right. Cell
        (0,0) is the spatial origin, coordinates are (z*spacing,x*spacing).
    observed : float32 (S,R,T), scalar wavefield units
        The judge's saved noisy receiver gathers; include every sample in loss.
    source_amplitudes : float32 (S,Q,T), wavefield units/m^2
        Saved RHS f of laplacian(u)-v**(-2)*u_tt=f. Point injection is
        -v_source**2*dt**2*f at the source cell, with no cell-area normalization.
        Time samples are n*dt. Source scaling depends on velocity and MUST be
        differentiated. Wavefields and all PML memory initially zero.
    source_locations : int64 (S,Q,2), zero-based cell indices
        (z,x) integer locations relative to the physical-grid origin, per shot.
    receiver_locations : int64 (S,R,2), zero-based cell indices
        (z,x) positions, sampled in exactly Deepwave scalar's time convention.
    spacing : float, m
        Uniform spacing along both axes.
    dt : float, s
        Source/receiver time interval and propagation time step; all locked
        cases satisfy CFL without internal resampling (step ratio 1).
    data_scale : float, wavefield units
        Positive fixed normalization supplied by the judge. Never recompute
        from observed or predicted gathers or differentiate through it.
    velocity_scale : float, m/s
        Positive amplitude of the bounded perturbation; 500 in these fixtures.
    coarse_shape : list[int] of length 2, cell counts
        Neural image (Cz,Cx), both >=2; ignored by grid mode. Locked value [5,7].
    hidden_size : int
        Hidden layer width, ignored by grid mode; locked value 8 (L=4).
    parameterization : str
        Exactly ``grid`` (plain FWI) or ``neural`` (generative prior).

    Discrete acoustic operator
    --------------------------
    Match Deepwave scalar: second order in time, fourth order in space, PML of
    12 cells on every side, edge-replicated velocity in the PML, pml_freq=18 Hz,
    and fixed max_vel=2600 m/s. No survey cropping, resampling, time/frequency
    taper, compression, explicit regularization or gradient clipping.
    Differentiate every time step (model_gradient_sampling_interval=1),
    including replicated PML velocities. PML profiles and time step stay fixed
    as parameters vary. Project 13 owns the broader stencil/convergence tests.

    Returns
    -------
    dict with exactly the following float32 NumPy arrays:
        objective : ()
            Dimensionless J=0.5*mean(((predicted-observed)/data_scale)**2).
            Mean includes S*R*T samples. No regularizer or extra scaling.
        gradient : (P,)
            dJ/dparameters in the flattened input order, via MLX autodiff.
        predicted : (S,R,T)
            Receiver wavefield in the same units and order as observed.
        velocity : (Nz,Nx), m/s
            Decoded physical-grid model, excluding PML cells.
    """
    raise NotImplementedError


def invert(
    *, parameters: np.ndarray, latent: np.ndarray,
    initial_velocity: np.ndarray, observed: np.ndarray,
    source_amplitudes: np.ndarray, source_locations: np.ndarray,
    receiver_locations: np.ndarray, spacing: float, dt: float,
    data_scale: float, velocity_scale: float, coarse_shape: list[int],
    hidden_size: int, parameterization: str, steps: int, learning_rate: float,
) -> dict[str, np.ndarray]:
    """Run a fixed Adam inversion using the objective in misfit_and_grad.

    Arguments (all arrays C contiguous)
    ----------------------------------
    parameters : float32 (P,), dimensionless
        Initial trainable vector. P=Nz*Nx for grid, or H*L+H+C*H+C for neural.
        Layout, decoder, output velocity bounds and acoustic conventions are
        exactly those in misfit_and_grad. Copy it; do not mutate caller inputs.
    latent : float32 (L,), dimensionless
        Fixed latent input; ignored for grid. L=4 in the fixtures.
    initial_velocity : float32 (Nz,Nx), m/s
        Fixed background, with (depth z, horizontal x) axes and origin (0,0).
    observed : float32 (S,R,T), wavefield units
        The same saved noisy gathers for both parameterizations. Neither truth
        nor clean data is an input to either inversion.
    source_amplitudes : float32 (S,Q,T), wavefield units/m^2
        Fixed RHS of laplacian(u)-v**(-2)*u_tt=f, samples at n*dt.
    source_locations : int64 (S,Q,2), zero-based (z,x) cell indices
        Fixed point sources on the physical grid, one set per shot.
    receiver_locations : int64 (S,R,2), zero-based (z,x) cell indices
        Fixed receivers on the physical grid, one set per shot.
    spacing : float, m
        Uniform grid spacing in both directions.
    dt : float, s
        Fixed propagation/source/receiver interval; no internal resampling.
    data_scale : float, wavefield units
        Fixed positive loss normalizer, supplied by the judge.
    velocity_scale : float, m/s
        Bound amplitude of v=initial_velocity+velocity_scale*tanh(q).
    coarse_shape : list[int] of length 2
        Neural image counts (Cz,Cx); C=Cz*Cx; ignored for grid.
    hidden_size : int
        Neural hidden width H; ignored for grid.
    parameterization : str
        ``grid`` for plain FWI, ``neural`` for the generative network prior.
    steps : int
        Nonnegative update count. Locked tests use 1 and 160. Return the last
        iterate, with no early stopping, best-iterate selection or truth access.
    learning_rate : float, dimensionless
        Constant Adam parameter step size; 0.02 for both modes.

    Algorithm
    ---------
    Start m=s=0; for t=1..steps evaluate g=dJ/dp at the current p, then
    m=0.9*m+0.1*g; s=0.999*s+0.001*g*g;
    p=p-learning_rate*(m/(1-0.9**t))/(sqrt(s/(1-0.999**t))+1e-8).
    The epsilon is OUTSIDE the square root. No weight decay, clipping,
    scheduler, batch sampling, regularizer, latent updates, or randomness.
    Acoustic/PML constants and units are exactly those in misfit_and_grad.

    Returns
    -------
    dict with exactly the following float32 NumPy arrays:
        parameters : (P,), dimensionless
            Final trainable parameters, in the input layout.
        velocity : (Nz,Nx), m/s
            Decode of final parameters, excluding PML.
        loss_history : (steps+1,), dimensionless
            J before every update and once after the last update.
    """
    raise NotImplementedError
