"""Implementation contract for project 42."""


def fk_filter(gather, dt, dx, velocity_cutoff):
    """Remove low apparent-velocity energy using a hard 2-D Fourier mask.

    gather: float32 array (nx, nt), amplitudes arbitrary, x and time ascending.
    dt: scalar seconds; dx: scalar meters; velocity_cutoff: positive m/s.
    Return {'filtered': float32 array (nx, nt)}, same units and axes. Use the
    unpadded periodic DFT with frequencies f in Hz and k in cycles/m. Remove
    bins satisfying |f| <= velocity_cutoff * |k|, including DC. Keep all
    other bins and take the real part of the inverse transform.
    """
    raise NotImplementedError


def radon_demultiple(gather, offsets, dt, slowness):
    """Remove one parabolic Radon component by frequency-domain projection.

    gather: float32 (nx, nt), arbitrary amplitude, time ascending. offsets:
    float32 (nx,), meters from the midpoint; dt: scalar seconds; slowness:
    scalar seconds/meter**2. Return {'filtered': float32 (nx, nt)}, same units.
    For each unpadded periodic DFT frequency f, define a[x]=exp(-2*pi*i*f*
    slowness*offsets[x]**2). Subtract a*(a^H d)/nx from the spatial frequency
    vector d. Apply this to every frequency, including DC, then inverse FFT
    in time and return its real part. This is an orthogonal least-squares
    projection; no padding, taper, regularization, or intercept shifts.
    """
    raise NotImplementedError
