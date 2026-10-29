"""One-dimensional magnetotelluric response for a layered earth."""

import math

import mlx.core as mx
import numpy as np


def _divide(ar, ai, br, bi):
    """Divide complex numbers represented by real and imaginary parts."""
    scale = mx.maximum(mx.abs(br), mx.abs(bi))
    ar, ai, br, bi = ar / scale, ai / scale, br / scale, bi / scale
    denominator = br * br + bi * bi
    return (ar * br + ai * bi) / denominator, (ai * br - ar * bi) / denominator


def _input(value, name):
    """Validate host inputs while allowing MLX arrays through autodiff."""
    if isinstance(value, mx.array):
        array = value
    else:
        host = np.asarray(value)
        if host.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        if not np.all(np.isfinite(host)) or np.any(host <= 0):
            raise ValueError(f"{name} must contain positive finite values")
        array = mx.asarray(host, dtype=mx.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array.astype(mx.float32)


def mt1d(resistivity, thickness, frequency):
    rho = _input(resistivity, "resistivity")
    depth = _input(thickness, "thickness")
    freq = _input(frequency, "frequency")
    if rho.shape[0] == 0:
        raise ValueError("resistivity must contain at least one layer")
    if depth.shape[0] != rho.shape[0] - 1:
        raise ValueError("thickness must contain one value per finite layer")

    # Scale impedances by sqrt(omega * mu0 / 2). The characteristic
    # impedance of each layer is then sqrt(rho) * (1 + i).
    zr = mx.sqrt(rho[-1]) * mx.ones_like(freq)
    zi = zr

    for layer in range(rho.shape[0] - 2, -1, -1):
        characteristic = mx.sqrt(rho[layer])
        rr, ri = _divide(
            characteristic - zr, characteristic - zi,
            characteristic + zr, characteristic + zi,
        )

        # Clamp before trigonometry, even if the product overflowed. At
        # kh=50 the reflection is negligible in float32.
        kh = depth[layer] * (mx.sqrt(freq) / characteristic) * math.sqrt(
            math.pi * 1.25663706212e-6
        )
        kh = mx.minimum(kh, 50.0)
        decay = mx.exp(-2.0 * kh)
        er = decay * mx.cos(2.0 * kh)
        ei = -decay * mx.sin(2.0 * kh)
        qr = rr * er - ri * ei
        qi = rr * ei + ri * er
        tr, ti = _divide(1.0 - qr, -qi, 1.0 + qr, qi)
        zr = characteristic * (tr - ti)
        zi = characteristic * (tr + ti)

    return {
        "apparent_resistivity": zr * (zr * 0.5) + zi * (zi * 0.5),
        "phase": mx.arctan2(zi, zr) * (180.0 / math.pi),
    }
