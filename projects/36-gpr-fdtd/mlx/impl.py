"""Lossless periodic two-dimensional Yee TE propagation using MLX."""

import mlx.core as mx


def propagate(epsilon_r, electric, hx, hz, dx, dz, dt, steps):
    """Advance electric and impedance-scaled magnetic fields on a Yee grid."""
    epsilon_r = mx.array(epsilon_r, dtype=mx.float32)
    electric = mx.array(electric, dtype=mx.float32)
    hx = mx.array(hx, dtype=mx.float32)
    hz = mx.array(hz, dtype=mx.float32)

    # Combine dimensional factors in host precision before rounding to float32.
    # A spacing's reciprocal can overflow float32 even for a finite c*dt/dx.
    cdt = 299792458.0 * float(dt)
    scale_x = mx.array(cdt / float(dx), dtype=mx.float32)
    scale_z = mx.array(cdt / float(dz), dtype=mx.float32)

    for _ in range(steps):
        hx = hx + scale_z * (mx.roll(electric, -1, axis=1) - electric)
        hz = hz - scale_x * (mx.roll(electric, -1, axis=0) - electric)
        scaled_curl_h = scale_z * (hx - mx.roll(hx, 1, axis=1)) - scale_x * (
            hz - mx.roll(hz, 1, axis=0)
        )
        electric = electric + scaled_curl_h / epsilon_r
        mx.eval(electric, hx, hz)

    return {"electric": electric, "hx": hx, "hz": hz}
