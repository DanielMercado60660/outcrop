"""Trace misfits and their MLX autodifferentiated data gradients."""

import mlx.core as mx
import numpy as np


def misfit_and_grad(predicted, observed, dt, kind, epsilon):
    """Evaluate the rectangular-time integral and its predicted-data gradient."""
    x = mx.array(predicted, dtype=mx.float32)
    y = mx.array(observed, dtype=mx.float32)
    scale = float(dt) / (2.0 * predicted.shape[0])

    if kind == "l2":
        def objective(p):
            residual = p - y
            return scale * mx.sum(residual * residual)
    elif kind == "envelope":
        n = predicted.shape[-1]
        # Construct the analytic signal: retain DC and Nyquist, double
        # strictly positive frequencies, and discard negative frequencies.
        mask = np.zeros(n, dtype=np.float32)
        mask[0] = 1.0
        mask[1:(n + 1) // 2] = 2.0
        if n % 2 == 0:
            mask[n // 2] = 1.0
        multiplier = mx.array(mask)

        def hilbert(a):
            spectrum = mx.fft.fft(a, axis=-1) * multiplier
            return mx.imag(mx.fft.ifft(spectrum, axis=-1))

        def objective(p):
            # Evaluate BOTH transforms and envelopes in one batch so that
            # identical and polarity-flipped inputs give identical envelopes.
            a = mx.concatenate([p, y], axis=0)
            h = hilbert(a)
            # A scaled three-component norm avoids squaring large amplitudes
            # or squaring tiny epsilon. At least one normalized component has
            # magnitude one, including at silent samples, so sqrt stays smooth.
            norm_scale = mx.stop_gradient(
                mx.maximum(mx.maximum(mx.abs(a), mx.abs(h)), float(epsilon))
            )
            real = a / norm_scale
            imag = h / norm_scale
            smooth = float(epsilon) / norm_scale
            unit_envelope = mx.sqrt(
                real * real + imag * imag + smooth * smooth
            )
            # Holding the numerical scale fixed preserves the norm's exact
            # derivative: scale cancels in dE/da and dE/dH(a).
            ntrace = p.shape[0]
            sp, sy = norm_scale[:ntrace], norm_scale[ntrace:]
            pair_scale = mx.maximum(sp, sy)
            rp, ry = p / pair_scale, y / pair_scale
            hp = h[:ntrace] / pair_scale
            hy = h[ntrace:] / pair_scale
            # E(p)-E(y) = (p^2-y^2 + H(p)^2-H(y)^2)/(E(p)+E(y)).
            # Factor the numerator before adding epsilon or taking square
            # roots, so weak signals do not subtract two rounded epsilons.
            # Normalize both sides first to avoid overflow in the squares.
            energy_difference = (rp - ry) * (rp + ry) + (hp - hy) * (hp + hy)
            # Keep the individual norm scales: even a silent trace paired
            # with a much larger trace then has a nonsingular sqrt derivative.
            # This dimensionless denominator is >= 1, so its reciprocal is safe.
            denominator = (
                (sp / pair_scale) * unit_envelope[:ntrace]
                + (sy / pair_scale) * unit_envelope[ntrace:]
            )
            residual = pair_scale * (energy_difference / denominator)
            return scale * mx.sum(residual * residual)
    else:
        raise ValueError("kind must be 'l2' or 'envelope'")

    value, gradient = mx.value_and_grad(objective)(x)
    mx.eval(value, gradient)
    return {
        "objective": np.asarray(value, dtype=np.float32),
        "gradient": np.asarray(gradient, dtype=np.float32),
    }
