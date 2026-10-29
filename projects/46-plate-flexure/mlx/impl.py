"""Closed-form flexure of an elastic plate on a Winkler foundation."""

import mlx.core as mx


def plate_flexure(x, rigidity, foundation, load, plate):
    """Return downward-positive deflection due to a vertical line force."""
    x = mx.array(x, dtype=mx.float32)
    rigidity = mx.array(rigidity, dtype=mx.float32)
    foundation = mx.array(foundation, dtype=mx.float32)
    load = mx.array(load, dtype=mx.float32)

    # alpha**4 = 4 D / k. Take fourth roots before dividing so that
    # the intermediate D/k cannot overflow (e.g. D=1e38, k=0.1).
    alpha = (2.0 ** 0.5) * (
        mx.sqrt(mx.sqrt(rigidity)) / mx.sqrt(mx.sqrt(foundation))
    )
    if plate == "continuous":
        t = mx.abs(x) / alpha
        deflection = (load / (2.0 * foundation * alpha)) * mx.exp(-t) * (
            mx.cos(t) + mx.sin(t)
        )
    elif plate == "broken":
        t = x / alpha
        deflection = (2.0 * load / (foundation * alpha)) * mx.exp(-t) * mx.cos(t)
    else:
        raise ValueError(f"Unknown plate type: {plate!r}")
    return {"deflection": deflection}
