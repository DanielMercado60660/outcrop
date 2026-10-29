"""One-dimensional periodic shallow-water propagation in MLX."""

import mlx.core as mx

_GRAVITY = 9.81


def _advance(state, dx, dt, steps, mode, depth, evaluate=False):
    """Evolve float32 state; materialize intermediate graphs only for inference."""
    if mode not in ("linear", "nonlinear"):
        raise ValueError(f"Unknown shallow-water mode: {mode}")
    if steps == 0:
        return state
    if mode == "linear":
        gh = _GRAVITY * depth
        speed = mx.sqrt(mx.array(gh, dtype=mx.float32))
    else:
        # Keep a constant reference throughout the evolution.  In particular,
        # do not round the small depth perturbation back to total depth after
        # every step.  The reference is independent of the depth argument.
        # Retain its autodiff connection: initial subtraction, pressure, and
        # final addition together describe the original total-depth map.
        reference = state[0, 0]
        state = mx.stack((state[:, 0] - reference, state[:, 1]), axis=1)

    for step in range(steps):
        surface = state[:, 0]
        q = state[:, 1]
        if mode == "linear":
            flux = mx.stack((q, gh * surface), axis=1)
            face_speed = speed
        else:
            h = reference + surface
            # The omitted constant g * reference**2 / 2 cancels at faces.
            pressure = _GRAVITY * surface * (reference + 0.5 * surface)
            flux = mx.stack((q, q * q / h + pressure), axis=1)
            cell_speed = mx.abs(q / h) + mx.sqrt(_GRAVITY * h)
            face_speed = mx.maximum(cell_speed, mx.roll(cell_speed, -1, axis=0))[:, None]

        # Entry i is face i+1/2, including the last-to-first periodic face.
        right = mx.roll(state, -1, axis=0)
        right_flux = mx.roll(flux, -1, axis=0)
        face = 0.5 * (flux + right_flux - face_speed * (right - state))
        state = state - (dt / dx) * (face - mx.roll(face, 1, axis=0))
        # Limit the size of the lazy graph during long forward runs.
        if evaluate and (step + 1) % 32 == 0:
            mx.eval(state)
    if mode == "nonlinear":
        state = mx.stack((state[:, 0] + reference, state[:, 1]), axis=1)
    return state


def propagate(initial, dx, dt, steps, mode, depth):
    """Return periodic Rusanov/Euler state in the input's units and ordering."""
    state = mx.array(initial, dtype=mx.float32)
    return {"state": _advance(state, dx, dt, steps, mode, depth, evaluate=True)}


def objective_and_grad(initial, weights, dx, dt, steps, mode, depth):
    """Differentiate the unnormalized weighted final-state sum with MLX."""
    state = mx.array(initial, dtype=mx.float32)
    weights = mx.array(weights, dtype=mx.float32)

    def objective(value):
        final = _advance(value, dx, dt, steps, mode, depth)
        return mx.sum(weights * final)

    value, gradient = mx.value_and_grad(objective)(state)
    return {"objective": value, "gradient": gradient}
