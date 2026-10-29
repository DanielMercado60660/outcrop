"""Factored EikoNet and physics-only optimization in MLX."""

import math

import mlx.core as mx


def _field(parameters, sources, receivers, origin, length_scale,
           reference_velocity, width, accurate=True):
    dimension = sources.shape[1]
    h = width
    k = 2 * dimension * h
    w1 = parameters[:k].reshape(2 * dimension, h)
    b1 = parameters[k:k + h]
    k += h
    w2 = parameters[k:k + h * h].reshape(h, h)
    k += h * h
    b2 = parameters[k:k + h]
    k += h
    w3 = parameters[k:k + h]
    b3 = parameters[k + h]

    x = mx.concatenate(((sources - origin) / length_scale,
                        (receivers - origin) / length_scale), axis=1)
    def product(left, right):
        if accurate:
            # Keep float32 reductions without an N-by-H-by-H temporary.
            # Each broadcast has at most eight reduction entries, so its
            # storage is O(NH), including when H or the query batch grows.
            result = mx.sum(left[:, :8, None] * right[None, :8, :], axis=1)
            for start in range(8, left.shape[1], 8):
                result = result + mx.sum(
                    left[:, start:start + 8, None]
                    * right[None, start:start + 8, :], axis=1)
            return result
        return left @ right

    a = mx.tanh(product(x, w1) + b1)
    b = mx.tanh(product(a, w2) + b2)
    f = mx.sum(b * w3, axis=1) + b3 if accurate else b @ w3 + b3

    # df/dr retains its dependence on all parameters for mixed derivatives.
    db = (1.0 - b * b) * w3
    da = product(db, w2.T) * (1.0 - a * a)
    df = product(da, w1[dimension:, :].T) / length_scale
    delta = receivers - sources
    # Scaling before squaring preserves both tiny and large separations.
    scale = mx.max(mx.abs(delta), axis=1)
    noncoincident = scale > 0
    safe_scale = mx.where(noncoincident, scale, mx.ones_like(scale))
    scaled_delta = delta / safe_scale[:, None]
    scaled_distance = mx.sqrt(mx.sum(scaled_delta * scaled_delta, axis=1))
    distance = scale * scaled_distance
    safe_scaled_distance = mx.where(
        noncoincident, scaled_distance, mx.ones_like(scaled_distance))
    direction = scaled_delta / safe_scaled_distance[:, None]
    # Mask before exp as well as at the outputs: even exp(1000) must not
    # turn the algebraic zero at a coincident pair into zero times infinity.
    factor = mx.exp(mx.where(noncoincident, f, mx.zeros_like(f))) / reference_velocity
    time = mx.where(noncoincident, distance * factor, mx.zeros_like(distance))
    gradient = mx.where(
        noncoincident[:, None],
        factor[:, None] * (direction + distance[:, None] * df),
        mx.zeros_like(delta))
    return time, gradient


def _objective(parameters, sources, receivers, velocities, origin,
               length_scale, reference_velocity, width, accurate=True):
    _, gradient = _field(parameters, sources, receivers, origin, length_scale,
                         reference_velocity, width, accurate)
    residual = velocities * velocities * mx.sum(gradient * gradient, axis=1) - 1.0
    return mx.mean(residual * residual)


def evaluate(parameters, sources, receivers, origin, length_scale,
             reference_velocity, width):
    time, gradient = _field(mx.array(parameters), mx.array(sources),
                            mx.array(receivers), mx.array(origin), length_scale,
                            reference_velocity, width)
    return {"time": time, "receiver_gradient": gradient}


def physics_loss_and_grad(parameters, sources, receivers, velocities, origin,
                          length_scale, reference_velocity, width):
    s, r, v, o = map(mx.array, (sources, receivers, velocities, origin))

    def loss(p):
        return _objective(p, s, r, v, o, length_scale, reference_velocity, width)

    value, gradient = mx.value_and_grad(loss)(mx.array(parameters))
    return {"loss": value, "gradient": gradient}


def train(initial_parameters, sources, receivers, velocities, origin,
          length_scale, reference_velocity, width, steps, learning_rate):
    s, r, v, o = map(mx.array, (sources, receivers, velocities, origin))
    p = mx.array(initial_parameters)

    def loss(parameters):
        return _objective(parameters, s, r, v, o, length_scale,
                          reference_velocity, width, accurate=False)

    value_grad = mx.value_and_grad(loss)
    initial_loss = loss(p)
    mx.eval(initial_loss)
    # Avoid amplifying roundoff by Adam at an already exact solution.
    if initial_loss.item() < 1e-12:
        return {"parameters": p}

    @mx.compile
    def update(parameters, first, second, rate, correction1, correction2):
        _, grad = value_grad(parameters)
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad * grad
        parameters = parameters - rate * (first / correction1) / (
            mx.sqrt(second / correction2) + 1e-8)
        return parameters, first, second

    first = mx.zeros_like(p)
    second = mx.zeros_like(p)
    for i in range(steps):
        # Cosine annealing from the supplied rate to one percent of that rate.
        rate = learning_rate * (0.01 + 0.99 *
                                (1.0 + math.cos(math.pi * i / steps)) / 2.0)
        p, first, second = update(p, first, second, mx.array(rate),
                                  mx.array(1.0 - 0.9 ** (i + 1)),
                                  mx.array(1.0 - 0.999 ** (i + 1)))
        mx.eval(p, first, second)
    return {"parameters": p}
