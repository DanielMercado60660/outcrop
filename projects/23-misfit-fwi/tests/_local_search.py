"""Deterministic local arrival inversion, using only saved forward traces."""

import numpy as np

from .tolerances import (
    SEARCH_BISECTIONS,
    SEARCH_MAX_STEPS,
    SEARCH_STEP_SECONDS,
    SEARCH_STOP_SECONDS,
)


def forward(g, delay):
    """Piecewise-linear interpolation of the saved analytic forward dictionary."""
    axis = g["delays"]
    i = int(np.clip(np.searchsorted(axis, delay, side="right") - 1, 0, len(axis) - 2))
    delta = axis[i + 1] - axis[i]
    derivative = (g["traces"][i + 1] - g["traces"][i]) / delta
    trace = g["traces"][i] + (delay - axis[i]) * derivative
    return trace[None].astype(np.float32), derivative[None]


def recover(g, initial, evaluate):
    """Follow local descent to the first stationary point; never global-search."""
    def slope(delay):
        trace, derivative = forward(g, delay)
        gradient = evaluate(trace)["gradient"]
        return float(np.sum(gradient.astype(np.float64) * derivative))

    current = float(initial)
    grad = slope(current)
    if grad == 0:
        return current
    step = -np.sign(grad) * SEARCH_STEP_SECONDS
    for _ in range(SEARCH_MAX_STEPS):
        candidate = current + step
        if candidate < g["delays"][0] or candidate > g["delays"][-1]:
            return current
        next_grad = slope(candidate)
        if grad * next_grad <= 0:
            left, right = sorted((current, candidate))
            left_grad = slope(left)
            for _ in range(SEARCH_BISECTIONS):
                middle = (left + right) / 2
                mid_grad = slope(middle)
                if left_grad * mid_grad <= 0:
                    right = middle
                else:
                    left, left_grad = middle, mid_grad
                if right - left <= SEARCH_STOP_SECONDS:
                    break
            return (left + right) / 2
        current, grad = candidate, next_grad
    raise AssertionError("local inversion exhausted its fixed search budget")
