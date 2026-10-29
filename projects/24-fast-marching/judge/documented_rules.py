"""Independent float64 transcription of tests/api.py, for oracle self-checks.

This does not call scikit-fmm or second_order.py. Initialization uses array
slices; marching uses Python floats (C doubles) and C-order flat indices.
Both orders deliberately retain their separately documented conventions.
"""
from heapq import heappop, heappush
from math import inf, prod, sqrt

import numpy as np


def travel_time(phi, speed, spacing, order):
    """Return float64 travel times using exactly the public API rules."""
    assert order in (1, 2)
    phi = np.asarray(phi, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    shape = phi.shape
    inverse_distance2 = np.zeros(shape, dtype=np.float64)
    for axis, step in enumerate(spacing):
        nearest = np.full(shape, np.inf)
        for direction in (-1, 1):
            here, there = [slice(None)] * phi.ndim, [slice(None)] * phi.ndim
            here[axis] = slice(1, None) if direction == -1 else slice(None, -1)
            there[axis] = slice(None, -1) if direction == -1 else slice(1, None)
            here, there = tuple(here), tuple(there)
            a, b = phi[here], phi[there]
            crossing = a * b < 0
            if order == 2:
                crossing |= (a != 0) & (b == 0)
            distance = np.full(a.shape, np.inf)
            distance[crossing] = step * np.abs(a[crossing] / (a[crossing] - b[crossing]))
            nearest[here] = np.minimum(nearest[here], distance)
        inverse_distance2 += 1.0 / nearest**2
    band = inverse_distance2 > 0
    initial = np.full(shape, np.inf)
    initial[band] = 1.0 / (np.sqrt(inverse_distance2[band]) * speed[band])
    initial[phi == 0] = 0.0
    times = initial.ravel().tolist()
    accepted = np.isfinite(initial).ravel().tolist()
    signs = phi.ravel().tolist()
    velocities = speed.ravel().tolist()
    axes = [(n, prod(shape[k + 1:]), 1.0 / float(h)**2)
            for k, (n, h) in enumerate(zip(shape, spacing))]
    queue = []

    def quadratic(terms, rhs, drop_dimensions):
        terms = sorted(terms)
        base = terms[0][0]
        weight_sum = offset_sum = square_sum = 0.0
        root = base
        for a, weight in terms:
            if drop_dimensions and weight_sum and root < a:
                break
            offset = a - base
            weight_sum += weight
            offset_sum += weight * offset
            square_sum += weight * offset * offset
            discriminant = offset_sum**2 - weight_sum * (square_sum - rhs)
            # At order 1 intermediate subsets are irrelevant: solve ALL axes.
            if drop_dimensions:
                if discriminant < 0:
                    return None
                root = base + (offset_sum + sqrt(discriminant)) / weight_sum
        if not drop_dimensions:
            if discriminant < 0:
                raise ArithmeticError("negative order-1 discriminant")
            root = base + (offset_sum + sqrt(discriminant)) / weight_sum
        return root

    def update(i):
        first_order, second_order = [], []
        for n, stride, weight in axes:
            position = i // stride % n
            # Negative direction wins a tie, as specified.
            first, direction = inf, 0
            if position > 0 and accepted[i - stride]:
                first, direction = times[i - stride], -1
            if position + 1 < n and accepted[i + stride] and times[i + stride] < first:
                first, direction = times[i + stride], 1
            if not direction:
                continue
            first_order.append((first, weight))
            a, w = first, weight
            if order == 2 and 0 <= position + 2 * direction < n:
                second = i + 2 * direction * stride
                if accepted[second] and times[second] <= first and signs[i] * signs[second] >= 0:
                    a, w = (4.0 * first - times[second]) / 3.0, 2.25 * weight
            second_order.append((a, w))
        if not first_order:
            return
        rhs = 1.0 / velocities[i]**2
        value = quadratic(second_order if order == 2 else first_order, rhs, order == 2)
        if value is None:
            value = quadratic(first_order, rhs, True)
        if value is None:
            raise ArithmeticError("negative fallback discriminant")
        if value < times[i]:
            times[i] = value
            heappush(queue, (value, i))

    def update_neighbors(i):
        for n, stride, _ in axes:
            position = i // stride % n
            if position > 0 and not accepted[i - stride]:
                update(i - stride)
            if position + 1 < n and not accepted[i + stride]:
                update(i + stride)

    for i, frozen in enumerate(accepted):
        if frozen:
            update_neighbors(i)
    while queue:
        value, i = heappop(queue)
        if accepted[i] or value != times[i]:
            continue
        accepted[i] = True
        update_neighbors(i)
    assert all(accepted)
    return np.asarray(times, dtype=np.float64).reshape(shape)
