"""Float64 second-order fast marching reference; see tests/api.py for rules."""
from heapq import heappop, heappush

import numpy as np


def travel_time(phi, speed, spacing):
    shape = phi.shape
    size = phi.size
    p = np.asarray(phi, dtype=np.float64).ravel()
    v = np.asarray(speed, dtype=np.float64).ravel()
    h = np.asarray(spacing, dtype=np.float64)
    strides = [int(np.prod(shape[k + 1:])) for k in range(len(shape))]
    coords = [np.arange(size) // strides[k] % shape[k] for k in range(len(shape))]
    t = np.full(size, np.inf)
    frozen = np.zeros(size, dtype=bool)
    inverse_distance2 = np.zeros(size)
    for k, (step, stride) in enumerate(zip(h, strides)):
        best = np.full(size, np.inf)
        for direction in (-1, 1):
            valid = (coords[k] + direction >= 0) & (coords[k] + direction < shape[k])
            indices = np.flatnonzero(valid)
            other = indices + direction * stride
            crossing = (p[indices] != 0) & ((p[indices] * p[other] < 0) | (p[other] == 0))
            indices, other = indices[crossing], other[crossing]
            distance = step * np.abs(p[indices] / (p[indices] - p[other]))
            best[indices] = np.minimum(best[indices], distance)
        inverse_distance2 += 1.0 / best**2
    initial = (inverse_distance2 > 0) & (p != 0)
    t[initial] = 1.0 / np.sqrt(inverse_distance2[initial]) / v[initial]
    t[p == 0] = 0.0
    frozen[initial | (p == 0)] = True
    heap = []

    def update(i):
        terms = []
        for k, stride in enumerate(strides):
            choices = []
            for direction in (-1, 1):
                c = coords[k][i] + direction
                if 0 <= c < shape[k]:
                    j = i + direction * stride
                    if frozen[j]:
                        choices.append((t[j], direction, j))
            if not choices:
                continue
            first, direction, j = min(choices)
            a, weight = first, 1.0 / h[k]**2
            c2 = coords[k][i] + 2 * direction
            if 0 <= c2 < shape[k]:
                j2 = j + direction * stride
                if frozen[j2] and t[j2] <= first and p[i] * p[j2] >= 0:
                    a = (4 * first - t[j2]) / 3
                    weight *= 2.25
            terms.append((a, weight, first, 1.0 / h[k]**2))
        if not terms:
            return

        def root(items):
            items = sorted(items)
            base = items[0][0]
            total_w = total_d = total_d2 = 0.0
            result = base
            for a, weight in items:
                if result < a and total_w:
                    break
                d = a - base
                total_w += weight
                total_d += weight * d
                total_d2 += weight * d * d
                disc = total_d**2 - total_w * (total_d2 - 1.0 / v[i]**2)
                if disc < 0:
                    return None
                result = base + (total_d + np.sqrt(disc)) / total_w
            return result

        value = root([(a, w) for a, w, _, _ in terms])
        if value is None:
            value = root([(first, w) for _, _, first, w in terms])
        if value < t[i]:
            t[i] = value
            heappush(heap, (value, i))

    for i in np.flatnonzero(frozen):
        for k, stride in enumerate(strides):
            for direction in (-1, 1):
                c = coords[k][i] + direction
                j = i + direction * stride
                if 0 <= c < shape[k] and not frozen[j]:
                    update(j)
    while heap:
        value, i = heappop(heap)
        if frozen[i] or value != t[i]:
            continue
        frozen[i] = True
        for k, stride in enumerate(strides):
            for direction in (-1, 1):
                c = coords[k][i] + direction
                j = i + direction * stride
                if 0 <= c < shape[k] and not frozen[j]:
                    update(j)
    assert frozen.all()
    return t.reshape(shape)
