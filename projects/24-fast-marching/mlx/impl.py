"""Float32 fast marching with a host heap and differentiable MLX stencils.

The heap selects the discrete stencil graph. MLX replays that graph in
dependency order, so mx.grad can differentiate the returned travel times
with respect to speed and level-set values on each fixed marching branch.
"""

import heapq
import math

import mlx.core as mx
import numpy as np


def _interface_times(phi, speed, spacing, order):
    level = mx.array(phi, dtype=mx.float32)
    velocity = mx.array(speed, dtype=mx.float32)
    grid_spacing = mx.array(spacing, dtype=mx.float32)
    distances = []
    ndim = phi.ndim
    for axis in range(ndim):
        h = grid_spacing[axis]
        left = [slice(None)] * ndim
        right = [slice(None)] * ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        p, q = level[tuple(left)], level[tuple(right)]
        strict = ((p < 0) & (q > 0)) | ((p > 0) & (q < 0))
        left_cross = strict
        right_cross = strict
        if order == 2:
            left_cross = strict | ((p != 0) & (q == 0))
            right_cross = strict | ((q != 0) & (p == 0))
        # Scale before subtracting: opposite-sign level values need not have
        # length units and may be much larger than the float32 range of p-q.
        scale = mx.maximum(mx.abs(p), mx.abs(q))
        scale = mx.where(scale == 0, mx.ones_like(scale), scale)
        scaled_p, scaled_q = p / scale, q / scale
        denominator = mx.where(left_cross | right_cross, scaled_p - scaled_q, 1)
        left_distance = mx.where(left_cross, h * mx.abs(scaled_p / denominator),
                                 float("inf"))
        right_distance = mx.where(right_cross, h * mx.abs(scaled_q / denominator),
                                  float("inf"))
        edge_shape = list(phi.shape)
        edge_shape[axis] = 1
        edge = mx.full(edge_shape, float("inf"), dtype=mx.float32)
        plus = mx.concatenate([left_distance, edge], axis=axis)
        minus = mx.concatenate([edge, right_distance], axis=axis)
        distances.append(mx.minimum(plus, minus))
    nearest = distances[0]
    for distance in distances[1:]:
        nearest = mx.minimum(nearest, distance)
    band = mx.isfinite(nearest)
    shortest = mx.where(band, nearest, 1)
    inverse_sq = mx.zeros_like(level)
    for distance in distances:
        present = mx.isfinite(distance)
        safe_distance = mx.where(present, distance, 1)
        ratio = mx.where(present, shortest / safe_distance, 0)
        inverse_sq = inverse_sq + ratio * ratio
    # shortest / sqrt(sum((shortest/d)**2)) equals the required inverse-square
    # combination without squaring 1/d at nodes almost on the interface.
    norm = mx.sqrt(mx.where(band, inverse_sq, 1))
    times = mx.where(level == 0, 0,
                    mx.where(band, (shortest / norm) / velocity, float("inf")))
    frozen = (level == 0) | band
    # Prepare the initial trial band on the GPU. Each node is updated once;
    # the frozen band is unchanged until all these updates have been queued.
    adjacent = mx.zeros(phi.shape, dtype=mx.bool_)
    for axis in range(ndim):
        left = [slice(None)] * ndim
        right = [slice(None)] * ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        edge_shape = list(phi.shape)
        edge_shape[axis] = 1
        edge = mx.zeros(edge_shape, dtype=mx.bool_)
        adjacent = adjacent | mx.concatenate([frozen[tuple(right)], edge], axis=axis)
        adjacent = adjacent | mx.concatenate([edge, frozen[tuple(left)]], axis=axis)
    trials = adjacent & ~frozen
    weights = 1 / (grid_spacing * grid_spacing)
    slowness_sq = 1 / (velocity * velocity)
    return times, frozen, trials, weights, slowness_sq


def _root(terms, slowness_sq, dropping):
    """Larger root, shifted to avoid subtracting squared absolute times.

    Every scalar and operation here stays float32, including sqrt. Keeping
    this shift is essential on fine grids, where the absolute time is much
    larger than its increment. No float64 accuracy exception is needed.
    """
    terms = sorted(terms, key=lambda term: term[:2])
    center = terms[0][0]
    total_w = total_b = np.float32(0)
    total_c = -slowness_sq
    result = center
    used = []
    for term in terms:
        a, w = term[:2]
        if dropping and total_w and result < a:
            break
        delta = a - center
        total_w += w
        total_b += w * delta
        total_c += w * delta * delta
        discriminant = total_b * total_b - total_w * total_c
        if discriminant < 0:
            if dropping:
                return None, ()
            # The order-1 contract retains every axis. Guard roundoff at a
            # repeated root without introducing dimension dropping or NaNs.
            discriminant = np.float32(0)
        result = center + (total_b + np.sqrt(discriminant)) / total_w
        used.append(term)
    return result, used


def _march_float32(initial, frozen_band, trial_band, phi, weights, slowness_sq, order):
    """Select winning stencils with the serial float32 priority queue.

    Each acceptance changes the frozen stencil for the next priority-queue
    decision. A GPU dispatch per heap operation would require host/device
    synchronization per node; this exact queue stays on the host instead.
    Its detached scalar arithmetic determines only discrete choices. The
    returned dependency layers are integer metadata for the MLX replay.
    """
    shape = phi.shape
    ndim = len(shape)
    strides = [math.prod(shape[axis + 1:]) for axis in range(ndim)]
    # Detach only the heap's decisions, not the values replayed by MLX.
    initial, frozen_band, trial_band, phi, weights, slowness_sq = (
        mx.stop_gradient(value)
        for value in (initial, frozen_band, trial_band, phi, weights, slowness_sq)
    )
    mx.eval(initial, frozen_band, trial_band, phi, weights, slowness_sq)
    # list(array), unlike array.tolist(), preserves the float32 scalar type.
    # These conversions are I/O for the serial loop, not CPU bulk arithmetic.
    times = list(np.asarray(initial, dtype=np.float32).ravel())
    frozen = np.asarray(frozen_band).ravel().tolist()
    levels = list(np.asarray(phi, dtype=np.float32).ravel())
    weights = list(np.asarray(weights, dtype=np.float32))
    rhs = list(np.asarray(slowness_sq, dtype=np.float32).ravel())
    third = np.float32(1) / np.float32(3)
    second_weight = np.float32(2.25)
    heap = []
    stencils = [None] * len(times)
    depths = [0] * len(times)
    layers = [{}]

    def update(index):
        first = []
        second = []
        for axis in range(ndim):
            stride = strides[axis]
            coordinate = (index // stride) % shape[axis]
            chosen = -1
            direction = 0
            if coordinate > 0 and frozen[index - stride]:
                chosen = index - stride
                direction = -1
            if coordinate + 1 < shape[axis] and frozen[index + stride]:
                plus = index + stride
                if chosen < 0 or times[plus] < times[chosen]:
                    chosen = plus
                    direction = 1
            if chosen < 0:
                continue
            t1 = times[chosen]
            weight = weights[axis]
            # Identical parents encode a first-order term; distinct parents
            # encode the second-order term in this same chosen direction.
            first.append((t1, weight, chosen, chosen, axis))
            if order == 2 and 0 <= coordinate + 2 * direction < shape[axis]:
                neighbor2 = index + 2 * direction * stride
                if (frozen[neighbor2] and times[neighbor2] <= t1
                        # Equivalent to phi_i*phi_second >= 0, without a
                        # product that could underflow or overflow float32.
                        and (levels[index] == 0 or levels[neighbor2] == 0
                             or ((levels[index] < 0) == (levels[neighbor2] < 0)))):
                    # (4*T1-T2)/3, rearranged to preserve a small increment.
                    a = t1 + (t1 - times[neighbor2]) * third
                    second.append((a, second_weight * weight, chosen, neighbor2, axis))
                    continue
            second.append((t1, weight, chosen, chosen, axis))
        if not first:
            return

        if order == 1:
            candidate, used = _root(first, rhs[index], False)
        else:
            candidate, used = _root(second, rhs[index], True)
            if candidate is None:
                candidate, used = _root(first, rhs[index], True)
            if candidate is None:
                return
        if candidate < times[index]:
            times[index] = candidate
            stencils[index] = [term[2:] for term in used]
            heapq.heappush(heap, (candidate, index))

    def update_neighbors(index):
        for axis in range(ndim):
            stride = strides[axis]
            coordinate = (index // stride) % shape[axis]
            if coordinate > 0 and not frozen[index - stride]:
                update(index - stride)
            if coordinate + 1 < shape[axis] and not frozen[index + stride]:
                update(index + stride)

    for index in np.flatnonzero(np.asarray(trial_band).ravel()):
        update(int(index))
    while heap:
        value, index = heapq.heappop(heap)
        if frozen[index] or value != times[index]:
            continue
        stencil = stencils[index]
        depth = 1 + max(depths[parent] for term in stencil for parent in term[:2])
        depths[index] = depth
        while len(layers) <= depth:
            layers.append({})
        row = [index] + [item for term in stencil for item in term]
        layers[depth].setdefault(len(stencil), []).append(row)
        stencils[index] = None
        frozen[index] = True
        update_neighbors(index)
    return layers[1:]


def _replay(initial, weights, slowness_sq, layers):
    """Recompute every accepted value in MLX; gradients use ordinary mx.grad.

    Parents in a winning update were already frozen when it was proposed.
    Thus dependency layers are acyclic even for a noncausal order-1 root.
    Within a layer, equal stencil sizes can be evaluated together. No heap
    time is substituted into the result or into the differentiation graph.
    """
    time = initial.reshape(-1)
    rhs = slowness_sq.reshape(-1)
    third = mx.array(1, dtype=mx.float32) / 3
    for layer in layers:
        for count, rows in layer.items():
            indices = mx.array(np.asarray(rows, dtype=np.int32))
            nodes = indices[:, 0]
            parent1 = indices[:, 1::3]
            parent2 = indices[:, 2::3]
            axes = indices[:, 3::3]
            t1, t2 = time[parent1], time[parent2]
            a = t1 + (t1 - t2) * third
            base_weight = weights[axes]
            w = mx.where(parent1 == parent2, base_weight, 2.25 * base_weight)
            center = a[:, 0]
            total_w = mx.zeros_like(center)
            total_b = mx.zeros_like(center)
            total_c = -rhs[nodes]
            for axis in range(count):
                delta = a[:, axis] - center
                weight = w[:, axis]
                total_w = total_w + weight
                total_b = total_b + weight * delta
                total_c = total_c + weight * delta * delta
            discriminant = total_b * total_b - total_w * total_c
            value = center + (total_b + mx.sqrt(mx.maximum(discriminant, 0))) / total_w
            # MLX indexed assignment is a functional scatter in the graph.
            time[nodes] = value
        # Bound the unevaluated forward graph. Synchronous eval retains the
        # graph when called under mx.grad / mx.value_and_grad.
        mx.eval(time)
    return time.reshape(initial.shape)


def travel_time(phi, speed, spacing, order):
    """Solve the nonperiodic eikonal problem described in tests/api.py."""
    phi = mx.array(phi, dtype=mx.float32)
    speed = mx.array(speed, dtype=mx.float32)
    if phi.ndim not in (2, 3) or phi.shape != speed.shape:
        raise ValueError("phi and speed must have the same 2D or 3D shape")
    if len(spacing) != phi.ndim or order not in (1, 2):
        raise ValueError("spacing must match the grid dimensions and order must be 1 or 2")
    # Scalar input validation only; solver coefficients are formed in MLX.
    for h in spacing:
        h2 = h * h
        if not math.isfinite(h) or h <= 0 or h2 == 0 or not math.isfinite(h2):
            raise ValueError("spacing is outside the supported numerical range")
        if not math.isfinite(1.0 / h2):
            raise ValueError("spacing is outside the supported numerical range")
    initial, frozen, trials, weights, slowness_sq = _interface_times(
        phi, speed, spacing, order
    )
    layers = _march_float32(initial, frozen, trials, phi, weights, slowness_sq, order)
    time = _replay(initial, weights, slowness_sq, layers)
    return {"time": time}
