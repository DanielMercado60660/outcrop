"""First-order nodal eikonal tomography, differentiated with MLX.

Directional Gauss--Seidel sweeps are vectorized over independent diagonals
and shots. A final causal sweep refreshes the converged computational graph:
unchanged float32 values alone do not imply converged iterative derivatives.
"""

import mlx.core as mx
import numpy as np


def _root(vertical, horizontal, slowness, dz, dx):
    """Float32 Godunov root, expressed as an increment above the minimum."""
    z_first = vertical <= horizontal
    a = mx.where(z_first, vertical, horizontal)
    b = mx.where(z_first, horizontal, vertical)
    ha = mx.where(z_first, dz, dx)
    hb = mx.where(z_first, dx, dz)

    # Both branches of where participate in autodiff. Keep even inactive
    # quadratics finite, avoiding inf-inf and zero times an infinite VJP.
    reachable = mx.isfinite(a)
    a_safe = mx.where(reachable, a, 0.0)
    b_safe = mx.where(mx.isfinite(b), b, a_safe)
    one_axis = a_safe + slowness * ha
    two_axes = reachable & mx.isfinite(b) & (one_axis > b)
    delta = mx.where(two_axes, b_safe - a_safe, 0.0)
    quadratic_s = mx.where(two_axes, slowness, 1.0)

    # Normalize extreme active increments before squaring. The scale is a
    # locally fixed change of units, so stopping its derivative leaves the
    # derivative of the equivalent root unchanged. Unit scaling retains the
    # original arithmetic for ordinary tomography models. Mask slowness BEFORE
    # squaring, including at unreachable nodes and one-axis updates.
    magnitude = mx.maximum(delta, quadratic_s)
    extreme = (magnitude > 2.0**32) | (magnitude < 2.0**-32)
    scale = mx.stop_gradient(mx.where(extreme, magnitude, 1.0))
    delta = delta / scale
    quadratic_s = quadratic_s / scale

    ha_squared = ha * ha
    hb_squared = hb * hb
    aa = 1.0 / ha_squared + 1.0 / hb_squared
    bb = (-2.0 * delta) / hb_squared
    cc = (delta * delta) / hb_squared - quadratic_s * quadratic_s
    discriminant = bb * bb - (4.0 * aa) * cc
    increment = ((-bb + mx.sqrt(discriminant)) / (2.0 * aa)) * scale
    value = mx.where(two_axes, a_safe + increment, one_axis)
    return mx.where(reachable, value, float("inf"))


def _axis_min(negative, positive):
    # Explicit selection gives the lower spatial index the entire derivative
    # at a minimizing-neighbor tie, rather than splitting it between neighbors.
    return mx.where(negative <= positive, negative, positive)


class _Sweeper:
    """Fixed geometry and a deterministic four-direction sweep schedule."""

    def __init__(self, shape, spacing, sources):
        self.nz, self.nx = shape
        self.size = self.nz * self.nx
        self.ns = len(sources)
        self.dz = mx.array(spacing[0], dtype=mx.float32)
        self.dx = mx.array(spacing[1], dtype=mx.float32)

        source_indices = sources[:, 0] * self.nx + sources[:, 1]
        self.source_nodes = source_indices
        nodes = np.arange(self.size, dtype=np.int32)
        iz, ix = nodes // self.nx, nodes % self.nx
        # One immutable +infinity sentinel represents every exterior neighbor.
        neighbors = (
            np.where(iz > 0, nodes - self.nx, self.size),
            np.where(iz + 1 < self.nz, nodes + self.nx, self.size),
            np.where(ix > 0, nodes - 1, self.size),
            np.where(ix + 1 < self.nx, nodes + 1, self.size),
        )
        self.neighbor_nodes = np.stack(neighbors)
        initial = np.full((self.ns, self.size + 1), np.inf, dtype=np.float32)
        initial[np.arange(self.ns), source_indices] = 0.0
        self.initial = mx.array(initial)

        self.schedule = []
        for z_direction, x_direction in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            for diagonal in range(self.nz + self.nx - 1):
                z = np.arange(
                    max(0, diagonal - self.nx + 1),
                    min(self.nz - 1, diagonal) + 1,
                    dtype=np.int32,
                )
                x = diagonal - z
                if z_direction < 0:
                    z = self.nz - 1 - z
                if x_direction < 0:
                    x = self.nx - 1 - x
                indices = z * self.nx + x
                self.schedule.append((
                    mx.array(indices[None, :], dtype=mx.int32),
                    tuple(mx.array(n[indices], dtype=mx.int32) for n in neighbors),
                    mx.array(source_indices[:, None] == indices[None, :]),
                ))

    def cycle(self, time, slowness):
        """Four Gauss--Seidel sweeps; nodes on a diagonal are independent."""
        for indices, neighbors, source_mask in self.schedule:
            up, down, left, right = (
                mx.take(time, n, axis=1) for n in neighbors
            )
            candidate = _root(
                _axis_min(up, down),
                _axis_min(left, right),
                mx.take(slowness, indices[0]),
                self.dz,
                self.dx,
            )
            old = mx.take(time, indices[0], axis=1)
            # On equality refresh the expression, not just its rounded value.
            value = mx.where(candidate <= old, candidate, old)
            value = mx.where(source_mask, 0.0, value)
            time = mx.put_along_axis(time, indices, value, axis=1)
        return time

    def converge(self, slowness):
        """Find the unroll length and arrival ordering outside autodiff."""
        time = self.initial
        # Even a winding causal path has fewer than size distinct nodes.
        # A complete cycle propagates at least one further causal edge.
        for count in range(1, self.size + 2):
            previous = time
            time = self.cycle(time, slowness)
            unchanged = mx.array_equal(time, previous)
            mx.eval(time, unchanged)
            if unchanged.item():
                return count, time
        raise RuntimeError("Eikonal sweeps did not reach a bitwise fixed point")

    def causal_schedule(self, time):
        """Batch independent nodes without changing the serial refresh.

        Orient EVERY neighboring pair by the original per-shot argsort rank,
        including inactive neighbors and equal arrivals. A node's batch follows
        all earlier neighbors and precedes all later neighbors. Consequently
        each root sees exactly the same refreshed/unrefreshed inputs as the
        serial implementation, even if reevaluation changes a rounded value.
        NumPy only builds this integer topology; it evaluates no local roots.
        """
        order = np.array(mx.argsort(time[:, :self.size], axis=1))
        batches = []
        stride = self.size + 1
        for shot in range(self.ns):
            ranks = np.full(stride, self.size, dtype=np.int32)
            ranks[order[shot]] = np.arange(self.size, dtype=np.int32)
            levels = np.full(stride, -1, dtype=np.int32)
            for rank, node in enumerate(order[shot]):
                neighbors = self.neighbor_nodes[:, node]
                earlier = neighbors[ranks[neighbors] < rank]
                level = 1 + max((int(levels[n]) for n in earlier), default=-1)
                levels[node] = level
                if level == len(batches):
                    batches.append([])
                batches[level].append(shot * stride + int(node))

        schedule = []
        for batch in batches:
            indices = np.asarray(batch, dtype=np.int32)
            shots, nodes = indices // stride, indices % stride
            neighbors = self.neighbor_nodes[:, nodes] + shots[None, :] * stride
            schedule.append((
                mx.array(indices),
                tuple(mx.array(n, dtype=mx.int32) for n in neighbors),
                mx.array(nodes, dtype=mx.int32),
                mx.array(nodes == self.source_nodes[shots]),
            ))
        return schedule

    def causal_refresh(self, time, slowness, schedule):
        """Differentiate causal roots, with one field update per batch.

        Active parents have smaller arrivals and are already refreshed. MLX
        differentiates the complete converged graph; no adjoint or derivative
        formula is supplied. On ordinary grids, independent causal fronts
        replace the previous one-full-field-update-per-node loop.
        """
        time = time.reshape(-1)
        for indices, neighbors, model_indices, source_mask in schedule:
            up, down, left, right = (
                mx.take(time, n) for n in neighbors
            )
            candidate = _root(
                _axis_min(up, down),
                _axis_min(left, right),
                mx.take(slowness, model_indices),
                self.dz,
                self.dx,
            )
            value = mx.where(source_mask, 0.0, candidate)
            time = mx.put_along_axis(time, indices, value, axis=0)
        return time.reshape((self.ns, self.size + 1))


def misfit_and_grad(slowness, spacing, sources, receivers, observed, weights):
    """Return arrivals, node samples, weighted sum loss and dL/dslowness."""
    # NumPy is used only for I/O and integer scheduling, never for the solve
    # or its derivatives. All numerical solver arithmetic is float32 MLX.
    slowness_mx = mx.array(np.asarray(slowness, dtype=np.float32))
    sources_np = np.asarray(sources, dtype=np.int32)
    receivers_np = np.asarray(receivers, dtype=np.int32)
    observed_mx = mx.array(np.asarray(observed, dtype=np.float32))
    weights_mx = mx.array(np.asarray(weights, dtype=np.float32))
    sweeper = _Sweeper(slowness_mx.shape, spacing, sources_np)
    receiver_indices = mx.array(
        receivers_np[:, 0] * sweeper.nx + receivers_np[:, 1], dtype=mx.int32
    )
    cycles, converged = sweeper.converge(slowness_mx.reshape(-1))
    causal_schedule = sweeper.causal_schedule(converged)

    def objective(model):
        model_flat = model.reshape(-1)
        time = sweeper.initial
        # Replay the complete deterministic solve inside value_and_grad.
        # No detached warm start or custom reverse pass is used.
        for _ in range(cycles):
            time = sweeper.cycle(time, model_flat)
        time = sweeper.causal_refresh(time, model_flat, causal_schedule)
        field = time[:, :sweeper.size].reshape((sweeper.ns, *model.shape))
        predicted = mx.take(time, receiver_indices, axis=1)
        residual = predicted - observed_mx
        loss = 0.5 * mx.sum(weights_mx * (residual * residual))
        return loss, field, predicted

    (loss, field, predicted), gradient = mx.value_and_grad(objective)(slowness_mx)
    mx.eval(loss, field, predicted, gradient)
    return {
        "traveltime": np.array(field),
        "predicted": np.array(predicted),
        "objective": np.array(loss),
        "gradient": np.array(gradient),
    }
