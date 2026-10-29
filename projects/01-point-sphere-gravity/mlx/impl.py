"""Newtonian gravity of point masses and homogeneous solid spheres."""

import mlx.core as mx

_OBSERVATION_CHUNK = 4096
_SOURCE_CHUNK = 32
_G = 6.67430e-11


def _field(coordinates, centers, masses, radii):
    delta = coordinates[:, None, :] - centers[None, :, :]

    # Scale before squaring, so a large but representable distance does not
    # overflow merely because its square is outside the float32 range.
    scale = mx.max(mx.abs(delta), axis=-1)
    safe_scale = mx.where(scale > 0, scale, mx.ones_like(scale))
    scaled_delta = delta / safe_scale[..., None]
    norm = mx.sqrt(mx.sum(scaled_delta * scaled_delta, axis=-1))
    distance = scale * norm
    inside = (radii > 0) & (distance < radii)

    # Mask both branches' denominators and arguments. In particular, neither
    # the sphere center nor an unused exterior branch should divide by zero.
    exterior_distance = mx.where(inside, mx.ones_like(distance), distance)
    exterior_norm = mx.where(inside, mx.ones_like(norm), norm)
    direction = mx.where(inside[..., None], mx.zeros_like(scaled_delta), scaled_delta)
    direction = direction / exterior_norm[..., None]
    safe_radius = mx.where(radii > 0, radii, mx.ones_like(radii))
    interior_delta = mx.where(inside[..., None], delta, mx.zeros_like(delta))
    interior_ratio = interior_delta / safe_radius[None, :, None]
    ratio_squared = mx.sum(interior_ratio * interior_ratio, axis=-1)

    gm = masses * _G
    exterior_potential = -gm / exterior_distance
    interior_potential = -(gm / safe_radius) * (1.5 - 0.5 * ratio_squared)
    # Divide by distance twice, then multiply by a unit vector. Forming
    # GM / r**3 first can underflow even when the final acceleration does not.
    exterior_acceleration = (exterior_potential / exterior_distance)[..., None] * direction
    interior_acceleration = (-(gm / safe_radius) / safe_radius)[None, :, None] * interior_ratio

    potential = mx.sum(mx.where(inside, interior_potential, exterior_potential), axis=1)
    acceleration = mx.sum(
        mx.where(inside[..., None], interior_acceleration, exterior_acceleration), axis=1
    )
    return potential, acceleration


def gravity(coordinates, centers, masses, radii):
    """Return potential and acceleration summed over all sources in SI units."""
    coordinates = mx.asarray(coordinates, dtype=mx.float32)
    centers = mx.asarray(centers, dtype=mx.float32)
    masses = mx.asarray(masses, dtype=mx.float32)
    radii = mx.asarray(radii, dtype=mx.float32)
    grid_shape = coordinates.shape[:-1]
    coordinates = mx.reshape(coordinates, (-1, 3))

    potentials = []
    accelerations = []
    for start in range(0, coordinates.shape[0], _OBSERVATION_CHUNK):
        points = coordinates[start : start + _OBSERVATION_CHUNK]
        potential = mx.zeros((points.shape[0],), dtype=mx.float32)
        acceleration = mx.zeros((points.shape[0], 3), dtype=mx.float32)
        for source in range(0, centers.shape[0], _SOURCE_CHUNK):
            end = source + _SOURCE_CHUNK
            p, a = _field(points, centers[source:end], masses[source:end], radii[source:end])
            potential = potential + p
            acceleration = acceleration + a
        potentials.append(potential)
        accelerations.append(acceleration)

    if not potentials:
        return {
            "potential": mx.zeros(grid_shape, dtype=mx.float32),
            "acceleration": mx.zeros((*grid_shape, 3), dtype=mx.float32),
        }
    return {
        "potential": mx.reshape(mx.concatenate(potentials, axis=0), grid_shape),
        "acceleration": mx.reshape(mx.concatenate(accelerations, axis=0), (*grid_shape, 3)),
    }
