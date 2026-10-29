"""Adaptive two-point Gauss-Legendre tesseroid gravity in MLX."""

import itertools
import math

import mlx.core as mx
import numpy as np

_DEG = math.pi / 180.0
_G = 6.6743e-11
_NODES = tuple(itertools.product((-1 / math.sqrt(3), 1 / math.sqrt(3)), repeat=3))


def _leaves(observation, source, ratio, radial_adaptive):
    """Find leaf bounds as fractions of the original source intervals."""
    west, east, south, north, bottom, top = map(float, source)
    lon, lat, radius = map(float, observation)
    widths = (east - west, north - south, top - bottom)
    stack = [(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)]
    leaves = []
    while stack:
        cell = stack.pop()
        a, b, c, d, e, f = cell
        lon_center = west + widths[0] * (a + b) * 0.5
        lat_center = south + widths[1] * (c + d) * 0.5
        r_center = bottom + widths[2] * (e + f) * 0.5
        lat_obs = lat * _DEG
        lat_c = lat_center * _DEG
        hav = (math.sin((lat - lat_center) * _DEG * 0.5) ** 2
               + math.cos(lat_obs) * math.cos(lat_c)
               * math.sin((lon - lon_center) * _DEG * 0.5) ** 2)
        distance = math.sqrt(max(0.0, (radius - r_center) ** 2
                                 + 4 * radius * r_center * hav))
        r_top = bottom + widths[2] * f
        lon_size = r_top * 2 * math.asin(max(-1.0, min(1.0,
            math.cos(lat_c) * math.sin(widths[0] * (b - a) * _DEG * 0.5))))
        lat_size = r_top * widths[1] * (d - c) * _DEG
        rad_size = widths[2] * (f - e)
        split = (distance < ratio * lon_size, distance < ratio * lat_size,
                 radial_adaptive and distance < ratio * rad_size)
        if not any(split):
            leaves.append(cell)
            continue
        intervals = []
        for axis, (low, high, divide) in enumerate(
                zip(cell[::2], cell[1::2], split)):
            mid = (low + high) * 0.5
            if divide:
                origin = (west, south, bottom)[axis]
                lower = origin + widths[axis] * low
                middle = origin + widths[axis] * mid
                upper = origin + widths[axis] * high
                # Boundary observations can qualify forever. Never enqueue a
                # child identical to its parent, even in physical coordinates.
                if not (low < mid < high and lower < middle < upper):
                    raise ValueError(
                        "adaptive subdivision cannot make numerical progress; "
                        "observation may lie on a source boundary"
                    )
            intervals.append(((low, mid), (mid, high)) if divide else ((low, high),))
        for parts in itertools.product(*intervals):
            stack.append(tuple(value for pair in parts for value in pair))
    return leaves


def _quadrature(observation, source, density, leaves, field):
    # Node fractions and cell volume fractions are fixed by the subdivision.
    fractions = []
    volumes = []
    for cell in leaves:
        bounds = tuple(zip(cell[::2], cell[1::2]))
        volume = math.prod(high - low for low, high in bounds)
        for node in _NODES:
            fractions.append(tuple((low + high + x * (high - low)) * 0.5
                                   for (low, high), x in zip(bounds, node)))
            volumes.append(volume)
    frac = mx.array(np.asarray(fractions, dtype=np.float32))
    volume = mx.array(np.asarray(volumes, dtype=np.float32))
    lon_width = source[1] - source[0]
    lat_width = source[3] - source[2]
    rad_width = source[5] - source[4]
    source_lat = (source[2] + lat_width * frac[:, 1]) * _DEG
    obs_lat = observation[1] * _DEG
    # Subtract the source origin before adding the small local offset.
    delta_lon = ((observation[0] - source[0]) - lon_width * frac[:, 0]) * _DEG
    delta_lat = ((observation[1] - source[2]) - lat_width * frac[:, 1]) * _DEG
    rad_offset = rad_width * frac[:, 2]
    source_radius = source[4] + rad_offset
    radial_difference = (observation[2] - source[4]) - rad_offset
    hav = (mx.sin(delta_lat * 0.5) ** 2
           + mx.cos(obs_lat) * mx.cos(source_lat) * mx.sin(delta_lon * 0.5) ** 2)
    one_minus_cos = 2 * hav
    # Normalize before squaring so distant float32 observations do not
    # overflow. The local radial difference still preserves near-field detail.
    distance_scale = mx.maximum(observation[2], source_radius)
    distance = distance_scale * mx.sqrt(
        (radial_difference / distance_scale) ** 2
        + 2 * (observation[2] / distance_scale)
        * (source_radius / distance_scale) * one_minus_cos
    )
    mass = (density * (lon_width * _DEG) * (lat_width * _DEG) * rad_width
            * (volume / 8) * source_radius ** 2 * mx.cos(source_lat))
    if field == "potential":
        return _G * mx.sum(mass / distance)
    radial_projection = radial_difference + source_radius * one_minus_cos
    # Divide before multiplying: neither L**3 nor mass * radial_projection
    # need be representable for the resulting acceleration to be finite.
    return (_G * 1e5) * mx.sum(
        (mass / distance) * (radial_projection / distance) / distance
    )


def tesseroid_gravity(coordinates, tesseroids, density, field, radial_adaptive):
    """Compute potential or downward radial acceleration for each observation.

    Raise ValueError if a required adaptive split cannot make numerical
    progress. In particular, source-boundary observations can require
    indefinite refinement and are rejected when that resolution is reached.
    """
    if field not in ("potential", "g_z"):
        raise ValueError("field must be 'potential' or 'g_z'")
    # Subdivision is a discrete choice. Keep its host-side snapshot separate
    # from the MLX values used in quadrature, so the latter retain gradients.
    coords = mx.array(coordinates, dtype=mx.float32)
    sources = mx.array(tesseroids, dtype=mx.float32)
    densities = mx.array(density, dtype=mx.float32)
    coordinates_host = np.asarray(mx.stop_gradient(coords), dtype=np.float32)
    sources_host = np.asarray(mx.stop_gradient(sources), dtype=np.float32)
    densities_host = np.asarray(mx.stop_gradient(densities), dtype=np.float32)
    ratio = 1.0 if field == "potential" else 2.5
    results = []
    for i, observation in enumerate(coordinates_host):
        value = mx.array(0.0, dtype=mx.float32)
        for j, source in enumerate(sources_host):
            if densities_host[j] == 0:
                continue
            leaves = _leaves(observation, source, ratio, radial_adaptive)
            value = value + _quadrature(coords[i], sources[j], densities[j], leaves, field)
        results.append(value)
    return {"field": mx.stack(results) if results else mx.zeros((0,), dtype=mx.float32)}
