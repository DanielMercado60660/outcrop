"""Build independent float64 gravity references for project 05.

Run with the repository's pinned Python environment. Only deterministic cases
are used. General polygons are integrated over their area, not by Talwani's
edge formula; rectangular prisms are additionally checked with Harmonica.
"""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import harmonica as hm
import numpy as np
from numpy.polynomial.legendre import leggauss

G = 6.67430e-11
OUT = Path(__file__).resolve().parent / "golden"


def integrated(vertices, observations, density, order=96):
    """Integrate 2 G rho z/r^2 over a convex polygon via triangle quadrature."""
    nodes, weights = leggauss(order)
    u = (nodes + 1) / 2
    w = weights / 2
    s, t = np.meshgrid(u, u, indexing="ij")
    wt = np.outer(w, w) * (1 - s)
    center = np.mean(vertices, axis=0)
    result = np.zeros(len(observations), dtype=np.float64)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        va, vb = a - center, b - center
        jac = abs(np.linalg.det(np.stack((va, vb))))
        xy = center + s[..., None] * va + ((1 - s) * t)[..., None] * vb
        for i, obs in enumerate(observations):
            delta = xy - obs
            result[i] += np.sum(wt * jac * delta[..., 1] / np.sum(delta**2, axis=-1))
    return result * (2 * G * density * 1e5)


def save(name, vertices, observations, density, expected):
    np.savez_compressed(OUT / (name + ".npz"), vertices=np.asarray(vertices, dtype=np.float32),
                        observations=np.asarray(observations, dtype=np.float32),
                        density=np.float64(density), gz=np.asarray(expected, dtype=np.float64))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {}
    obs = np.array([[-2100, 0], [-700, 0], [0, 0], [550, 0], [1700, 200]], dtype=np.float32)
    rect = np.array([[-800, 500], [900, 500], [900, 1600], [-800, 1600]], dtype=np.float32)
    rho = 600.0
    quad = integrated(rect, obs, rho)
    # Harmonica coordinates are easting, northing and upward; strike is north.
    prism = [-800, 900, -1e7, 1e7, -1600, -500]
    h = hm.prism_gravity((obs[:, 0].astype(float), np.zeros(len(obs)), -obs[:, 1].astype(float)),
                         prism, rho, field="g_z", parallel=False, dtype="float64")
    assert np.max(np.abs(h - quad)) / np.max(abs(quad)) < 1e-7
    save("rectangle", rect, obs, rho, h)
    cases["rectangle"] = {"source": "Harmonica finite prism, strike +/- 10,000 km; independently checked by area quadrature", "density_kg_m3": rho, "prism_w_e_s_n_bottom_top_m": prism}

    poly = np.array([[-1200, 700], [-450, 350], [650, 500], [1300, 1300], [150, 2200], [-1000, 1600]], dtype=np.float32)
    obs2 = np.array([[-2400, 0], [-900, 0], [0, -250], [500, 100], [1800, 0], [2400, 500]], dtype=np.float32)
    ref = integrated(poly, obs2, -425)
    assert np.max(abs(ref - integrated(poly, obs2, -425, order=64))) / np.max(abs(ref)) < 1e-11
    save("polygon", poly, obs2, -425, ref)
    cases["polygon"] = {"source": "96-point Gauss-Legendre area integration per triangle, checked at order 64", "density_kg_m3": -425}

    save("polygon_reversed", poly[::-1], obs2, -425, ref)
    cases["polygon_reversed"] = {"source": "same area integral as polygon, reversed winding", "density_kg_m3": -425}
    shift = np.array([4000, 3000], dtype=np.float32)
    save("polygon_translated", poly + shift, obs2 + shift, -425, ref)
    cases["polygon_translated"] = {"source": "same area integral as polygon, translated with observations", "density_kg_m3": -425, "translation_m": shift.tolist()}
    save("zero_density", poly, obs2, 0, np.zeros(len(obs2)))
    cases["zero_density"] = {"source": "exact zero density response", "density_kg_m3": 0}

    # A concave L is independently decomposed into two non-overlapping rectangles.
    ell = np.array([[-800, 400], [800, 400], [800, 900], [0, 900], [0, 1800], [-800, 1800]], dtype=np.float32)
    top = np.array([[-800, 400], [800, 400], [800, 900], [-800, 900]], dtype=np.float32)
    leg = np.array([[-800, 900], [0, 900], [0, 1800], [-800, 1800]], dtype=np.float32)
    eo = np.array([[-1600, 0], [0, 0], [1100, 0], [0, 2600]], dtype=np.float32)
    er = integrated(top, eo, 700) + integrated(leg, eo, 700)
    save("concave", ell, eo, 700, er)
    cases["concave"] = {"source": "sum of independent area integrations over two rectangles", "density_kg_m3": 700}

    # The inscribed 1024-gon differs from the exact cylinder by < 1e-5.
    angle = np.arange(1024) * (2 * np.pi / 1024)
    circle = np.column_stack((400 * np.cos(angle), 1200 + 400 * np.sin(angle))).astype(np.float32)
    co = np.array([[-1500, 0], [-600, 0], [0, 0], [600, 0], [1500, 0]], dtype=np.float32)
    cylinder = 2 * np.pi * G * 800 * 400**2 * (1200 - co[:, 1]) / (co[:, 0]**2 + (1200 - co[:, 1])**2) * 1e5
    cq = integrated(circle, co, 800, order=8)
    assert np.max(abs(cq - cylinder)) / np.max(abs(cylinder)) < 1e-5
    save("cylinder", circle, co, 800, cylinder)
    cases["cylinder"] = {"source": "analytic infinite circular cylinder; 1024-sided inscribed polygon approximation independently checked by area quadrature", "density_kg_m3": 800, "radius_m": 400, "center_m": [0, 1200]}

    # Use the exact finite rectangle response for the tested polygon. Its
    # aspect ratio still illustrates a slab, without losing the end-edge
    # radius differences in float32 arithmetic.
    width = 100000.0
    slab = np.array([[-width, 1000], [width, 1000], [width, 1300], [-width, 1300]], dtype=np.float32)
    so = np.array([[0, 0]], dtype=np.float32)
    ideal = np.array([2 * np.pi * G * 500 * 300 * 1e5])
    def finite_slab(half_width):
        def primitive(z):
            return 2 * (z * np.arctan(half_width / z) + half_width / 2 * np.log1p((z / half_width)**2))
        return 2 * G * 500 * (primitive(1300) - primitive(1000)) * 1e5

    finite = finite_slab(width)
    # The much wider rectangle verifies the independent infinite-slab limit;
    # it is not an implementation input.
    limit_width = 10000000.0
    limit_error = abs(finite_slab(limit_width) / ideal[0] - 1)
    assert limit_error < 1e-4
    save("slab", slab, so, 500, [finite])
    cases["slab"] = {"source": "exact analytic finite-width rectangle; separate infinite-slab limit assertion", "density_kg_m3": 500, "half_width_m": width, "thickness_m": 300, "infinite_slab_mgal": float(ideal[0]), "tested_relative_difference_from_infinite": float(abs(finite / ideal[0] - 1)), "limit_half_width_m": limit_width, "limit_relative_error": float(limit_error)}

    meta = {"units": {"positions": "m, x horizontal and z positive down", "density": "kg/m^3", "gz": "mGal, positive down"}, "gravitational_constant_SI": G, "seed": "none; all inputs deterministic", "versions": {name: version(name) for name in ("numpy", "harmonica", "scipy", "numba")}, "cases": cases, "tolerances": {"relative_error": {"value": 1e-4, "reason": "project-wide float32 limit, including geometric truncation"}, "cross_path_relative_error": {"value": 2e-4, "reason": "sum of two independent 1e-4 implementation error budgets"}, "zero_absolute_error_mgal": {"value": 0.0, "reason": "zero density has exactly zero response"}}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name in cases:
        with np.load(OUT / (name + ".npz")) as data:
            print(f"{name}: gz range {data['gz'].min():.9g} to {data['gz'].max():.9g} mGal")


if __name__ == "__main__":
    main()
