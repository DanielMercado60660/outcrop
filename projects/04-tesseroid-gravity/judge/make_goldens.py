"""Generate deterministic Harmonica tesseroid reference data."""

import importlib.metadata
import json
from pathlib import Path

import harmonica as hm
import numpy as np

ROOT = Path(__file__).resolve().parent / "golden"


def subdivide(tesseroids, density, count):
    """Pre-split each source into count cells per axis, preserving density."""
    blocks, values = [], []
    for source, rho in zip(tesseroids, density, strict=True):
        axes = [np.linspace(source[i], source[i + 1], count + 1) for i in (0, 2, 4)]
        for i in range(count):
            for j in range(count):
                for k in range(count):
                    blocks.append([axes[0][i], axes[0][i + 1], axes[1][j],
                                   axes[1][j + 1], axes[2][k], axes[2][k + 1]])
                    values.append(rho)
    return np.asarray(blocks), np.asarray(values)


def reference(coords, sources, density, field, radial):
    # Harmonica specializes its kernels on input dtypes, not just output dtype.
    coords, sources, density = (np.asarray(x, dtype=np.float64)
                                for x in (coords, sources, density))
    return hm.tesseroid_gravity(tuple(coords.T), sources, density, field,
                                radial_adaptive_discretization=radial,
                                parallel=False, dtype=np.float64)


def split_margin(coords, sources, density, ratio, radial):
    """Smallest relative distance from any active adaptive split threshold."""
    smallest = float("inf")
    for point in coords.astype(np.float64):
        for source, rho in zip(sources.astype(np.float64), density, strict=True):
            if rho == 0:
                continue
            stack = [source]
            while stack:
                w, e, s, n, bottom, top = stack.pop()
                lon, lat, radius = point
                lon_c, lat_c, rad_c = (w + e) / 2, (s + n) / 2, (bottom + top) / 2
                phi, phi_c = np.radians([lat, lat_c])
                hav = (np.sin((phi - phi_c) / 2) ** 2
                       + np.cos(phi) * np.cos(phi_c)
                       * np.sin(np.radians(lon - lon_c) / 2) ** 2)
                distance = np.sqrt((radius - rad_c) ** 2 + 4 * radius * rad_c * hav)
                sizes = [2 * top * np.arcsin(np.cos(phi_c) * np.sin(np.radians(e - w) / 2)),
                         top * np.radians(n - s), top - bottom]
                split = [distance / size < ratio for size in sizes]
                active = 3 if radial else 2
                smallest = min(smallest, *(abs(distance / sizes[i] / ratio - 1)
                                           for i in range(active)))
                if not radial:
                    split[2] = False
                if any(split):
                    axes = [np.linspace(a, b, 3 if flag else 2)
                            for a, b, flag in zip((w, s, bottom), (e, n, top), split,
                                                  strict=True)]
                    for i in range(len(axes[0]) - 1):
                        for j in range(len(axes[1]) - 1):
                            for k in range(len(axes[2]) - 1):
                                stack.append(np.array([axes[0][i], axes[0][i + 1],
                                                       axes[1][j], axes[1][j + 1],
                                                       axes[2][k], axes[2][k + 1]]))
    return smallest


def main():
    ROOT.mkdir(exist_ok=True)
    # Inputs are quantized first: references describe exactly the float32 API inputs.
    cases = {
        "near": (
            [[0, 0, 6375000], [0.75, 0.25, 6381000], [-1.5, 0.5, 6391000]],
            [[-1, 1, -1, 1, 6361000, 6371000]], [2670]),
        "mixed": (
            [[20.13, 34, 6381000], [22, 36, 6401000], [27, 32, 6471000]],
            [[19, 21, 33, 35, 6351000, 6371000],
             [22, 24, 34, 36, 6341000, 6361000],
             [18, 19, 31, 32, 6351000, 6371000]], [2500, -800, 0]),
        "distance": (
            [[0, 0, 6381000], [0, 0, 6471000], [0, 0, 7371000],
             [0, 0, 16371000], [45, 30, 7371000]],
            [[-1, 1, -1, 1, 6361000, 6371000]], [2670]),
    }
    metadata = {"versions": {n: importlib.metadata.version(n) for n in
                             ("numpy", "harmonica", "numba")},
                "seed": "none; explicit deterministic cases",
                "settings": {"glq_degrees": [2, 2, 2],
                             "distance_size_ratios": {"potential": 1, "g_z": 2.5},
                             "input_precision": "float32-quantized inputs promoted to float64 before every Harmonica call",
                             "minimum_split_margin": "minimum abs(distance / size / ratio - 1) across active decisions for nonzero sources, computed with stable spherical formulas; must exceed 1e-4",
                             "fine": "uniform 8 subdivisions on each source axis, then Harmonica adaptive radial=True",
                             "finer": "uniform 16 subdivisions on each source axis, then Harmonica adaptive radial=True"},
                "tolerances": {"matched_relative": [1e-4, "project-wide float32 limit"],
                               "cross_relative": [1e-4, "float32 implementation parity"],
                               "fine_relative": [3e-3, "adaptive quadrature approximation; proposed bound"]},
                "cases": {}}
    for name, (coords, sources, rho) in cases.items():
        coords, sources, rho = (np.asarray(x, np.float32) for x in (coords, sources, rho))
        fine_sources, fine_rho = subdivide(sources.astype(float), rho.astype(float), 8)
        finer_sources, finer_rho = subdivide(sources.astype(float), rho.astype(float), 16)
        arrays = {"coordinates": coords, "tesseroids": sources, "density": rho}
        case_meta = {"coordinates": coords.tolist(), "tesseroids": sources.tolist(),
                     "density": rho.tolist(), "results": {}}
        for field in ("potential", "g_z"):
            fine = reference(coords, fine_sources, fine_rho, field, True)
            finer = reference(coords, finer_sources, finer_rho, field, True)
            arrays[f"{field}_fine"] = finer
            for radial in (False, True):
                margin = split_margin(coords, sources, rho,
                                      1 if field == "potential" else 2.5, radial)
                case_meta["results"][f"{field}_{int(radial)}_minimum_split_margin"] = margin
                assert margin > 1e-4, (name, field, radial, margin)
                matched = reference(coords, sources, rho, field, radial)
                arrays[f"{field}_{int(radial)}"] = matched
                case_meta["results"][f"{field}_{int(radial)}"] = matched.tolist()
                case_meta["results"][f"{field}_{int(radial)}_fine_max_relative"] = float(
                    np.max(np.abs(matched - finer) / np.maximum(np.abs(finer), 1e-30)))
            case_meta["results"][f"{field}_fine"] = finer.tolist()
            case_meta["results"][f"{field}_fine_finer_max_relative"] = float(
                np.max(np.abs(fine - finer) / np.maximum(np.abs(finer), 1e-30)))
            assert np.all(np.isfinite(finer))
        np.savez_compressed(ROOT / f"{name}.npz", **arrays)
        metadata["cases"][name] = case_meta
    # At 10,000 km above the source the finite source is almost a point mass.
    # Exact spherical volume, with point placed at the central radius.
    r0, r1 = 6361000.0, 6371000.0
    volume = (np.radians(2) * (np.sin(np.radians(1)) - np.sin(np.radians(-1)))
              * (r1**3 - r0**3) / 3)
    distance = 16371000.0 - (r0 + r1) / 2
    gm = 6.6743e-11 * 2670 * volume
    point_potential = gm / distance
    point_gz = gm / distance**2 * 1e5
    far = metadata["cases"]["distance"]["results"]
    assert abs(far["potential_fine"][3] / point_potential - 1) < 0.01
    assert abs(far["g_z_fine"][3] / point_gz - 1) < 0.01
    assert all(x > 0 for x in far["potential_fine"] + far["g_z_fine"])
    assert far["g_z_fine"][0] > far["g_z_fine"][1] > far["g_z_fine"][2] > far["g_z_fine"][3]
    metadata["sanity"] = {"far_point_mass_potential": point_potential,
                          "far_point_mass_g_z": point_gz,
                          "far_comparison": "finite source versus central point mass; within 1%"}
    (ROOT / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["cases"], indent=2))


if __name__ == "__main__":
    main()
