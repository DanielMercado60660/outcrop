"""Write float64 analytic Newtonian gravity references for project 01."""

import json
import platform
import runpy
from pathlib import Path

import numpy as np

G = 6.67430e-11
OUT = Path(__file__).resolve().parent / "golden"


def field(coordinates, centers, masses, radii):
    """Analytic homogeneous solid sphere (shell theorem) and point fields."""
    delta = coordinates[..., None, :] - centers
    distance = np.linalg.norm(delta, axis=-1)
    inside = (radii > 0) & (distance < radii)
    safe_radius = np.where(radii > 0, radii, 1.0)
    safe_distance = np.where(inside, 1.0, distance)
    potential = np.where(
        inside,
        -G * masses * (3 * safe_radius**2 - distance**2) / (2 * safe_radius**3),
        -G * masses / safe_distance,
    )
    factor = np.where(inside, -G * masses / safe_radius**3, -G * masses / safe_distance**3)
    return potential.sum(axis=-1), (factor[..., None] * delta).sum(axis=-2)


def save(name, coordinates, centers, masses, radii):
    # Round inputs first: the float64 oracle describes the exact float32 inputs.
    inputs = {k: np.asarray(v, dtype=np.float32) for k, v in locals().copy().items() if k != "name"}
    potential, acceleration = field(*(inputs[k].astype(np.float64) for k in
                                      ("coordinates", "centers", "masses", "radii")))
    np.savez(OUT / f"{name}.npz", **inputs, potential=potential, acceleration=acceleration)
    return {"coordinates_shape": list(inputs["coordinates"].shape),
            "centers": inputs["centers"].tolist(), "masses_kg": inputs["masses"].tolist(),
            "radii_m": inputs["radii"].tolist(),
            "potential_range": [float(potential.min()), float(potential.max())],
            "acceleration_component_range": [float(acceleration.min()), float(acceleration.max())]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    cases = {}
    grid = np.stack(np.meshgrid([-1800, -600, 200, 1400], [-1300, 400, 1800],
                                [700, 2400], indexing="ij"), axis=-1)
    cases["point_grid"] = save("point_grid", grid, [[100, -200, -500]], [8e15], [0])
    # Center, interior, surface and immediately on either side, plus exterior.
    sphere_coords = [[0, 0, 0], [125, -250, 500], [1000, 0, 0],
                     [999.875, 0, 0], [1000.125, 0, 0], [0, -2000, 0],
                     [1200, 1600, 2000], [-5000, 1000, -3000]]
    cases["sphere"] = save("sphere", sphere_coords, [[0, 0, 0]], [5e15], [1000])
    # Unequal, displaced point and sphere sources exercise superposition.
    coords = rng.uniform([-2800, -2400, 1800], [3100, 2900, 5500], size=(19, 3))
    cases["mixed"] = save("mixed", coords, [[-500, 350, -700], [900, -800, -300],
                                              [100, 500, 100]],
                          [2e15, 7e15, 1.5e15], [0, 650, 1200])
    # Float64 centered potential differences provide an independent gradient check.
    data = np.load(OUT / "mixed.npz")
    xyz = data["coordinates"].astype(np.float64)
    args = [data[k].astype(np.float64) for k in ("centers", "masses", "radii")]
    step = 0.1  # metres; truncation O(h^2) is negligible at kilometre distances.
    gradient = np.empty_like(xyz)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        gradient[:, axis] = (field(xyz + offset, *args)[0] - field(xyz - offset, *args)[0]) / (2 * step)
    np.savez(OUT / "gradient.npz", coordinates=data["coordinates"], centers=data["centers"],
             masses=data["masses"], radii=data["radii"], acceleration=-gradient)
    np.testing.assert_allclose(-gradient, data["acceleration"], rtol=1e-7, atol=1e-11)
    # Analytic anchor at a sphere surface and center, and attraction direction.
    sphere = np.load(OUT / "sphere.npz")
    gm = G * float(sphere["masses"][0])
    np.testing.assert_allclose(sphere["potential"][[0, 2]], [-1.5 * gm / 1000, -gm / 1000])
    np.testing.assert_allclose(sphere["acceleration"][2], [-gm / 1000**2, 0, 0])
    tolerances = runpy.run_path(str(OUT.parent.parent / "tests" / "tolerances.py"))
    meta = {"oracle": "Newtonian point mass and homogeneous solid sphere shell theorem",
            "G_m3_kg-1_s-2": G, "seed": 20261028,
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
            "input_precision": "inputs rounded to float32 before float64 reference evaluation",
            "cases": cases, "gradient_step_m": step,
            "tolerances": {"relative_error": [tolerances["RELATIVE_ERROR"], "project-wide float32 elementwise requirement"],
                           "zero_scale_error": [tolerances["ZERO_SCALE_ERROR"], "float32 cancellation at zero, scaled by reference field maximum"],
                           "cross_relative_error": [tolerances["CROSS_RELATIVE_ERROR"], "sum of two reference error bounds"],
                           "cross_zero_scale_error": [tolerances["CROSS_ZERO_SCALE_ERROR"], "sum of two zero error bounds"],
                           "gradient_relative_error": [tolerances["GRADIENT_RELATIVE_ERROR"], "project-wide centered difference requirement"]}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
