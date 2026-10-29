"""Generate fixed SimPEG 2.5D DC reference data and half-space checks."""
import json
from pathlib import Path

import discretize
import numpy as np
import scipy
import simpeg
from simpeg.electromagnetics.static import resistivity as dc

HERE = Path(__file__).resolve().parent / "golden"


def main():
    HERE.mkdir(exist_ok=True)
    # x core centers include every integer; z core ends exactly at the surface.
    pad = 0.25 * 1.5 ** np.arange(1, 13)
    hx = np.r_[pad[::-1], np.full(97, 0.25), pad]
    hz = np.r_[pad[::-1], np.full(48, 0.25)]
    origin = np.array([-pad.sum() - 12.125, -hz.sum()])
    mesh = discretize.TensorMesh([hx, hz], origin=origin)
    x, z = mesh.cell_centers.T
    # Surface dipoles, three spreads and a second, asymmetric source geometry.
    ab = np.array([[-3., 3.], [-4., 4.], [-6., 6.], [-5., 3.]])
    mn = np.array([[-1., 1.], [-2., 2.], [-2., 2.], [-1., 1.]])
    currents = np.array([1., 1., 1., 2.])
    sources = []
    for (a, b), (m, n), current in zip(ab, mn, currents):
        rx = dc.receivers.Dipole(np.array([[m, 0.]]), np.array([[n, 0.]]))
        sources.append(dc.sources.Dipole([rx], np.array([a, 0.]),
                                         np.array([b, 0.]), current=current))
    survey = dc.Survey(sources)
    sim = dc.Simulation2DCellCentered(mesh, survey=survey, bc_type="Robin",
                                     nky=61, do_trap=True)
    # Flattened conductivity is x-fastest (Fortran ordering).
    homogeneous = np.full(mesh.n_cells, 0.01)
    block = homogeneous.copy()
    block[(x > -1.5) & (x < 2.5) & (z < -0.5) & (z > -4.5)] = 0.04
    layered = homogeneous.copy()
    layered[z < -2.0] = 0.025
    geom = (1 / abs(ab[:, 0] - mn[:, 0]) - 1 / abs(ab[:, 1] - mn[:, 0])
            - 1 / abs(ab[:, 0] - mn[:, 1]) + 1 / abs(ab[:, 1] - mn[:, 1])) / (2 * np.pi)
    results = {}
    for name, sigma in [("homogeneous", homogeneous), ("block", block), ("layered", layered)]:
        sim.sigma = sigma
        voltage = sim.dpred()
        rhoa = voltage / currents / geom
        results[name] = rhoa
        np.savez_compressed(HERE / f"{name}.npz", hx=hx, hz=hz, origin=origin,
                            conductivity=sigma.reshape(mesh.shape_cells, order="F"),
                            ab=ab, mn=mn, currents=currents,
                            ky=sim._quad_points, weights=sim._quad_weights,
                            voltage=voltage, apparent_resistivity=rhoa,
                            analytic=np.full(len(ab), 100.))
        print(name, "voltage", voltage, "rhoa", rhoa)
    error = np.max(np.abs(results["homogeneous"] / 100. - 1))
    print("maximum analytic relative error:", error)
    assert error < 0.01, "mesh/quadrature does not meet the analytic criterion"
    meta = {
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "discretize": discretize.__version__, "simpeg": simpeg.__version__},
        "seed": "none; deterministic geometry and models",
        "reference": "SimPEG Simulation2DCellCentered, Robin sides/bottom, insulating top, 61-point trapezoidal Fourier quadrature",
        "inputs": {"hx_count": len(hx), "hz_count": len(hz), "origin_m": origin.tolist(),
                   "core_cell_m": 0.25, "padding_growth": 1.5, "padding_cells": 12,
                   "source_ab_m": ab.tolist(), "receiver_mn_m": mn.tolist(),
                   "currents_A": currents.tolist(), "models": "100 ohm-m half-space; 25 ohm-m block; 40 ohm-m lower layer"},
        "tolerances": {"reference_relative": {"value": 0.0001, "reason": "project-wide float32 same-mesh parity"},
                       "analytic_relative": {"value": 0.01, "reason": "finite-volume mesh and Fourier quadrature discretization"},
                       "cross_path_relative": {"value": 0.0001, "reason": "float32 independent implementation parity"}},
        "homogeneous_max_analytic_relative_error": error,
    }
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
