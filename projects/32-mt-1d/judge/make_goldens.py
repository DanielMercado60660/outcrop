"""Generate deterministic 1D MT reference responses with SimPEG and analytics."""
import json
from pathlib import Path

import numpy as np
import scipy
import simpeg
from simpeg.electromagnetics.natural_source import Simulation1DRecursive

MU0 = 1.25663706212e-6
OUT = Path(__file__).parent / "golden"


def main():
    OUT.mkdir(exist_ok=True)
    # Exactly representable float32 inputs are promoted for the reference.
    freq = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000], dtype=np.float32)
    cases = {
        "half_low": ([10], []),
        "half_high": ([1000], []),
        "conductive_cover": ([10, 1000], [500]),
        "resistive_cover": ([1000, 10], [500]),
        "three_layers": ([100, 5, 500], [100, 1200]),
        "four_layers": ([50, 500, 20, 2000], [30, 300, 1500]),
        "uniform_layers": ([100, 100, 100], [10, 5000]),
    }
    sim = Simulation1DRecursive()
    metadata = {}
    for name, (rho, thick) in cases.items():
        rho = np.array(rho, dtype=np.float32)
        thick = np.array(thick, dtype=np.float32)
        f64 = freq.astype(np.float64)
        # SimPEG expects bottom-first conductivity and thickness, returns Z_yx.
        z = -sim._get_recursive_impedances(
            f64, thick.astype(np.float64)[::-1], 1.0 / rho.astype(np.float64)[::-1]
        )
        apparent = np.abs(z) ** 2 / (MU0 * 2 * np.pi * f64)
        phase = np.degrees(np.angle(z))
        if len(rho) == 1 or name == "uniform_layers":
            # SciPy's version of mu_0 can differ slightly from the API constant.
            np.testing.assert_allclose(apparent, rho[0], rtol=1e-8)
            np.testing.assert_allclose(phase, 45.0, rtol=1e-12)
            # Save the independent analytic answer for these cases.
            apparent = np.full(f64.shape, float(rho[0]))
            phase = np.full(f64.shape, 45.0)
        assert np.all(np.isfinite(apparent)) and np.all(apparent > 0)
        assert np.all((phase > 0) & (phase < 90))
        np.savez(OUT / f"{name}.npz", resistivity=rho, thickness=thick,
                 frequency=freq, apparent_resistivity=apparent, phase=phase)
        metadata[name] = {"resistivity_ohm_m": rho.tolist(),
                          "thickness_m": thick.tolist(), "frequency_hz": freq.tolist(),
                          "apparent_range": [float(apparent.min()), float(apparent.max())],
                          "phase_range_degrees": [float(phase.min()), float(phase.max())]}
        print(name, metadata[name]["apparent_range"], metadata[name]["phase_range_degrees"])
    assert np.load(OUT / "conductive_cover.npz")["apparent_resistivity"][-1] < 20
    assert np.load(OUT / "resistive_cover.npz")["apparent_resistivity"][-1] > 900
    meta = {"versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "simpeg": simpeg.__version__}, "seed": "none; explicit deterministic inputs",
            "mu0_H_per_m": MU0, "reference": "SimPEG Simulation1DRecursive private recursion; analytic uniform half-space",
            "tolerances": {"response_rtol": [1e-4, "project-wide float32 elementwise relative threshold"],
                           "cross_rtol": [1e-4, "float32 implementation parity"]}, "cases": metadata}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
