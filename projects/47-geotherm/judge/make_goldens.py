"""Generate independent float64 cooling-half-space analytic reference data."""

import json
import platform
import runpy
from pathlib import Path

import numpy as np
import scipy
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "judge" / "golden"
SECONDS_PER_MYR = 1e6 * 365.25 * 86400
PARAMS = {"surface_temp_c": 0.0, "mantle_temp_c": 1350.0,
          "diffusivity_m2_s": 1e-6, "conductivity_w_m_k": 3.3}
STEP_MYR = 0.01


def forward(depth, age, p):
    t = np.asarray(age, dtype=np.float64)[:, None] * SECONDS_PER_MYR
    z = np.asarray(depth, dtype=np.float64)[None, :]
    eta = z / (2 * np.sqrt(p["diffusivity_m2_s"] * t))
    delta = p["mantle_temp_c"] - p["surface_temp_c"]
    temp = p["surface_temp_c"] + delta * erf(eta)
    flux = (p["conductivity_w_m_k"] * delta
            / np.sqrt(np.pi * p["diffusivity_m2_s"] * t) * np.exp(-eta**2))
    return temp, flux


def objective(age, observed, scale, p):
    predicted = forward(np.array([0.0]), age, p)[1][:, 0]
    return 0.5 * np.mean(((predicted - observed) / scale)**2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Values are explicitly specified; no random generator or sampling is used.
    depth = np.array([0, 100, 1000, 5000, 15000, 30000, 60000, 100000, 200000], dtype=np.float64)
    age = np.array([1, 5, 20, 50, 100, 200], dtype=np.float64)
    temperature, flux = forward(depth, age, PARAMS)
    np.savez_compressed(OUT / "forward.npz", depth_m=depth, age_myr=age,
                        temperature_c=temperature, heat_flow_w_m2=flux)
    # Different boundaries and properties exercise scaling and nonzero surface T.
    p2 = {"surface_temp_c": 12.0, "mantle_temp_c": 1275.0,
          "diffusivity_m2_s": 8e-7, "conductivity_w_m_k": 2.7}
    age2 = np.array([3, 17, 73, 160], dtype=np.float64)
    temperature2, flux2 = forward(depth, age2, p2)
    np.savez_compressed(OUT / "forward_alt.npz", depth_m=depth, age_myr=age2,
                        temperature_c=temperature2, heat_flow_w_m2=flux2)
    truth = np.array([2, 8, 25, 80, 180], dtype=np.float64)
    observed = forward(np.array([0.0]), truth, p2)[1][:, 0]
    np.savez_compressed(OUT / "inversion.npz", age_myr=truth, surface_heat_flow_w_m2=observed)
    trial = np.array([4, 12, 35, 110], dtype=np.float64)
    targets = forward(np.array([0.0]), np.array([6, 8, 55, 70]), PARAMS)[1][:, 0]
    scale = 0.1
    base = objective(trial, targets, scale, PARAMS)
    plus, minus = [], []
    for i in range(trial.size):
        offset = np.zeros_like(trial)
        offset[i] = STEP_MYR
        plus.append(objective(trial + offset, targets, scale, PARAMS))
        minus.append(objective(trial - offset, targets, scale, PARAMS))
    fd = (np.array(plus) - np.array(minus)) / (2 * STEP_MYR)
    np.savez_compressed(OUT / "gradient.npz", age_myr=trial, observed_heat_flow_w_m2=targets,
                        objective=np.array(base), objective_plus=np.array(plus),
                        objective_minus=np.array(minus), gradient_fd=fd)
    assert np.all(np.diff(temperature, axis=1) >= 0)
    assert np.all(np.diff(flux, axis=1) <= 0)
    assert np.all(np.diff(flux[:, 0]) < 0)
    assert np.all(temperature[:, 0] == PARAMS["surface_temp_c"])
    assert 0.01 < flux[-1, 0] < flux[0, 0] < 1.0
    tolerances = {k: v for k, v in runpy.run_path(str(ROOT / "tests" / "tolerances.py")).items()
                  if k.isupper()}
    meta = {"model": "T=Ts+(Tm-Ts) erf(z/(2 sqrt(kappa t))); upward q=k dT/dz",
            "seconds_per_myr": SECONDS_PER_MYR, "parameters_forward_and_gradient": PARAMS,
            "parameters_alternate_and_inversion": p2, "depth_m": depth.tolist(),
            "forward_age_myr": age.tolist(), "alternate_age_myr": age2.tolist(),
            "inversion_truth_myr": truth.tolist(), "gradient_trial_myr": trial.tolist(),
            "gradient_target_age_myr": [6, 8, 55, 70], "flux_scale_w_m2": scale,
            "finite_difference_step_myr": STEP_MYR, "random_seed": None,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "tolerances": tolerances,
            "tolerance_reasons": "See comments in tests/tolerances.py; forward and gradient relative limits follow the project-wide limits, the 5% age limit is the project pass criterion, and the 1e-4 age limit checks the noise-free analytic inverse."}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"temperature range: {temperature.min():.6g} to {temperature.max():.6g} C")
    print(f"surface heat flow range: {flux[:, 0].min():.6g} to {flux[:, 0].max():.6g} W/m²")
    print(f"inversion heat flow range: {observed.min():.6g} to {observed.max():.6g} W/m²")
    print(f"gradient objective: {base:.8g}; finite differences: {fd}")


if __name__ == "__main__":
    main()
