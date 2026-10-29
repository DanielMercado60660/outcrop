"""Locked numeric criteria; metadata consumes this same registry."""
TOLERANCES = {
    "parity": {"value": 1e-4, "reason": "Project-wide float32 relative norm/max-scale parity against float64 SimPEG."},
    "adjoint": {"value": 1e-5, "reason": "Project pass: normalized dot-product residual, Cauchy scale avoids cancellation."},
    "gradient": {"value": 1e-3, "reason": "Project-wide centered float32 finite-difference directional derivative check."},
    "fd_step": {"value": 0.01, "reason": "g/cc along saved unit direction; quadratic objective has zero centered truncation error; avoids float32 cancellation."},
    "chi_fraction": {"value": 0.10, "reason": "Project pass: absolute chi-squared deviation <=10% of N observations."},
    "correlation": {"value": 0.70, "reason": "Project pass: Pearson correlation of recovered and true cell densities >=0.7."},
    "model_error_fraction": {"value": 0.20, "reason": "Project pass: recovered L2 error <=1.20 times SimPEG error, better solutions are allowed."},
    "stationarity": {"value": 1e-4, "reason": "Normal-equation residual relative to RHS ensures actual minimization at locked beta; avoids rewarding arbitrary data fits."},
    "oracle_check": {"value": 1e-10, "reason": "Float64 stored-array algebra consistency; no implementation involved."},
    "oracle_regularization": {"value": 1e-12, "reason": "Float64 absolute and relative agreement of independent edge construction and SimPEG Hessian."},
    "oracle_gradient": {"value": 1e-7, "reason": "Float64 centered differences of SimPEG's quadratic objective."},
    "oracle_fd_step": {"value": 1e-5, "reason": "Float64 quadratic centered-difference step in g/cc along saved unit direction."},
    "beta_root_xtol": {"value": 1e-12, "reason": "Log-beta scalar discrepancy search precision; cheap float64 reference only."},
}


def tol(name):
    return TOLERANCES[name]["value"]
