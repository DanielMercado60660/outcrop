"""Fixtures call the frozen shared harness; this project is MLX-only."""
import numpy as np
import pytest
from outcrop_harness import call, golden

# MLX-only: this overrides the shared harness's default of both paths.
PATHS = ["mlx"]
GEOMETRY_KEYS = ("hx", "hz", "origin", "quadrature", "electrodes", "abmn", "current")


def inputs(g):
    """Convert saved float64 oracle inputs to the required implementation dtypes."""
    return {key: (float(g[key]) if key == "current" else
                  g[key].astype(np.int32 if key == "abmn" else np.float32))
            for key in GEOMETRY_KEYS}


def objective_inputs(g, model=None):
    return dict(inputs(g),
                log10_rho=np.asarray(g["log10_rho"] if model is None else model, np.float32),
                observed=g["observed"].astype(np.float32),
                standard_deviation=g["standard_deviation"].astype(np.float32),
                reference=g["reference"].astype(np.float32), beta=float(g["beta"]))


@pytest.fixture(scope="module", params=PATHS, ids=PATHS)
def path(request):
    return request.param


@pytest.fixture(scope="module", params=["conductor", "resistor"])
def recovery(path, request):
    g = golden(request.param)
    out = call(path, "dc_invert", **inputs(g),
               observed=g["observed"].astype(np.float32),
               standard_deviation=g["standard_deviation"].astype(np.float32),
               initial=g["initial"].astype(np.float32), active=g["active"].astype(np.int32),
               beta=float(g["beta"]), lower_bound=float(g["lower_bound"]),
               upper_bound=float(g["upper_bound"]), max_iterations=int(g["max_iterations"]))
    assert set(out) == {"log10_rho", "predicted"}
    for key, shape in (("log10_rho", g["truth"].shape), ("predicted", g["observed"].shape)):
        assert out[key].shape == shape
        assert out[key].dtype == np.float32
        assert np.isfinite(out[key]).all()
    # Recompute response at the returned model; don't trust reported predictions.
    response = call(path, "dc_predict", **inputs(g), log10_rho=out["log10_rho"])
    assert set(response) == {"predicted"}
    assert response["predicted"].shape == g["observed"].shape
    assert response["predicted"].dtype == np.float32
    assert np.isfinite(response["predicted"]).all()
    return g, out, response["predicted"].astype(np.float64)


@pytest.fixture(scope="module", params=["gradient_data", "gradient_regularized"])
def evaluated(path, request):
    g = golden(request.param)
    out = call(path, "dc_misfit_and_grad", **objective_inputs(g))
    assert set(out) == {"objective", "gradient", "predicted"}
    for key, shape in (("objective", ()), ("gradient", g["log10_rho"].shape),
                       ("predicted", g["observed"].shape)):
        assert out[key].shape == shape
        assert out[key].dtype == np.float32
        assert np.isfinite(out[key]).all()
    return g, out
