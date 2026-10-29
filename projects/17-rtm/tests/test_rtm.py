"""Locked RTM tests. Only saved judge inputs and the frozen harness are used."""

import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden, rel_err

from .tolerances import (
    CROSS_RTOL,
    DEPTH_CELLS,
    FD_STEP,
    GRADIENT_RTOL,
    MIN_LAYER_FRACTION,
    REFERENCE_RTOL,
    ZERO_ATOL,
)

LAYER_CASES = ["layers_uniform", "layers_variable"]
REFLECTORS = [
    ("layers_uniform", 0), ("layers_uniform", 1),
    ("layers_variable", 0), ("layers_variable", 1), ("layers_variable", 2),
]


@pytest.fixture(scope="module", params=PATHS, ids=PATHS)
def path(request):
    """Both paths: backend name is present in every implementation test ID."""
    return request.param


@pytest.fixture(scope="module")
def small():
    return golden("adjoint")


def arguments(g):
    """Copies prevent implementations from mutating cached saved inputs."""
    return {
        "velocity": g["velocity"].astype(np.float32, copy=True),
        "source_amplitudes": g["source_amplitudes"].astype(np.float32, copy=True),
        "source_locations": g["source_locations"].astype(np.int64, copy=True),
        "receiver_locations": g["receiver_locations"].astype(np.int64, copy=True),
        "data": g["data"].astype(np.float32, copy=True),
        "grid_spacing": g["grid_spacing"].tolist(), "dt": float(g["dt"]),
        "pml_width": int(g["pml_width"]), "pml_freq": float(g["pml_freq"]),
    }


def array_contract(out, shapes):
    assert set(out) == set(shapes), f"output keys: {set(out)}, expected {set(shapes)}"
    for key, shape in shapes.items():
        arr = out[key]
        assert isinstance(arr, np.ndarray), key
        assert arr.dtype == np.float32, (key, arr.dtype)
        assert arr.shape == shape, (key, arr.shape, shape)
        assert np.all(np.isfinite(arr)), key


def image_call(path, **kwargs):
    out = call(path, "rtm", **kwargs)
    array_contract(out, {"image": kwargs["velocity"].shape})
    return out["image"]


def misfit_call(path, **kwargs):
    out = call(path, "born_misfit_and_grad", **kwargs)
    array_contract(out, {"predicted": kwargs["data"].shape,
                         "gradient": kwargs["velocity"].shape, "loss": ()})
    assert float(out["loss"]) >= ZERO_ATOL
    return out


def close(actual, expected, tolerance=REFERENCE_RTOL):
    error = rel_err(actual, expected)
    assert error <= tolerance, f"global max relative error {error:.8g} > {tolerance:.8g}"


def cross_close(a, b, reference):
    # Normalize by the same float64 oracle as each individual parity check,
    # making the sum of their 1e-4 error budgets exactly 2e-4.
    denominator = float(np.max(np.abs(reference)))
    numerator = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    error = numerator / denominator if denominator else numerator
    assert error <= CROSS_RTOL, f"cross relative error {error:.8g} > {CROSS_RTOL}"


@pytest.fixture(scope="module")
def small_outputs(path, small):
    args = arguments(small)
    return {"image": image_call(path, **args),
            **misfit_call(path, scatter=small["scatter"].copy(), **arguments(small))}


@pytest.fixture(scope="module")
def layer_images(path):
    return {case: image_call(path, **arguments(golden(case))) for case in LAYER_CASES}


@pytest.mark.parametrize("case", LAYER_CASES)
def test_layer_image_matches_float64(path, layer_images, case):
    close(layer_images[case], golden(case)["image"])


@pytest.mark.parametrize("case,reflector", REFLECTORS,
                         ids=[f"{case}-reflector-{i}" for case, i in REFLECTORS])
def test_every_reflector_within_two_cells(path, layer_images, case, reflector):
    g = golden(case)
    image = layer_images[case].astype(np.float64)
    xlo, xhi = g["lateral_roi"]
    # Full RMS-depth profile in a saved central lateral aperture. Each peak
    # is found over a BROAD Voronoi zone (not within the allowed error band).
    profile = np.mean(image[:, xlo:xhi] ** 2, axis=1)
    lo, hi = g["depth_zones"][reflector]
    peak = int(lo + np.argmax(profile[lo:hi]))
    truth = int(g["reflector_rows"][reflector])
    assert profile.max() > ZERO_ATOL, "zero image is not a recovered reflector"
    assert profile[peak] > MIN_LAYER_FRACTION ** 2 * profile.max()
    assert abs(peak - truth) <= DEPTH_CELLS, (case, reflector, peak, truth)


def test_small_rtm_matches_float64(path, small, small_outputs):
    close(small_outputs["image"], small["image"])


def test_born_prediction_loss_gradient_match_float64(path, small, small_outputs):
    for key in ["predicted", "loss", "gradient"]:
        close(small_outputs[key], small[key])
    residual = small_outputs["predicted"].astype(np.float64) - small["data"]
    close(small_outputs["loss"], 0.5 * np.sum(residual ** 2))


@pytest.mark.parametrize("direction", range(3), ids=["dense", "sparse", "aligned"])
def test_gradient_centered_finite_difference(path, small, small_outputs, direction):
    # All points and directions came from the oracle. Perturbations are
    # binary-exact in float32; the oracle records the same step in meta.json.
    d = small["directions"][direction]
    losses = []
    for sign in [-1, 1]:
        scatter = small["scatter"] + sign * FD_STEP * d
        out = misfit_call(path, scatter=scatter, **arguments(small))
        losses.append(float(out["loss"]))
    fd = (losses[1] - losses[0]) / (2 * FD_STEP)
    ad = float(np.sum(small_outputs["gradient"].astype(np.float64) * d))
    oracle_fd = float(small["fd_derivatives"][direction])
    # Pairwise AND independent float64 checks avoid self-consistent fake
    # objective/gradient pairs. No denominator floor hides tiny derivatives.
    assert abs(oracle_fd) > ZERO_ATOL
    assert abs(ad - fd) / max(abs(ad), abs(fd)) <= GRADIENT_RTOL
    assert abs(fd - oracle_fd) / abs(oracle_fd) <= GRADIENT_RTOL


def test_discrete_adjoint_dot_product(path, small, small_outputs):
    # <J^T d, dm> == <d, J dm>, using independently saved forward J dm.
    left = float(np.sum(small_outputs["image"].astype(np.float64) * small["directions"][0]))
    right = float(np.sum(small["data"].astype(np.float64) * small["jdirection"]))
    assert abs(left - right) / max(abs(left), abs(right)) <= REFERENCE_RTOL


def test_zero_data_is_zero_image(path, small):
    args = arguments(small)
    args["data"] = np.zeros_like(args["data"])
    image = image_call(path, **args)
    assert np.max(np.abs(image)) <= ZERO_ATOL


@pytest.mark.parametrize("scale", [-1.5, 0.25], ids=["negative", "fractional"])
def test_signed_data_scaling(path, small, scale):
    args = arguments(small)
    args["data"] *= scale
    close(image_call(path, **args), scale * small["image"])


def test_source_scaling(path, small):
    args = arguments(small)
    args["source_amplitudes"] *= -0.5
    close(image_call(path, **args), -0.5 * small["image"])


def test_shot_sum_and_permutation(path, small, small_outputs):
    fields = ["data", "source_amplitudes", "source_locations", "receiver_locations"]
    images = []
    for shot in range(small["data"].shape[0]):
        args = arguments(small)
        for key in fields:
            args[key] = args[key][shot:shot + 1].copy()
        images.append(image_call(path, **args))
    close(np.sum(np.array(images, dtype=np.float64), axis=0), small_outputs["image"])
    args = arguments(small)
    for key in fields:
        args[key] = args[key][::-1].copy()
    close(image_call(path, **args), small_outputs["image"])


def test_rtm_equals_residual_gradient(path, small, small_outputs):
    args = arguments(small)
    args["data"] = (small_outputs["predicted"] - args["data"]).astype(np.float32)
    close(image_call(path, **args), small_outputs["gradient"])


def test_rtm_is_negative_gradient_at_zero(path, small, small_outputs):
    out = misfit_call(path, scatter=np.zeros_like(small["scatter"]), **arguments(small))
    assert np.max(np.abs(out["predicted"])) <= ZERO_ATOL
    close(-out["gradient"], small_outputs["image"])
    close(out["loss"], 0.5 * np.sum(small["data"].astype(np.float64) ** 2))


@pytest.mark.parametrize("case", [*LAYER_CASES, "adjoint"])
def test_cross_backend_images(case):
    g = golden(case)
    a = image_call("mlx", **arguments(g))
    b = image_call("metal", **arguments(g))
    cross_close(a, b, g["image"])


def test_cross_backend_born_outputs(small):
    a = misfit_call("mlx", scatter=small["scatter"].copy(), **arguments(small))
    b = misfit_call("metal", scatter=small["scatter"].copy(), **arguments(small))
    for key in a:
        cross_close(a[key], b[key], small[key])
