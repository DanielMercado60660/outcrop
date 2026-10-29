"""Published Okada checks, spatial derivatives, and cross-path parity."""
import numpy as np
import pytest
from outcrop_harness import PATHS, call, golden
from tolerances import (
    FD_ABS,
    FD_CANCELLATION_ULPS,
    FD_REL,
    FD_SPACING_ABS,
    FD_STEPS,
    FLOAT32_EPS,
    MANUAL_CANCELLATION_ULPS,
    MANUAL_PRINTED_HALF_UNIT,
    MANUAL_REFERENCE_ABS,
    PARITY_ABS,
    PARITY_REL,
    TABLE_CANCELLATION_ULPS,
    TABLE_ZERO_ABS,
)

CASE_NAMES = [
    "case1_strike", "case1_dip", "case1_tensile",
    "case2_strike", "case2_dip", "case2_tensile",
    "case3_strike", "case3_dip", "case3_tensile",
    "case4_strike", "case4_dip", "case4_tensile",
]


def table_args(g, i, coordinates=None):
    return {
        "coordinates": np.asarray(g["coordinates"][i:i+1] if coordinates is None else coordinates, dtype=np.float32),
        "source": str(g["source"][i]),
        "depth": float(g["depth"][i]),
        "dip_deg": float(g["dip_deg"][i]),
        "bounds": g["bounds"][i].astype(np.float32),
        "slip": g["slip"][i].astype(np.float32),
        "alpha": 2 / 3,
    }


def manual_args(g):
    return {
        "coordinates": g["manual_coordinates"].astype(np.float32),
        "source": "finite_rectangle",
        "depth": float(g["manual_depth"]),
        "dip_deg": float(g["manual_dip"]),
        "bounds": g["manual_bounds"].astype(np.float32),
        "slip": g["manual_slip"].astype(np.float32),
        "alpha": 2 / 3,
    }


def checked_call(path, args):
    out = call(path, "okada", **args)
    n = len(args["coordinates"])
    assert set(out) == {"displacement", "horizontal_derivatives"}
    for key, shape in (("displacement", (n, 3)), ("horizontal_derivatives", (n, 3, 2))):
        assert out[key].shape == shape
        assert out[key].dtype == np.float32
        assert np.isfinite(out[key]).all()
    return out


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("i", range(12), ids=CASE_NAMES)
def test_table(path, i):
    g = golden("checks")
    assert g["names"][i] == CASE_NAMES[i]
    out = checked_call(path, table_args(g, i))
    actual = np.concatenate((out["displacement"][0], out["horizontal_derivatives"][0].reshape(-1))).astype(np.float64)
    expected = g["values"][i]
    cancellation = TABLE_CANCELLATION_ULPS*FLOAT32_EPS*np.sum(abs(g["slip"][i]))*g["corner_scale"][i]
    limit = np.where(expected == 0, TABLE_ZERO_ABS, g["half_units"][i] + cancellation)
    assert np.all(np.abs(actual - expected) <= limit), (
        f"{CASE_NAMES[i]} actual={actual}, expected={expected}, limits={limit}"
    )


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
def test_dc3d_manual(path):
    g = golden("checks")
    out = checked_call(path, manual_args(g))
    cancellation = MANUAL_CANCELLATION_ULPS*FLOAT32_EPS*np.sum(abs(g["manual_slip"]))*g["manual_corner_scale"]
    limit = MANUAL_PRINTED_HALF_UNIT + MANUAL_REFERENCE_ABS + cancellation
    error = abs(out["displacement"].astype(np.float64)-g["manual_values"])
    assert np.all(error <= limit), f"manual error={error}, absolute limits={limit}"


@pytest.mark.parametrize("path", PATHS, ids=PATHS)
@pytest.mark.parametrize("fd_index,case_index", [(0, 0), (1, 3)], ids=["point", "rectangle"])
def test_horizontal_derivatives_finite_difference(path, fd_index, case_index):
    g = golden("checks")
    coords = g["fd_coordinates"][fd_index].astype(np.float32)
    base = checked_call(path, table_args(g, case_index))
    perturbed = checked_call(path, table_args(g, case_index, coords))
    u = perturbed["displacement"].astype(np.float64)
    # Divide by actual float32 coordinate spacing to avoid a quantization bias.
    dx = float(coords[0, 0]) - float(coords[1, 0])
    dy = float(coords[2, 1]) - float(coords[3, 1])
    assert abs(dx / 2 - FD_STEPS[fd_index]) < FD_SPACING_ABS
    assert abs(dy / 2 - FD_STEPS[fd_index]) < FD_SPACING_ABS
    fd = np.stack(((u[0] - u[1]) / dx, (u[2] - u[3]) / dy), axis=-1)
    cancellation = FD_CANCELLATION_ULPS[fd_index]*FLOAT32_EPS*np.sum(abs(g["slip"][case_index]))*g["fd_corner_scale"][fd_index]
    limit = FD_ABS[fd_index] + FD_REL*abs(fd) + cancellation
    error = abs(base["horizontal_derivatives"][0].astype(np.float64)-fd)
    assert np.all(error <= limit), f"FD error={error}, limits={limit}, cancellation={cancellation}"


@pytest.mark.parametrize("i", range(13), ids=CASE_NAMES + ["dc3d_manual"])
def test_mlx_metal_parity(i):
    g = golden("checks")
    args = table_args(g, i) if i < 12 else manual_args(g)
    mlx = checked_call("mlx", args)
    metal = checked_call("metal", args)
    for key in mlx:
        np.testing.assert_allclose(mlx[key], metal[key], rtol=PARITY_REL, atol=PARITY_ABS)
