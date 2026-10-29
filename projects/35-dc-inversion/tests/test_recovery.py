"""Numeric synthetic recovery pass criteria, independent of optimizer choice."""
import numpy as np

from .tolerances import (
    ANOMALY_LOG10_MAX,
    BACKGROUND_LOG10_RMS,
    LOCATION_CELLS,
    MODEL_ATOL,
    PARITY_RTOL,
    SUPPORT_CONTRAST_FRACTION,
    TARGET_CHI2,
    VOLTAGE_ATOL,
)


def test_reaches_target_misfit(recovery):
    g, _, predicted = recovery
    chi2 = np.mean(((predicted - g["observed"]) / g["standard_deviation"])**2)
    assert chi2 <= TARGET_CHI2, f"reduced chi2={chi2} > {TARGET_CHI2}"


def test_anomaly_location_within_two_cells(recovery):
    g, out, _ = recovery
    anomaly = g["anomaly"].astype(bool)
    contrast = float((g["truth"] - g["initial"])[anomaly][0])
    signed = np.sign(contrast) * (out["log10_rho"].astype(np.float64) - g["initial"])
    support = g["active"].astype(bool) & (signed >= SUPPORT_CONTRAST_FRACTION * abs(contrast))
    assert support.any(), "no recovered cells have the anomaly polarity and half its contrast"
    z, x = np.indices(signed.shape)
    weights = np.where(support, signed, 0.0)
    center = np.array([(z * weights).sum(), (x * weights).sum()]) / weights.sum()
    truth_center = np.array([z[anomaly].mean(), x[anomaly].mean()])
    distance = np.linalg.norm(center - truth_center)
    assert distance <= LOCATION_CELLS, f"centroid error {distance} cells"


def test_log10_error_in_every_anomaly_cell(recovery):
    g, out, _ = recovery
    mask = g["anomaly"].astype(bool)
    error = np.max(abs(out["log10_rho"].astype(np.float64)[mask] - g["truth"][mask]))
    assert error <= ANOMALY_LOG10_MAX, f"worst anomaly log10 error {error} dex"


def test_localized_background(recovery):
    g, out, _ = recovery
    mask = g["active"].astype(bool) & ~g["anomaly"].astype(bool)
    rms = np.sqrt(np.mean((out["log10_rho"].astype(np.float64)[mask] - g["truth"][mask])**2))
    assert rms <= BACKGROUND_LOG10_RMS, f"background RMS {rms} dex"


def test_bounds_and_fixed_cells(recovery):
    g, out, _ = recovery
    model = out["log10_rho"].astype(np.float64)
    active = g["active"].astype(bool)
    assert model[active].min() >= float(g["lower_bound"]) - MODEL_ATOL
    assert model[active].max() <= float(g["upper_bound"]) + MODEL_ATOL
    np.testing.assert_allclose(model[~active], g["initial"][~active], rtol=0, atol=MODEL_ATOL)


def test_returned_prediction_is_consistent(recovery):
    _, out, predicted = recovery
    np.testing.assert_allclose(out["predicted"], predicted, rtol=PARITY_RTOL, atol=VOLTAGE_ATOL)
