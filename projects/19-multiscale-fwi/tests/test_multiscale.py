"""MLX-only locked parity, derivative and cycle-skipping tests."""
import numpy as np
import pytest
from outcrop_harness import call, golden, golden_meta, rel_err

from .tolerances import TOL

PATHS = ["mlx"]


@pytest.fixture(scope="module", params=PATHS, ids=PATHS)
def path(request):
    return request.param


@pytest.fixture(scope="module", params=["case_0", "case_1"])
def case(request):
    return golden(request.param)


def arguments(g):
    return {"basis": g["basis"].astype(np.float32),
            "source": g["source"].astype(np.float32),
            "sources": g["sources"], "receivers": g["receivers"],
            "observed": g["observed"].astype(np.float32),
            "dx": float(g["dx"]), "dt": float(g["dt"])}


def evaluate(path, g, parameters, cutoff):
    out = call(path, "misfit_and_grad", **arguments(g),
               parameters=np.asarray(parameters, np.float32), cutoff=float(cutoff))
    assert set(out) == {"predicted", "objective", "gradient"}
    for key, shape in [("predicted", g["observed"].shape),
                       ("objective", ()), ("gradient", g["truth"].shape)]:
        check_array(out[key], shape)
    return out


def check_array(a, shape):
    assert isinstance(a, np.ndarray)
    assert a.shape == shape
    assert a.dtype == np.float32
    assert np.isfinite(a).all()


def assert_parity(actual, expected, atol=0.0):
    error = np.max(np.abs(np.asarray(actual, np.float64) - expected))
    assert error <= TOL["parity"] * np.max(np.abs(expected)) + atol


def assert_single_parameters(actual, expected, flat_mask):
    # The mask is saved by the oracle, not inferred from the implementation.
    # Use an absolute limit (not rtol + atol) only in the unresolved direction.
    allowance = np.where(flat_mask, TOL["single_flat_velocity"],
                         TOL["parity"] * np.max(np.abs(expected)))
    assert np.all(np.abs(np.asarray(actual, np.float64) - expected) <= allowance)


def measured_objective(predicted, g, cutoff):
    # This measures the returned waveform; it does not propagate or make data.
    observed = g["observed"].astype(np.float64)
    predicted = np.asarray(predicted, np.float64)
    n = observed.shape[-1]
    frequencies = np.fft.rfftfreq(n, float(g["dt"]))
    weights = 1 / np.sqrt(1 + (frequencies / cutoff) ** 8)
    residual = np.fft.irfft(np.fft.rfft(predicted - observed) * weights, n=n)
    data = np.fft.irfft(np.fft.rfft(observed) * weights, n=n)
    return 0.5 * np.sum(residual**2) / np.sum(data**2)


def model_error(parameters, g):
    delta = np.asarray(parameters, np.float64) - g["truth"]
    return np.linalg.norm(np.einsum("p,pzx->zx", delta, g["basis"]))


@pytest.mark.parametrize("pi", [0, 1, 2], ids=["initial", "probe", "truth"])
@pytest.mark.parametrize("ci", [0, 1, 2], ids=["3Hz", "9Hz", "18Hz"])
def test_waveform_objective_and_gradient(path, case, pi, ci):
    cutoff = case["probe_cutoffs"][ci]
    out = evaluate(path, case, case["probe_parameters"][pi], cutoff)
    assert_parity(out["predicted"], case["predictions"][pi])
    assert np.count_nonzero(out["predicted"][..., 0]) == 0
    measured = measured_objective(out["predicted"], case, cutoff)
    assert_parity(out["objective"], measured, TOL["history_atol"])
    if pi == 2:
        assert float(out["objective"]) <= TOL["zero_objective"]
        assert np.max(np.abs(out["gradient"])) <= TOL["zero_gradient"]
    else:
        assert_parity(out["objective"], case["objectives"][pi, ci])
        assert_parity(out["gradient"], case["gradients"][pi, ci])


@pytest.mark.parametrize("ci", [0, 1, 2], ids=["3Hz", "9Hz", "18Hz"])
def test_centered_gradient(path, case, ci):
    cutoff = case["probe_cutoffs"][ci]
    p = case["probe"].astype(np.float32)
    step = float(case["fd_step"])  # 1 m/s; chosen and audited in float64 judge.
    out = evaluate(path, case, p, cutoff)
    fd = np.empty_like(p, dtype=np.float64)
    # Use both independently varying coefficients, not one favorable direction.
    for j in range(p.size):
        plus, minus = p.copy(), p.copy()
        plus[j] += step
        minus[j] -= step
        jp = float(evaluate(path, case, plus, cutoff)["objective"])
        jm = float(evaluate(path, case, minus, cutoff)["objective"])
        fd[j] = (jp - jm) / (2 * step)
    # Componentwise relative errors also rule out hiding a weak component.
    assert np.all(np.abs(fd - out["gradient"]) <= TOL["gradient_fd"] * np.abs(out["gradient"]))
    assert rel_err(fd, case["finite_differences"][ci]) <= TOL["gradient_fd"]


@pytest.fixture(scope="module")
def inversions(path, case):
    out = {}
    for label, cutoffs in [("multi", case["cutoffs"]),
                           ("single", np.full_like(case["cutoffs"], case["cutoffs"][-1]))]:
        out[label] = call(path, "invert", **arguments(case),
                          initial_parameters=case["initial"].astype(np.float32),
                          cutoffs=cutoffs.tolist(), steps_per_stage=int(case["steps_per_stage"]))
    return out


def test_oracle_fixture_is_cycle_skipped(path, case):
    # This group is independent of the implementation; IDs still select mlx.
    assert path == "mlx"
    cycles = np.dot(case["travel_weights"], 1/case["initial"] - 1/case["truth"])
    assert cycles * float(case["peak_frequency"]) > TOL["half_cycle"]
    initial = model_error(case["initial"], case)
    single = model_error(case["single_parameters"], case)
    assert single >= TOL["baseline_error_fraction"] * initial
    assert model_error(case["multi_parameters"], case) < TOL["improvement"] * single
    # The float32 allowance applies only after entering the final flat basin.
    config = golden_meta()["optimizer"]
    endpoint, states = case["single_parameters"], case["single_trajectory"]
    free = (endpoint > config["lower"]) & (endpoint < config["upper"])
    near = np.max(np.abs(states - endpoint), axis=1) <= TOL["single_flat_velocity"]
    bound = np.all(states[:, ~free] == endpoint[~free], axis=1)
    np.testing.assert_array_equal(case["single_flat_mask"], (near & bound)[:, None] & free)
    assert not case["single_flat_mask"][0].any()
    assert case["single_flat_mask"][-1].any()
    sweep_p = case["single_resolution_parameters"].astype(np.float64)
    for gradient in [case["single_resolution_gradients"], case["single_resolution_gradients32"]]:
        projected = sweep_p - np.clip(sweep_p - gradient, config["lower"], config["upper"])
        assert np.max(np.abs(projected)) <= TOL["baseline_stationarity"]
    assert np.all(np.abs(case["finite_differences"] - case["gradients"][1]) <=
                  TOL["oracle_fd"] * np.max(np.abs(case["gradients"][1]), axis=-1, keepdims=True))
    meta = golden_meta()
    assert meta["paths"] == PATHS
    assert {key: record["value"] for key, record in meta["tolerances"].items()} == TOL


@pytest.mark.parametrize("label", ["multi", "single"])
def test_inversion_trajectory_and_parity(path, case, inversions, label):
    out = inversions[label]
    assert set(out) == {"parameters", "trajectory", "objective_history"}
    steps = int(case["steps_per_stage"])
    count = steps * case["cutoffs"].size
    check_array(out["parameters"], case["truth"].shape)
    check_array(out["trajectory"], (count + 1, case["truth"].size))
    check_array(out["objective_history"], (count,))
    np.testing.assert_array_equal(out["trajectory"][0], case["initial"].astype(np.float32))
    np.testing.assert_array_equal(out["trajectory"][-1], out["parameters"])
    config = golden_meta()["optimizer"]
    assert (out["trajectory"] >= config["lower"]).all()
    assert (out["trajectory"] <= config["upper"]).all()
    assert (np.abs(np.diff(out["trajectory"].astype(np.float64), axis=0)) <=
            config["max_step"] + TOL["parity"] * config["max_step"]).all()
    assert (out["objective_history"] >= 0).all()
    if label == "single":
        assert_single_parameters(out["parameters"], case["single_parameters"],
                                 case["single_flat_mask"][-1])
        assert_single_parameters(out["trajectory"], case["single_trajectory"],
                                 case["single_flat_mask"])
    else:
        assert_parity(out["parameters"], case["multi_parameters"])
        assert_parity(out["trajectory"], case["multi_trajectory"])
    # Near-zero histories need an absolute allowance; never compare across cutoffs.
    for stage in range(case["cutoffs"].size):
        values = out["objective_history"][stage*steps:(stage+1)*steps].astype(np.float64)
        assert (np.diff(values) <= TOL["history_atol"] + TOL["parity"] * values[:-1]).all()
        cutoff = case["cutoffs"][stage] if label == "multi" else case["cutoffs"][-1]
        for i in [stage*steps, (stage+1)*steps - 1]:
            result = evaluate(path, case, out["trajectory"][i], cutoff)
            assert_parity(out["objective_history"][i], result["objective"], TOL["history_atol"])
    assert_parity(out["objective_history"], case[f"{label}_objective_history"], TOL["history_atol"])


def test_multiscale_beats_converged_single_scale(path, case, inversions):
    multi, single = (inversions[label]["parameters"] for label in ["multi", "single"])
    single_error = model_error(single, case)
    assert single_error >= TOL["baseline_error_fraction"] * model_error(case["initial"], case)
    assert model_error(multi, case) < TOL["improvement"] * single_error
    # Recompute stationarity from the actual single-scale endpoint: a deliberately
    # short or broken baseline cannot manufacture the improvement ratio.
    final = evaluate(path, case, single, case["cutoffs"][-1])
    config = golden_meta()["optimizer"]
    p = single.astype(np.float64)
    projected = p - np.clip(p - final["gradient"].astype(np.float64),
                            config["lower"], config["upper"])
    assert np.max(np.abs(projected)) <= TOL["baseline_stationarity"]
