"""STEAD phase-picking tests, MLX only."""
import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from outcrop_harness import call, golden, golden_meta

from .metrics import decode, score, score_decoded
from .tolerances import (
    BROKEN_F1_GAP,
    F1_MIN,
    FD_STEP,
    GRAD_RTOL,
    MICRO_F1_MIN,
    P_F1_MIN,
    P_MEDIAN_MAX_S,
    PARITY_ATOL,
    PARITY_RTOL,
    PROB_SUM_ATOL,
    REFERENCE_F1_MARGIN,
    REFERENCE_MAX_TRAINING_S,
    RETIRED_ROUND3_F1_GATES,
    S_F1_MIN,
    S_MEDIAN_MAX_S,
)

# MLX only: this overrides the shared harness's default two-path list. Every test requests
# this fixture, so every collected ID contains mlx; -k metal selects no cases.
PATHS = ["mlx"]


@pytest.fixture(scope="module", params=PATHS, ids=PATHS)
def path(request):
    return request.param


@pytest.fixture(scope="module")
def verified_data():
    meta = golden_meta()
    assert meta["real_data_qualified"] and meta["dataset_kind"] == "licensed_heldout"
    golden_dir = Path(__file__).resolve().parents[1] / "judge" / "golden"
    for name, info in meta["files"].items():
        with (golden_dir / name).open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == info["sha256"]
    # Repo-relative access supports a clean checkout elsewhere; no path hacks.
    root = Path(__file__).resolve().parents[3] / "data" / "phasepicker"
    missing = [f"{s}.npz" for s in ("train", "val", "test") if not (root / f"{s}.npz").exists()]
    if missing:
        pytest.skip(f"STEAD data not built ({', '.join(missing)} missing in data/phasepicker/); "
                    "see data/phasepicker/README.md to build it")
    event_sets = []
    for split in ("train", "val", "test"):
        file = root / f"{split}.npz"
        with file.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == meta["data_files"][split]["sha256"]
        with np.load(file, allow_pickle=False) as z:
            labels = np.column_stack([z["p_idx"], z["s_idx"]])
            np.testing.assert_array_equal(labels, golden(f"stead_{split}_labels")["labels"])
            assert labels.dtype == np.int32
            assert float(z["sampling_rate"]) == 100 and str(z["component_order"]) == "ZNE"
            np.testing.assert_array_equal(labels[:, 1] == -1, z["s_outside_window"])
            event_sets.append(set(z["source_id"].tolist()))
    assert not (event_sets[0] & event_sets[1] or event_sets[0] & event_sets[2] or event_sets[1] & event_sets[2])
    return root


@pytest.fixture(scope="module")
def trained(path, verified_data):
    cfg = golden_meta()["training"]
    schedule = golden(Path(cfg["schedule"]).stem)
    with np.load(verified_data / "train.npz", allow_pickle=False) as z:
        waveforms = np.ascontiguousarray(z["waveforms"].transpose(0, 2, 1))
        labels = np.column_stack([z["p_idx"], z["s_idx"]])
    start = time.perf_counter()
    out = call(path, "train_phase_picker", waveforms=waveforms, labels=labels,
               batch_indices=schedule["batch_indices"], shifts=schedule["shifts"],
               polarities=schedule["polarities"], learning_rates=schedule["learning_rates"],
               rotation_angles=schedule["rotation_angles"],
               sample_interval_s=0.01, seed=cfg["seed"],
               label_sigma_samples=cfg["label_sigma_samples"])
    print(f"{path} training call wall seconds: {time.perf_counter()-start:.3f}")
    assert out["steps"].shape == () and out["steps"].dtype == np.int32
    assert int(out["steps"]) == schedule["batch_indices"].shape[0] == cfg["steps"]
    assert out["loss"].shape == () and out["loss"].dtype == np.float32
    assert np.isfinite(out["loss"]) and float(out["loss"]) >= 0
    return out


@pytest.fixture(scope="module")
def evaluation(verified_data):
    with np.load(verified_data / "test.npz", allow_pickle=False) as z:
        x = np.ascontiguousarray(z["waveforms"].transpose(0, 2, 1))
    assert x.shape == (3000, 3000, 3) and x.dtype == np.float32
    g = golden("stead_test_labels")
    return {"waveforms": x, **g}


def predict(path, x, dt):
    p = call(path, "predict_phases", waveforms=x, sample_interval_s=dt)["probabilities"]
    assert p.shape == x.shape and p.dtype == np.float32
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=-1), 1, rtol=0, atol=PROB_SUM_ATOL)
    return p


@pytest.fixture(scope="module")
def inference(path, trained, evaluation):
    # Train exactly once, then stream ALL held-out traces in fixed batches.
    # Neither evaluation labels nor validation data are passed to the model.
    x, dt = evaluation["waveforms"], float(evaluation["sample_interval_s"])
    return np.concatenate([predict(path, x[i:i+32], dt) for i in range(0, len(x), 32)])


@pytest.fixture(scope="module")
def scores(inference, evaluation):
    return score(inference, evaluation["labels"], float(evaluation["sample_interval_s"]))


def test_real_heldout_data_qualification(path, verified_data):
    # No deliberate failure gate: actual hashes, labels and event split checked.
    assert verified_data.is_dir()


def test_training_schedule_contract(path):
    cfg = golden_meta()["training"]
    g = golden(Path(cfg["schedule"]).stem)
    for name in ("batch_indices", "shifts", "polarities", "rotation_angles"):
        assert g[name].shape == (cfg["steps"], cfg["batch_size"]) == (36000, 32)
    assert g["batch_indices"].dtype == g["shifts"].dtype == np.int32
    assert g["polarities"].dtype == g["learning_rates"].dtype == np.float32
    assert np.all((g["batch_indices"] >= 0) & (g["batch_indices"] < 30000))
    assert np.all((g["shifts"] >= -500) & (g["shifts"] <= 500))
    assert np.isin(g["polarities"], [-1, 1]).all()
    assert g["rotation_angles"].dtype == np.float32
    assert np.isfinite(g["rotation_angles"]).all()
    assert np.all(np.abs(g["rotation_angles"]) <= np.pi)
    assert cfg["seed"] == 43043 and cfg["label_sigma_samples"] == 10.0
    assert cfg["reference_widths"] == [16, 32, 64, 128, 192]
    assert cfg["optimizer"] == {
        "name": "Adam", "betas": [.9, .999], "eps": 1e-8,
        "bias_correction": True, "epsilon_outside_sqrt": True,
        "weight_decay": 0., "amsgrad": False, "gradient_clipping": False,
        "gradient_accumulation": False, "initial_moments": "zero", "first_update_index": 1,
    }
    rates = g["learning_rates"]
    assert rates.shape == (36000,) and np.isfinite(rates).all()
    warmup = cfg["warmup_steps"]
    assert warmup == 1000
    assert (np.diff(rates[:warmup]) > 0).all() and (np.diff(rates[warmup:]) <= 0).all()
    assert rates[0] == np.float32(cfg["learning_rate"] / warmup)
    assert rates[warmup-1] == rates[warmup] == np.float32(cfg["learning_rate"])
    assert rates[-1] == np.float32(cfg["final_learning_rate"])


def test_judge_selfcheck_round4_gpu_calibration(path):
    # Deliberately fails while BLOCKED: collection is not permission to lock.
    # No skip/xfail, and historical CPU runs cannot satisfy this prerequisite.
    meta = golden_meta()
    assert meta["lock_ready"], f"Lock blocked: {meta['status']}; {meta['blocker']}"
    f = meta["feasibility"]
    assert f["reference_completed"] and f["reference_passes_active_gates"] and f["gates_calibrated"]
    assert f["seed_order"] == [43043, 43044, 43045]
    f1 = []
    for seed in f["seed_order"]:
        report = meta["reference_training"][str(seed)]
        assert report["training"]["steps"] == 36000
        assert "gpu" in report["training"]["device"].lower()
        assert 0 < report["training"]["training_seconds"] <= REFERENCE_MAX_TRAINING_S
        assert report["completion"]["checkpoint_removed"]
        g = golden(f"round4_reference_audit_{seed}")
        actual = score_decoded(g["picks"][0], golden("stead_test_labels")["labels"], .01)
        np.testing.assert_array_equal(actual, g["expected_scores"][0])
        tp, fp, fn = actual[:, :3].sum(axis=0)
        f1.append([*actual[:, 3], 2*tp / (2*tp+fp+fn)])
        assert actual[0, 4] <= P_MEDIAN_MAX_S and actual[1, 4] <= S_MEDIAN_MAX_S
    worst = np.min(f1, axis=0)
    assert meta["feasibility"]["campaign_F1_min"] == F1_MIN
    expected = np.where(worst >= F1_MIN, F1_MIN, worst - REFERENCE_F1_MARGIN)
    np.testing.assert_array_equal([P_F1_MIN, S_F1_MIN, MICRO_F1_MIN], expected)


@pytest.mark.parametrize("case", ["perfect", "boundary", "outside", "duplicates", "missed", "empty", "swapped"])
def test_scoring_hand_counts(path, case):
    g = golden(f"score_{case}")
    actual = score(g["probabilities"], g["labels"], .01)
    np.testing.assert_array_equal(actual[:, :3], g["expected"][:, :3])
    np.testing.assert_allclose(actual[:, 3:], g["expected"][:, 3:], rtol=PARITY_RTOL, atol=PARITY_ATOL)


def test_decoder_boundaries_plateaus_and_threshold(path):
    g = golden("decoder")
    np.testing.assert_array_equal(decode(g["probabilities"][0, :, 1]), g["expected"])


def test_training_run_contract(path, trained):
    assert np.isfinite(trained["loss"])


def test_probability_contract(path, inference):
    assert inference.shape == (3000, 3000, 3)


@pytest.mark.parametrize("phase,limit", [(0, P_MEDIAN_MAX_S), (1, S_MEDIAN_MAX_S)], ids=["P", "S"])
def test_heldout_median_pick_error(path, scores, phase, limit):
    assert scores[phase, 4] <= limit, scores


@pytest.mark.parametrize("phase", [0, 1, 2], ids=["P", "S", "micro"])
def test_heldout_f1_at_point_one_second(path, scores, phase):
    if phase < 2:
        f1 = scores[phase, 3]
    else:
        tp, fp, fn = scores[:, :3].sum(axis=0)
        f1 = 2 * tp / (2 * tp + fp + fn)
    assert f1 >= (P_F1_MIN, S_F1_MIN, MICRO_F1_MIN)[phase], scores


@pytest.mark.parametrize("seed", [43043, 43044, 43045])
def test_judge_selfcheck_round3_reference_scores(path, seed):
    g = golden(f"reference_audit_{seed}")
    labels = golden("stead_test_labels")["labels"]
    actual = score_decoded(g["picks"][0], labels, .01)
    np.testing.assert_array_equal(actual, g["expected_scores"][0])
    assert actual[0, 4] <= P_MEDIAN_MAX_S and actual[1, 4] <= S_MEDIAN_MAX_S
    assert actual[0, 3] >= RETIRED_ROUND3_F1_GATES[0] and actual[1, 3] >= RETIRED_ROUND3_F1_GATES[1]
    tp, fp, fn = actual[:, :3].sum(axis=0)
    assert 2*tp/(2*tp+fp+fn) >= RETIRED_ROUND3_F1_GATES[2]


@pytest.mark.parametrize("seed", [43043, 43044, 43045])
@pytest.mark.parametrize("variant", [1, 2, 3, 4], ids=["untrained", "swapped", "half-time-scale", "bias-0.15s"])
def test_judge_selfcheck_round3_rejects_broken_models(path, seed, variant):
    g = golden(f"reference_audit_{seed}")
    actual = score_decoded(g["picks"][variant], golden("stead_test_labels")["labels"], .01)
    np.testing.assert_array_equal(actual, g["expected_scores"][variant])
    tp, fp, fn = actual[:, :3].sum(axis=0)
    f1 = np.r_[actual[:, 3], 2*tp/(2*tp+fp+fn)]
    assert np.all(f1 <= np.array(RETIRED_ROUND3_F1_GATES)-BROKEN_F1_GAP), f1


def test_deterministic_inference(path, inference, evaluation):
    # Same batch shape as the first tested call; repeat inference, never train.
    p = predict(path, evaluation["waveforms"][:32], float(evaluation["sample_interval_s"]))
    np.testing.assert_array_equal(p, inference[:32])


def test_batch_independence(path, inference, evaluation):
    # Paired events and a real record with no in-window S label.
    absent_s = int(np.flatnonzero(evaluation["labels"][:, 1] == -1)[0])
    for index in (0, 1, absent_s):
        p = predict(path, evaluation["waveforms"][index:index+1], float(evaluation["sample_interval_s"]))
        np.testing.assert_allclose(p, inference[index:index+1], rtol=PARITY_RTOL, atol=PARITY_ATOL)


def test_trace_gain_normalization(path, trained, evaluation):
    g = golden("normalization")
    x, dt = evaluation["waveforms"][:3], float(evaluation["sample_interval_s"])
    expected = predict(path, x, dt)
    actual = predict(path, x * g["gains"], dt)
    np.testing.assert_allclose(actual, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL)


def test_constant_trace_normalization_guard(path, trained):
    p = predict(path, golden("normalization")["constant_waveforms"], .01)
    np.testing.assert_allclose(p[0], p[1], rtol=PARITY_RTOL, atol=PARITY_ATOL)


def loss_call(path, logits, targets):
    out = call(path, "phase_loss_and_grad", logits=logits, targets=targets)
    assert out["loss"].shape == () and out["loss"].dtype == np.float32
    assert out["gradient"].shape == logits.shape and out["gradient"].dtype == np.float32
    assert np.isfinite(out["loss"]).all() and np.isfinite(out["gradient"]).all()
    return out


def test_loss_and_gradient_float64_parity(path):
    g = golden("loss")
    out = loss_call(path, g["logits"], g["targets"])
    np.testing.assert_allclose(out["loss"], g["loss"], rtol=PARITY_RTOL, atol=PARITY_ATOL)
    np.testing.assert_allclose(out["gradient"], g["gradient"], rtol=PARITY_RTOL, atol=PARITY_ATOL)


def test_loss_stability_extreme_logits(path):
    g = golden("loss")
    out = loss_call(path, g["extreme_logits"], g["extreme_targets"])
    np.testing.assert_allclose(out["loss"], g["extreme_loss"], rtol=PARITY_RTOL, atol=PARITY_ATOL)


def test_loss_invariant_to_per_sample_logit_shift(path):
    g = golden("loss")
    out = loss_call(path, g["logits"] + g["offsets"], g["targets"])
    np.testing.assert_allclose(out["loss"], g["loss"], rtol=PARITY_RTOL, atol=PARITY_ATOL)
    np.testing.assert_allclose(out["gradient"], g["gradient"], rtol=PARITY_RTOL, atol=PARITY_ATOL)


def test_centered_finite_difference_gradient(path):
    g = golden("loss")
    out = loss_call(path, g["logits"], g["targets"])
    observed, projected = [], []
    for plus_logits, minus_logits, effective_direction in zip(g["plus_logits"], g["minus_logits"], g["directions"]):
        # Saved float32 endpoints; h=.02. Use their effective secant direction
        # rather than nominal direction, accounting for endpoint quantization.
        plus = loss_call(path, plus_logits, g["targets"])
        minus = loss_call(path, minus_logits, g["targets"])
        observed.append((float(plus["loss"]) - float(minus["loss"])) / (2 * FD_STEP))
        projected.append(np.sum(out["gradient"].astype(np.float64) * effective_direction))
    observed, projected = np.array(observed), np.array(projected)
    assert np.linalg.norm(projected-observed) / np.linalg.norm(observed) <= GRAD_RTOL
    assert np.linalg.norm(projected-g["finite_difference"]) / np.linalg.norm(g["finite_difference"]) <= GRAD_RTOL
