"""Independent evaluator, never delegated to an implementation."""
import numpy as np

from .tolerances import MATCH_TOLERANCE_S, PICK_THRESHOLD


def decode(channel):
    """One earliest argmax per contiguous inclusive-threshold region."""
    active = channel >= PICK_THRESHOLD
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return np.array([start + np.argmax(channel[start:stop])
                     for start, stop in zip(starts, stops)], dtype=np.int64)


def score(probabilities, labels, dt):
    """Return per-phase TP,FP,FN,F1,matched median; only one truth/phase/trace.

    Nearest within tolerance matches the truth, every additional prediction is
    false. There is no cross-trace/cross-phase matching. An absent label is -1.
    Noise-only records contribute false positives. No matches => median inf.
    """
    picks = [[decode(p[:, phase+1]) for phase in range(2)] for p in probabilities]
    return score_decoded(picks, labels, dt)


def score_decoded(picks, labels, dt):
    """Same evaluator on decoded rows, with optional -1 padding in goldens."""
    result = []
    for phase in range(2):
        tp = fp = fn = 0
        errors = []
        for b in range(len(labels)):
            predicted = np.asarray(picks[b][phase], dtype=np.int64)
            predicted = predicted[predicted >= 0]
            truth = int(labels[b, phase])
            distances = np.abs(predicted - truth)
            match = truth >= 0 and len(predicted) > 0 and distances.min() <= MATCH_TOLERANCE_S / dt
            tp += int(match)
            fp += len(predicted) - int(match)
            fn += int(truth >= 0) - int(match)
            if match:
                errors.append(float(distances.min()) * dt)
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 1.0
        result.append([tp, fp, fn, f1, np.median(errors) if errors else np.inf])
    return np.array(result, dtype=np.float64)
