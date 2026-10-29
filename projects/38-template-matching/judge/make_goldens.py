"""Generate deterministic float64 ObsPy reference outputs."""
import json
from pathlib import Path

import numpy as np
import obspy
import scipy
from obspy.signal.cross_correlation import correlate_template

OUT = Path(__file__).resolve().parent / "golden"


def save(name, data, templates, threshold):
    data = np.asarray(data, dtype=np.float32)
    templates = np.asarray(templates, dtype=np.float32)
    nt, ns, m = templates.shape
    corr = np.empty((nt, ns, data.shape[1] - m + 1), dtype=np.float64)
    for t in range(nt):
        for s in range(ns):
            corr[t, s] = correlate_template(
                data[s].astype(np.float64), templates[t, s].astype(np.float64),
                mode="valid", normalize="full", demean=True, method="direct")
    detections = (corr >= threshold).astype(np.float32)
    np.savez_compressed(OUT / (name + ".npz"), data=data, templates=templates,
                        threshold=np.array(threshold), correlation=corr,
                        detections=detections)
    return corr, detections


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3801)
    data = rng.normal(size=(3, 128)).astype(np.float32)
    templates = rng.normal(size=(4, 3, 17)).astype(np.float32)
    for s in range(3):
        data[s, 23:40] = 2 * templates[0, s] + 3
        data[s, 83:100] = -templates[1, s]
    c1, d1 = save("multi", data, templates, 0.8)
    assert np.all(c1[0, :, 23] > 0.99999)
    assert np.all(c1[1, :, 83] < -0.99999)
    c2, d2 = save("degenerate", [[1, 2, 3, 4], [7, 7, 7, 7]],
                  [[[4, 3, 2, 1], [1, 2, 3, 4]],
                   [[9, 9, 9, 9], [2, 2, 2, 2]]], 0.0)
    assert c2[0, 0, 0] == -1 and np.all(c2.ravel()[1:] == 0)
    assert d2.sum() == 3
    c3, d3 = save("edges", [[2, -1, 3, 0, 5, 2, -1, 3]],
                  [[[2, -1, 3]]], 0.99)
    assert c3[0, 0, 0] > 0.99999 and c3[0, 0, -1] > 0.99999
    cases = {"multi": "3 stations, 4 templates, 128 samples, length 17, threshold 0.8; planted positive matches at 23 and negative at 83", "degenerate": "2 stations, 2 templates, 4 samples, length 4, threshold 0; constant and reversed traces", "edges": "1 station, 1 template, 8 samples, length 3, threshold 0.99; matches at first and last window"}
    meta = {"seed": 3801, "numpy": np.__version__, "scipy": scipy.__version__, "obspy": obspy.__version__, "reference": "ObsPy correlate_template valid, full normalization, demean=True, direct; float64 casts of saved float32 inputs", "cases": cases, "tolerances": {"correlation_absolute": {"value": 1e-4, "reason": "float32 rounding and bounded coefficients, including near zero"}, "detections": "exact"}, "ranges_min_max_detection_count": {name: [float(c.min()), float(c.max()), int(d.sum())] for name, c, d in [("multi", c1, d1), ("degenerate", c2, d2), ("edges", c3, d3)]}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["ranges_min_max_detection_count"], indent=2))


if __name__ == "__main__":
    main()
