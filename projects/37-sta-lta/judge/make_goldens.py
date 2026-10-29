"""Build deterministic ObsPy reference cases."""
import json
from pathlib import Path

import numpy as np
import obspy
from obspy.signal.trigger import classic_sta_lta, trigger_onset

root = Path(__file__).resolve().parent / "golden"
root.mkdir(exist_ok=True)
rng = np.random.default_rng(3701)
x = rng.normal(0, .25, 65536).astype(np.float32)
for start, length, scale in [(5000, 350, 4), (27000, 600, 3), (65000, 536, 5)]:
    x[start:start + length] += rng.normal(0, scale, length).astype(np.float32)
cases = {"long": (x, 40, 400),
         "constant": (np.full(1024, 2, np.float32), 8, 80),
         "impulse": (np.r_[np.ones(512), 20, np.ones(511)].astype(np.float32), 5, 50),
         "equal_windows": (rng.normal(size=256).astype(np.float32), 16, 16)}
meta = {"seed": 3701, "numpy": np.__version__, "obspy": obspy.__version__,
        "tolerances": {"rtol": [1e-4, "project-wide float32 accuracy"],
                       "atol": [1e-6, "roundoff near zero; startup is exact"],
                       "trigger_indices": [0, "identical sample indices required"]}, "cases": {}}
for name, (trace, nsta, nlta) in cases.items():
    cf = classic_sta_lta(trace.astype(np.float64), nsta, nlta)
    picks = np.asarray(trigger_onset(cf, 3., 1.5), dtype=np.int64).reshape(-1, 2)
    assert np.isfinite(cf).all() and np.all(cf[:nlta - 1] == 0)
    if name in ("constant", "equal_windows"):
        assert np.allclose(cf[nlta - 1:], 1)
    np.savez_compressed(root / f"{name}.npz", trace=trace, characteristic=cf,
                        triggers=picks, nsta=nsta, nlta=nlta, on=3., off=1.5)
    meta["cases"][name] = {"samples": len(trace), "nsta": nsta, "nlta": nlta,
                           "on": 3., "off": 1.5, "min": float(cf.min()),
                           "max": float(cf.max()), "triggers": picks.tolist()}
    print(name, cf.min(), cf.max(), picks.tolist())
cf = np.array([0, 2, 3, 2.5, 2, 1.5, 1, 3, 3, 1, 3, 2], np.float32)
picks = np.asarray(trigger_onset(cf.astype(np.float64), 3., 2.), dtype=np.int64).reshape(-1, 2)
np.savez_compressed(root / "trigger_edges.npz", characteristic=cf, triggers=picks, on=3., off=2.)
meta["cases"]["trigger_edges"] = {"characteristic": cf.tolist(), "on": 3., "off": 2., "triggers": picks.tolist()}
print("trigger_edges", picks.tolist())
(root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
