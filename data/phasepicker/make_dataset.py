#!/usr/bin/env python
"""Build the 43-phase-picker train/val/test arrays from STEAD (SeisBench mirror).

Deterministic: fixed SEED, events split disjointly, per-trace RNG keyed on the
STEAD metadata row index. Only the selected traces are fetched, via HTTP Range
requests against the uncompressed, contiguous SeisBench HDF5 buckets
(no full 91 GB download).

Usage (from anywhere; Python 3.12 venv with seisbench/h5py/fsspec/requests/pandas,
e.g. `uv venv -p 3.12 data/phasepicker/.venv && uv pip install --python data/phasepicker/.venv
seisbench==0.12.6 h5py fsspec requests pandas`):
    data/phasepicker/.venv/bin/python -I data/phasepicker/make_dataset.py

Cache (metadata.csv, bucket offsets, fetched raw traces) lives in
~/.cache/outcrop-phasepicker/stead and makes re-runs resumable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 43
BASE_URL = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/stead"
META_URL = f"{BASE_URL}/metadata.csv"
WAVE_URL = f"{BASE_URL}/waveforms.hdf5"

OUT_DIR = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "outcrop-phasepicker" / "stead"

TARGETS = {"test": 3000, "val": 3000, "train": 30000}  # assigned in this order
OVERSAMPLE = 1.06          # extra candidates to absorb QC rejections
MAX_TRACES_PER_EVENT = 2   # event diversity
MIN_MEDIAN_SNR_DB = 10.0   # median of the 3 per-component STEAD SNRs
ORIG_LEN = 6000            # STEAD: 60 s @ 100 Hz
WIN = 3000                 # 30 s @ 100 Hz
P_MIN = 100                # P at least 1 s into the window
S_MARGIN = 50              # S at least 0.5 s before window end when we force it in
CHANNELS = {"HH", "BH", "EH"}  # broadband / short-period velocity; drops SH, EL, HL, LH (resampled oddities)
TRACE_BYTES = 3 * ORIG_LEN * 4
N_WORKERS = 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- metadata
def load_metadata() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "metadata.csv"
    if not p.exists():
        import requests

        log("downloading", META_URL)
        with requests.get(META_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(p.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        p.with_suffix(".part").rename(p)
    m = pd.read_csv(p, low_memory=False)
    m["row"] = np.arange(len(m))
    return m


def select_candidates(m: pd.DataFrame) -> pd.DataFrame:
    q = m[
        (m.trace_category == "earthquake_local")
        & (m.trace_p_status == "manual")
        & (m.trace_s_status == "manual")
        & m.trace_p_arrival_sample.notna()
        & m.trace_s_arrival_sample.notna()
        & m.source_id.notna()
    ].copy()
    q["source_id"] = q.source_id.astype(str)
    snr = np.stack([np.fromstring(s.strip("[]"), sep=" ", count=3) for s in q.trace_snr_db.values])
    q["snr_z"], q["snr_n"], q["snr_e"] = snr[:, 0], snr[:, 1], snr[:, 2]
    q = q[np.nanmedian(snr, axis=1) >= MIN_MEDIAN_SNR_DB]
    # STEAD arrival samples are float-typed in the CSV; round to nearest sample.
    q["p_orig"] = np.rint(q.trace_p_arrival_sample.values).astype(np.int64)
    q["s_orig"] = np.rint(q.trace_s_arrival_sample.values).astype(np.int64)
    q = q[(q.s_orig > q.p_orig) & (q.p_orig >= P_MIN) & (q.p_orig <= ORIG_LEN - WIN) & (q.s_orig < ORIG_LEN)]
    assert q.trace_name.str.endswith(",:3,:6000").all()
    return q


def split_events(q: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    q = q.sort_values(["source_id", "trace_name"])
    groups = {sid: g for sid, g in q.groupby("source_id", sort=True)}
    events = np.array(sorted(groups))
    order = rng.permutation(len(events))
    out, it = {}, iter(order)
    for split, n in TARGETS.items():
        need = int(np.ceil(n * OVERSAMPLE))
        rows = []
        while sum(len(r) for r in rows) < need:
            g = groups[events[next(it)]]
            k = min(MAX_TRACES_PER_EVENT, len(g))
            pick = np.sort(rng.choice(len(g), size=k, replace=False))
            rows.append(g.iloc[pick])
        out[split] = pd.concat(rows)
    ev = {s: set(d.source_id) for s, d in out.items()}
    assert not (ev["train"] & ev["val"]) and not (ev["train"] & ev["test"]) and not (ev["val"] & ev["test"])
    return out


# ----------------------------------------------------------------------------- fetching
def bucket_offsets(buckets: list[str]) -> dict[str, list]:
    p = CACHE / "bucket_offsets.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    missing = [b for b in buckets if b not in cache]
    if missing:
        import fsspec
        import h5py

        log(f"reading HDF5 layout for {len(missing)} buckets")
        f = fsspec.filesystem("https").open(WAVE_URL, "rb", block_size=1 << 16)
        with h5py.File(f, "r") as h:
            fmt = {k: h["data_format"][k][()] for k in h["data_format"]}
            assert fmt["component_order"] == b"ZNE" and int(fmt["sampling_rate"]) == 100, fmt
            for b in missing:
                d = h["data"][b]
                assert d.chunks is None and d.compression is None, b
                assert d.dtype == np.dtype("<f4") and d.shape[1:] == (3, ORIG_LEN), (b, d.dtype, d.shape)
                cache[b] = [int(d.id.get_offset()), int(d.shape[0])]
        p.write_text(json.dumps(cache))
    return cache


def fetch_all(names: np.ndarray, offsets: dict) -> np.memmap:
    """Fetch [N,3,6000] float32 into a resumable memmap keyed by the sorted name list."""
    key = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    raw_path, done_path = CACHE / f"raw_{key}.f32", CACHE / f"raw_{key}.done.npy"
    n = len(names)
    raw = np.memmap(raw_path, dtype="<f4", mode="r+" if raw_path.exists() else "w+", shape=(n, 3, ORIG_LEN))
    done = np.load(done_path) if done_path.exists() else np.zeros(n, bool)
    todo = np.flatnonzero(~done)
    log(f"{n} traces, {len(todo)} to fetch")
    if len(todo) == 0:
        return raw

    import threading

    import requests

    local = threading.local()

    def get(i):
        b, rest = names[i].split("$")
        j = int(rest.split(",")[0])
        off, nb = offsets[b]
        assert 0 <= j < nb
        start = off + j * TRACE_BYTES
        if not hasattr(local, "s"):
            local.s = requests.Session()
        for attempt in range(6):
            try:
                r = local.s.get(WAVE_URL, headers={"Range": f"bytes={start}-{start + TRACE_BYTES - 1}"}, timeout=60)
                if r.status_code == 206 and len(r.content) == TRACE_BYTES:
                    raw[i] = np.frombuffer(r.content, dtype="<f4").reshape(3, ORIG_LEN)
                    return i
            except requests.RequestException:
                pass
            time.sleep(2**attempt)
        raise RuntimeError(f"failed to fetch {names[i]}")

    t0, k = time.time(), 0
    with ThreadPoolExecutor(N_WORKERS) as ex:
        for i in ex.map(get, todo):
            done[i] = True
            k += 1
            if k % 2000 == 0 or k == len(todo):
                raw.flush()
                np.save(done_path, done)
                log(f"  fetched {k}/{len(todo)} ({k / (time.time() - t0):.1f}/s)")
    return raw


def cross_check(names: np.ndarray, raw: np.memmap, n_check: int = 40):
    import fsspec
    import h5py

    rng = np.random.default_rng(SEED + 1)
    idx = rng.choice(len(names), size=min(n_check, len(names)), replace=False)
    f = fsspec.filesystem("https").open(WAVE_URL, "rb", block_size=1 << 17)
    with h5py.File(f, "r") as h:
        for i in idx:
            b, rest = names[i].split("$")
            ref = h["data"][b][int(rest.split(",")[0])]
            assert np.array_equal(ref, raw[i]), names[i]
    log(f"cross-check vs h5py OK on {len(idx)} traces")


# ----------------------------------------------------------------------------- windows
def make_split(df: pd.DataFrame, raw: np.memmap, pos: dict, target: int) -> dict:
    keep, W = [], []
    p_idx, s_idx, start_l, s_out = [], [], [], []
    for row in df.itertuples(index=False):
        if row.trace_channel not in CHANNELS:
            continue
        x = np.asarray(raw[pos[row.trace_name]])
        rng = np.random.default_rng([SEED, int(row.row)])
        p, s = int(row.p_orig), int(row.s_orig)
        sp = s - p
        hi = min(p, WIN - S_MARGIN - sp)
        if hi >= P_MIN:
            p_new = int(rng.integers(P_MIN, hi + 1))
        else:
            p_new = int(rng.integers(P_MIN, p + 1))
        st = p - p_new
        w = x[:, st : st + WIN]
        # QC: finite, no dead component, no duplicated components
        if not np.isfinite(w).all():
            continue
        if (np.abs(w).max(axis=1) == 0).any() or (w.std(axis=1) == 0).any():
            continue
        if np.array_equal(w[0], w[1]) or np.array_equal(w[0], w[2]) or np.array_equal(w[1], w[2]):
            continue
        s_new = s - st
        keep.append(row)
        W.append(w.astype(np.float32))
        p_idx.append(p_new)
        s_idx.append(s_new if s_new < WIN else -1)
        s_out.append(s_new >= WIN)
        start_l.append(st)
        if len(keep) == target:
            break
    assert len(keep) == target, f"only {len(keep)} traces passed QC (target {target})"
    k = pd.DataFrame(keep)

    def U(col):
        return np.asarray(k[col].astype(str).values, dtype=str)

    def F(col):
        return pd.to_numeric(k[col], errors="coerce").to_numpy(np.float32)

    return dict(
        waveforms=np.stack(W),
        p_idx=np.asarray(p_idx, np.int32),
        s_idx=np.asarray(s_idx, np.int32),
        s_outside_window=np.asarray(s_out, bool),
        window_start=np.asarray(start_l, np.int32),
        p_arrival_orig=k.p_orig.to_numpy(np.int32),
        s_arrival_orig=k.s_orig.to_numpy(np.int32),
        snr_db=k[["snr_z", "snr_n", "snr_e"]].to_numpy(np.float32),
        distance_km=F("source_distance_km"),
        magnitude=F("source_magnitude"),
        magnitude_type=U("source_magnitude_type"),
        depth_km=F("source_depth_km"),
        back_azimuth_deg=F("path_back_azimuth_deg"),
        p_weight=F("trace_p_weight"),
        s_weight=F("trace_s_weight"),
        source_id=U("source_id"),
        source_origin_time=U("source_origin_time"),
        network=U("station_network_code"),
        station=U("station_code"),
        channel=U("trace_channel"),
        trace_name=U("trace_name"),
        trace_name_original=U("trace_name_original"),
        stead_split=U("split"),
        sampling_rate=np.float32(100.0),
        component_order=np.asarray("ZNE"),
    )


def sanity(name: str, d: dict) -> dict:
    W, p, s = d["waveforms"], d["p_idx"], d["s_idx"]
    assert W.dtype == np.float32 and W.shape[1:] == (3, WIN)
    assert ((p >= P_MIN) & (p < WIN)).all()
    assert ((s == -1) | ((s > p) & (s < WIN))).all()
    z = W[:, 0, :].astype(np.float64)
    z = z - z.mean(axis=1, keepdims=True)
    ratios = []
    for i in range(len(p)):
        pre = z[i, max(0, p[i] - 100) : p[i]]
        post = z[i, p[i] : p[i] + 100]
        ratios.append(np.sqrt((post**2).mean() / max((pre**2).mean(), 1e-20)))
    ratios = np.asarray(ratios)
    st = dict(
        n=int(len(p)),
        n_events=int(len(set(d["source_id"]))),
        frac_s_present=float((s >= 0).mean()),
        p_idx_pct=np.percentile(p, [0, 25, 50, 75, 100]).tolist(),
        s_idx_pct=np.percentile(s[s >= 0], [0, 25, 50, 75, 100]).tolist(),
        z_rms_ratio_post_pre_P_median=float(np.median(ratios)),
        frac_ratio_gt_1_5=float((ratios > 1.5).mean()),
        mag_pct=np.nanpercentile(d["magnitude"], [5, 50, 95]).tolist(),
        dist_km_pct=np.nanpercentile(d["distance_km"], [5, 50, 95]).tolist(),
        channels={c: int(n) for c, n in zip(*np.unique(d["channel"], return_counts=True))},
    )
    assert st["z_rms_ratio_post_pre_P_median"] > 2.0, (name, st)
    return st


def main():
    m = load_metadata()
    q = select_candidates(m)
    log(f"candidates after QC filter: {len(q)} traces, {q.source_id.nunique()} events")
    splits = split_events(q)
    allc = pd.concat(splits.values())
    names = np.array(sorted(set(allc.trace_name)))
    pos = {n: i for i, n in enumerate(names)}
    offsets = bucket_offsets(sorted({n.split("$")[0] for n in names}))
    raw = fetch_all(names, offsets)
    cross_check(names, raw)

    stats = {}
    for split in ["train", "val", "test"]:
        d = make_split(splits[split], raw, pos, TARGETS[split])
        stats[split] = sanity(split, d)
        out = OUT_DIR / f"{split}.npz"
        np.savez(out.with_suffix(".tmp.npz"), **d)
        os.replace(out.with_suffix(".tmp.npz"), out)
        log(split, json.dumps(stats[split]))
        del d
    ev = {s: set(np.load(OUT_DIR / f"{s}.npz")["source_id"]) for s in ["train", "val", "test"]}
    assert not (ev["train"] & ev["val"] or ev["train"] & ev["test"] or ev["val"] & ev["test"])
    stats["events_disjoint"] = True
    for s in ["train", "val", "test"]:
        h = hashlib.sha256()
        with open(OUT_DIR / f"{s}.npz", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        stats[s]["sha256"] = h.hexdigest()
        stats[s]["bytes"] = (OUT_DIR / f"{s}.npz").stat().st_size
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    log("done")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
