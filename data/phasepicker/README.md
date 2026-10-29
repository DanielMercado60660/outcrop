# phasepicker: labeled 3-component waveforms for 43-phase-picker

Real earthquake recordings with analyst (manual) P and S picks, cut into 30 s windows at 100 Hz.
`train.npz` is for training. `val.npz` is for model selection. `test.npz` is the held-out judge oracle.
No synthetic data is used.

## Source

- **Dataset:** STEAD, the STanford EArthquake Dataset (Mousavi, Sheng, Zhu & Beroza, 2019).
  - Original: https://github.com/smousavi05/STEAD
  - Paper: doi:10.1109/ACCESS.2019.2947848
- **Copy used:** the SeisBench mirror of STEAD, read with HTTP Range requests (only the selected traces were transferred, about 2.75 GB out of the 91 GB file).
  - `https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/stead/metadata.csv` (402,560,190 bytes)
  - `https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/stead/waveforms.hdf5` (91,127,786,704 bytes)
  - Read via `seisbench==0.12.6`.
  - HDF5 `data_format`: ZNE, 100 Hz, velocity, counts, response not removed.
  - Accessed 2026-10-28.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
  - Verified 2026-10-28: `https://raw.githubusercontent.com/smousavi05/STEAD/master/LICENSE` is the "Attribution 4.0 International" legal text, Crossref lists the paper's license as CC BY 4.0, and SeisBench's `STEAD` class declares `license = "CC BY 4.0"`.
  - CC BY 4.0 allows redistribution and adaptation, including derived arrays like these, provided attribution is given.
- **Required attribution:**

  > Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019). STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI. IEEE Access, 7, 179464–179476. doi:10.1109/ACCESS.2019.2947848. Licensed under CC BY 4.0. Derived subset (windowed/cropped, re-split) produced for Outcrop project 43.

  If SeisBench is credited as the access path: Woollam et al. (2022), "SeisBench — A Toolbox for Machine Learning in Seismology", Seismological Research Letters, doi:10.1785/0220210324.

## Files

| file | traces | events | bytes | sha256 |
|---|---|---|---|---|
| train.npz | 30,000 | 19,799 | 1,095,396,380 | `b3761f94e11c49e1921300da9ee8344871eed21954a0275bc4a2e7f2e228e00f` |
| val.npz | 3,000 | 1,993 | 109,533,380 | `fc1026565870fbc00de4503f572193017ee7f6656488a392fae454c676a1e29e` |
| test.npz | 3,000 | 1,978 | 109,533,380 | `72938647e303dde3b95548a2d30972fb235ca9d019ef7429e4dbee23fc9dd7e6` |

- **Total size:** about 1.3 GB.
- **Event overlap:** no `source_id` appears in more than one split. The build script asserts this, and the assertion is recorded in `stats.json` as `events_disjoint: true`.
- **Other files:**
  - `stats.json`: per-split summary statistics.
  - `make_dataset.py`: the exact build script.
  - `.venv/`: the Python 3.12 build environment (about 0.9 GB, including torch pulled in by seisbench). It is not needed to *load* the data and can be deleted.

Load the files with plain numpy: `np.load("test.npz")`. No pickle is needed, because all string arrays are fixed-width unicode.

### Arrays (per file, N traces)

| key | dtype / shape | meaning |
|---|---|---|
| `waveforms` | float32 [N, 3, 3000] | components in Z, N, E order, at 100 Hz over 30 s. Values are raw counts as distributed: not normalized, not filtered, not demeaned. |
| `p_idx`, `s_idx` | int32 [N] | P and S arrival sample index in the window. A value of -1 means the arrival is absent from the window. |
| `s_outside_window` | bool [N] | True when the trace has a manual S pick that falls after the window end, which is why `s_idx` is -1. Every trace in these files has a manual S label, so -1 always means "S after the end of the window", never "unknown". |
| `window_start` | int32 [N] | crop start, as a sample in the original 60 s STEAD trace |
| `p_arrival_orig`, `s_arrival_orig` | int32 [N] | STEAD arrival samples in the original trace, rounded to the nearest sample |
| `snr_db` | float32 [N, 3] | STEAD per-component SNR (dB) for Z, N, E |
| `distance_km`, `magnitude`, `magnitude_type`, `depth_km`, `back_azimuth_deg` | | source and path metadata from STEAD |
| `p_weight`, `s_weight` | float32 [N] | STEAD pick weights, recorded as-is. Their exact semantics were not verified, so they were not used for filtering. |
| `source_id`, `source_origin_time` | str [N] | event id (the split key) and origin time |
| `network`, `station`, `channel` | str [N] | station identity. The channel code is one of HH, BH, EH. |
| `trace_name` | str [N] | SeisBench bucket locator, e.g. `bucket12$345,:3,:6000` |
| `trace_name_original` | str [N] | original STEAD trace name |
| `stead_split` | str [N] | STEAD/SeisBench's own split label. It is recorded only and not used. |
| `sampling_rate` | float32 scalar | 100.0 |
| `component_order` | str scalar | "ZNE" |

## Preprocessing (what `make_dataset.py` does)

1. **Candidate pool.**
   - Rows from the 1,265,657-row STEAD metadata with:
     - `trace_category == earthquake_local`
     - `trace_p_status == manual` and `trace_s_status == manual`; autopicker and automatic picks are excluded.
     - Both arrival samples present, and a `source_id` present.
   - Median of the three per-component SNRs at least 10 dB.
   - Picks are rounded to integer samples, with `S > P`.
   - Result: 627,786 traces from 283,384 events.
2. **Event split.** All rows are inputs to the split, including traces later dropped by the channel QC in step 5.
   - Unique `source_id`s are sorted and then permuted with `numpy.random.default_rng(43)`.
   - Events are taken in that order and assigned to test, then val, then train.
   - From each event, up to 2 traces are chosen at random with the same RNG, which favors event diversity.
   - Each split is filled to 6% above its target, to absorb QC rejections.
3. **Fetch.**
   - Bucket byte offsets are read once from the HDF5 header with h5py over fsspec. The script asserts that every bucket is contiguous, uncompressed, and `<f4` with shape `[*, 3, 6000]`.
   - Each trace is then fetched as one 72,000-byte HTTP Range request.
   - 40 random traces are cross-checked byte for byte against h5py reads.
4. **Windowing.** STEAD places P only at original samples 400 to 1000.
   - A per-trace RNG `default_rng([43, metadata_row])` draws the P position `p_idx` uniformly from `[100, min(P_orig, 2950 − (S−P))]`, so that S lands at least 0.5 s before the window end.
   - When S−P is too long for that, `p_idx` is drawn from `[100, P_orig]` and S may fall outside the window (`s_idx = -1`).
   - The window is `orig[:, P_orig − p_idx : P_orig − p_idx + 3000]`. There is no padding and no resampling, since the data is already 100 Hz.
5. **QC on the window.** Traces are rejected if they:
   - have a channel code other than HH, BH, or EH (this drops rare SH, EL, HL, and LH traces);
   - contain non-finite samples;
   - have a dead component (all zero or zero variance);
   - have duplicated components.

   The first N survivors of each split are kept, in the deterministic candidate order, up to the exact target size.
6. **Sanity checks** (asserted):
   - `100 ≤ p_idx < 3000`
   - `s_idx == -1` or `p_idx < s_idx < 3000`
   - The median Z-component RMS ratio, 1 s after P over 1 s before P, is above 2. Observed values are about 7.7 to 8.1 per split, and about 96% of traces have a ratio above 1.5.
7. **Determinism.** The build was run again from the cache and the outputs are bit-identical (same sha256), verified 2026-10-28.

The three `.npz` files (1.3 GB in all) are too large for the repository. To build them, create the builder's environment with `uv venv -p 3.12 data/phasepicker/.venv && uv pip install --python data/phasepicker/.venv seisbench==0.12.6 h5py fsspec requests pandas`, then run `data/phasepicker/.venv/bin/python -I data/phasepicker/make_dataset.py`. The output is bit-identical to the files the tests pin by sha256. Without them, project 43's training and held-out tests are skipped. This needs network access the first time. The cache is kept in `~/.cache/outcrop-phasepicker/stead`: metadata.csv (403 MB) plus a 2.75 GB raw-trace memmap. Delete the cache to reclaim the space.

## Split statistics (test; train and val are similar, see stats.json)

| quantity | test value | train and val |
|---|---|---|
| S present | 99.1% (26 traces with S after the window end) | 99.3% |
| `p_idx` (samples) | median 340, IQR 221–503, range 100–983 | |
| `s_idx` (samples) | median 929, IQR 649–1293, range 146–2947 | |
| S−P | 5th/50th/95th percentiles 1.1 / 5.3 / 15.2 s | |
| Magnitude | 5th/50th/95th percentiles 0.3 / 1.3 / 3.5 (mostly ML and Md) | |
| Distance | 5th/50th/95th percentiles 4 / 39 / 132 km | |
| Channels | HH 1409, BH 764, EH 827 | |

## Caveats for the judge

- **Label quality.** The picks are human-analyst picks from the contributing networks, collected by STEAD. They are not re-reviewed, and their pick uncertainty is not given in seconds.
  - Inter-analyst S-pick scatter is typically reported as about 0.05 to 0.1 s, so the S threshold of median ≤ 0.1 s is reasonable but not loose.
  - A few percent of traces have weak P onsets: about 4% have a post/pre-P RMS ratio below 1.5.
- **P position bias.** STEAD's original P lies at 4 to 10 s, and windows cannot be zero-padded, so `p_idx` is always within the first 10 s (1 to 10 s, median about 3.4 s).
  - The test set therefore does not probe late-in-window P arrivals.
  - Training code should randomly shift and crop on train, rather than learn "P near 3 s".
- **No noise-only windows.** Every window contains a P. S is present in about 99% of windows.
  - F1 false positives therefore only come from extra or duplicate picks inside event windows. For F1, count `p_idx`/`s_idx == -1` as "no true pick".
- **Amplitudes.** The waveforms are raw counts with no instrument correction, and amplitude varies over orders of magnitude.
  - Models must normalize each trace themselves, for example by demeaning and dividing by the per-trace max or std.
  - Some traces have a small DC offset.
- **Sampling.** All windows are 100 Hz. STEAD resampled all traces to 100 Hz before release (per the STEAD paper).
- **Geography.** STEAD is globally sourced but heavily weighted to US networks and local events (median 39 km, M about 1.3). It is not representative of teleseismic picking.
