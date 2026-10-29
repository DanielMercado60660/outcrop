# Marmousi2 (AGL Elastic Marmousi) model, for project 16-fwi-marmousi

Local copy of the Marmousi2 elastic model (P velocity, S velocity and density)
plus two block-averaged P-velocity grids for FWI benchmarks. The build
environment had no network, so everything needed is in this directory. `make_marmousi2.py`
regenerates every `.npy`, `vp_meta.json` and `SHA256SUMS` deterministically
from `raw/` (checked: two runs give identical hashes; runtime about 2 s). It
downloads only if a `raw/` zip is missing.

The arrays are not committed; this README, the licence and the build script are. Unlike `data/marmousi/` (classic IFP
Marmousi, no licence), this model is **CC BY 4.0, so it may be redistributed
and shown in figures, with attribution**.

## Source

| Item | Value |
|---|---|
| Model | Marmousi2, also called AGL Elastic Marmousi. Martin, Wiley and Marfurt (2006), Allied Geophysical Laboratories (AGL), University of Houston |
| Official download page | https://www.agl.uh.edu/downloads/downloads.htm (section "Modeling Grids"). Snapshot saved as `raw/agl_downloads_page_2026-10-28.htm` |
| Files downloaded (2026-10-28, `curl`) | `https://www.agl.uh.edu/downloads/modeling_grids/MODEL_P-WAVE_VELOCITY_1.25m.segy.zip` (2,305,936 B, Last-Modified 05 Jul 2005)<br>`.../MODEL_S-WAVE_VELOCITY_1.25m.segy.zip` (2,126,006 B)<br>`.../MODEL_DENSITY_1.25m.segy.zip` (2,142,514 B). Saved byte-for-byte in `raw/` |
| Note on the AGL page | The "model_p-wave_velocity" link text partly points to the S-wave zip (an HTML typo on the page). The URLs above are the correct ones; the SEG-Y text header of each file confirms its property |
| Not used | The 2004 "Model Definition" files on the same page (`vp_marmousi-ii.segy.gz` etc.) have a different, non-standard SEG-Y layout (Segy2sep synthetic header). They are not the modeling grids and were not kept |
| Licence source (see below) | SEG Open Data entry "AGL Elastic Marmousi": https://wiki.seg.org/wiki/AGL_Elastic_Marmousi. The live page is behind a Cloudflare challenge, so it was read from the Wayback snapshot http://web.archive.org/web/20240915125424/https://wiki.seg.org/wiki/AGL_Elastic_Marmousi (saved as `raw/seg_wiki_AGL_Elastic_Marmousi_wayback_20240915125424.htm`) |
| SEG Open Data package | `https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz` (live URL returns HTTP 403 now). Fetched from Wayback: http://web.archive.org/web/20240328082847/http://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz (153,722,387 B, original Last-Modified 22 Apr 2019, sha256 `b6fe8d10ce6a6dfadf203a8d0e1a040ce9f2c6cb52c1366193d35113d61797fe`). Only its `LICENSE` and `README` were kept here, as `LICENSE` and `README_SEG_OPEN_DATA.txt` |
| Identity check | The three SEG-Y files inside the AGL zips are **byte-identical** (`cmp`) to the three in the SEG Open Data package (`model/MODEL_*_1.25m.segy.tar.gz`). So the CC BY 4.0 release covers exactly these bytes |

## Licence and attribution

- **Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)**,
  https://creativecommons.org/licenses/by/4.0/. Full text in `LICENSE`.
- **Where it is stated.** The AGL download page itself has no licence text.
  The licence comes from the 2019 SEG Open Data release, where the authors
  released the model "with the permission of the program director of the
  Allied Geophysical Labs consortium". That release's README (copied here
  verbatim as `README_SEG_OPEN_DATA.txt`) and the SEG wiki page say the model
  is licensed CC BY 4.0, permitting copying, redistribution and adaptation for
  any purpose, including commercially.
- **Copyright notice (keep it):** Copyright 2004 Allied Geophysical
  Laboratory, University of Houston, TX, USA.
- **Condition from the release README:** do not separate the model and data
  from the licence, and always include the licence statement and link with the
  data. If any array from this folder is redistributed, ship `LICENSE` and
  `README_SEG_OPEN_DATA.txt` with it. The downsampled grids are adaptations,
  so say they were modified (block-averaged), as CC BY 4.0 section 3(a)(1)(B)
  requires.

**Attribution for figure captions** (short form):

> Velocity model: Marmousi2 (AGL Elastic Marmousi), © 2004 Allied Geophysical Laboratory, University of Houston; Martin, Wiley & Marfurt (2006); CC BY 4.0. Block-averaged to 50 m / 25 m.

**Attribution for READMEs** (full form):

> This work uses the Marmousi2 (AGL Elastic Marmousi) model, created by Gary S. Martin, Robert Wiley and Kurt J. Marfurt at the Allied Geophysical Laboratories, University of Houston. Copyright 2004 Allied Geophysical Laboratory, University of Houston, TX, USA. Licensed under the Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/). Obtained from https://www.agl.uh.edu/downloads/downloads.htm (licence per the SEG Open Data release, https://wiki.seg.org/wiki/AGL_Elastic_Marmousi). Changes: converted from SEG-Y to NumPy arrays and block-averaged to coarser grids.
>
> Martin, G. S., Wiley, R., and Marfurt, K. J., 2006, Marmousi2: An elastic upgrade for Marmousi: The Leading Edge, 25(2), 156–166, doi:10.1190/1.2172306.

## Original model

| Property | Value |
|---|---|
| Format | SEG-Y rev 0, IBM float (format code 1), 3600-byte header, 240-byte trace headers, 155,653,444 B per file |
| Traces | 13,601 traces = x from 0 to 17,000 m every 1.25 m (trace header bytes 73–76 hold x in mm) |
| Samples | 2,801 per trace = z from 0 to 3,500 m every 1.25 m (header dt = 1250 is used as depth step) |
| Grid spacing | **dz = dx = 1.25 m**; nodes include both ends (0 and 3500 m, 0 and 17000 m) |
| Water layer | **Yes.** Samples 0–360 (z = 0–450 m) are water at every x: Vp = 1500 m/s, Vs = 0, rho = 1.01 g/cm³. The sea floor is flat at 450 m |
| Vp range | 1028–4700 m/s. Values below 1500 m/s are gas sands (19,743 nodes, 0.05 %), lowest 1028 m/s |
| Vs range | 0 (water) to 2802 m/s |
| Density | 1.01–2.627 g/cm³ in the SEG-Y |

## Files produced

| File | Shape [nz, nx] | Spacing | Content |
|---|---|---|---|
| `vp.npy` | 2801 × 13601 | 1.25 m | P velocity, m/s, float32. Value (i, j) at z = 1.25 i, x = 1.25 j |
| `vs.npy` | 2801 × 13601 | 1.25 m | S velocity, m/s, float32 |
| `rho.npy` | 2801 × 13601 | 1.25 m | Density, **kg/m³** (SEG-Y g/cm³ × 1000, computed in float64), float32 |
| `vp_70x340_d50m.npy` | 70 × 340 | 50 m | Block mean of vp, factor 40 |
| `vp_140x680_d25m.npy` | 140 × 680 | 25 m | Block mean of vp, factor 20 |
| `vp_meta.json` | | | Shapes, spacing, units, method, min/max, resolution numbers (same keys as `data/marmousi/vp_meta.json`) |
| `LICENSE`, `README_SEG_OPEN_DATA.txt` | | | CC BY 4.0 text and the SEG release statement, verbatim |

### Why 70 × 340 and 140 × 680, not 94 × 340 and 188 × 680

The full model is 3.5 km × 17 km (aspect 0.206), but 94/340 = 0.276. With
square cells (needed for the FD stencil) and the whole model kept, nz must be
0.206 nx. 2800 and 13600 share the integer factors 20 and 40, which give
exactly nx = 680 and 340 as asked, with nz = 140 and 70. No cropping or
padding is needed, and the 50 m grid is exactly the 2 × 2 block mean of the
25 m grid.

### Block averaging (exact)

For factor f (40 → 50 m, 20 → 25 m):

1. Drop the last node row (z = 3500 m) and the last node column (x = 17000 m),
   leaving 2800 × 13600 nodes.
2. Reshape to (2800/f, f, 13600/f, f), take the arithmetic mean of **velocity**
   (not slowness) over axes 1 and 3 in float64, and cast to float32.
3. Cell (i, j) is the mean over nodes z ∈ [i d, (i+1) d), x ∈ [j d, (j+1) d),
   d = 1.25 f. The node centroid is at i d + (d − 1.25)/2, so 24.375 m
   (not 25 m) into a 50 m cell. The origin is (0, 0).

Water/sediment boundary: at 50 m, rows 0–8 are pure water and row 9 mixes one
water node with 39 sediment nodes. At 25 m, rows 0–17 are pure water and
row 18 mixes 1 + 19 nodes.

### Resolution check at 10 Hz (8th-order stencil)

Points per wavelength = v_min / (f · dx). A common rule for 8th-order FD is about 4 or more.

| Grid | Min Vp after averaging | ppw at 10 Hz, min Vp | ppw at 10 Hz, water (1500) | f_max for 4 ppw at min Vp |
|---|---|---|---|---|
| 50 m (70 × 340) | 1133.6 m/s (13 cells < 1500) | 2.27 | 3.0 | 5.7 Hz |
| 25 m (140 × 680) | 1109.1 m/s (61 cells < 1500) | 4.44 | 6.0 | 11.1 Hz |

The **25 m grid resolves 10 Hz**. The **50 m grid does not**: it is only good
to about 7.5 Hz in water and 5.7 Hz in the gas sands. Use it for low-frequency
or timing benchmarks, or clip Vp to ≥ 1500 m/s and keep the source below
~7 Hz.

## Processing steps

1. Downloaded the three AGL zips (table above) and the SEG Open Data tarball
   (Wayback). Checked that the SEG-Y bytes are identical in both.
2. `make_marmousi2.py`: checked each zip's sha256. Read the SEG-Y from the
   zip in memory. Checked the binary header (dt = 1250, ns = 2801, format 1),
   the file size (3600 + 13601 × (240 + 4 × 2801)) and the trace x positions
   (0 … 17,000,000 mm). Decoded IBM floats with a vectorised numpy decoder
   (checked bit-for-bit against `obspy.io.segy` on traces 0, 3000, 6800 and
   13600: max difference 0.0). Transposed to [nz, nx].
3. Saved `vp.npy`, `vs.npy` and `rho.npy` (density × 1000 → kg/m³). Checked
   that the water layer is samples 0–360.
4. Block-averaged vp (factors 40 and 20, above). Wrote `vp_meta.json` and
   `SHA256SUMS`.

Rebuild: `.venv/bin/python -I data/marmousi2/make_marmousi2.py`

## sha256

Also in `SHA256SUMS` (all files except this README and `SHA256SUMS` itself).

```
1aaf0648e34e568b31c2f1511958c1915a164511989c693fc133e82a10e557cb  LICENSE
02d9f7bde15bf070a4826b7080130669e8c3c1ac894c97c6be5a1e2223c4d624  README_SEG_OPEN_DATA.txt
dbf76426b9b506251139dba93c40a235edeee2d805f5562d95be6f2b4726c5d4  make_marmousi2.py
d4f4459b28457bdef23bc1eb1bcf409756ac479205b6bb9a9467c3718e9fc34e  raw/MODEL_DENSITY_1.25m.segy.zip
366785977e33e3c9460b855ceba5b71b5ca32ecd46c2ef229b1ff43b5bfc23fc  raw/MODEL_P-WAVE_VELOCITY_1.25m.segy.zip
c2a4498e3442a1c401e82531f4f756a50932068e97baa78a3fb3f08fec10a913  raw/MODEL_S-WAVE_VELOCITY_1.25m.segy.zip
8b05de202b680e739d1029610eac8e41e17bbe6c3f6d5d647c67a18a2ccda6c6  raw/agl_downloads_page_2026-10-28.htm
4f213b1c281d22cc963d232a30661c7faa2cdea6ee5b3536b99387ae4414c57a  raw/seg_wiki_AGL_Elastic_Marmousi_wayback_20240915125424.htm
2ee194a582a6be4cd0ff264ef4897ac7bf1abeed67f2fe37d651a986b3dddf67  rho.npy
ffbc657036dd3307d3cd6d2c116422559b9cb9ec46d4b3efb1b09bab811c9e31  vp.npy
f91045a1032141987cbe0b3a8092e4e6ada62d10710aeb5d45b822194ff2b7a0  vp_140x680_d25m.npy
98699e31e3c0745aa0bc941ae2160585737e907d8b3f2ab763841ecf7f54007d  vp_70x340_d50m.npy
2371e42ee5aa57d663e94d42aff1a1ad84b29c9554c5ca27b214cf2579ff4b37  vp_meta.json
9fdff66f2e27704c1a59ce7ee7b8edc7c5c4717fc1690b73b67a5c194064c848  vs.npy
```

sha256 of the extracted SEG-Y files: P `64a6bc60c5074e64a5ecf16e471077aaf47c3ad6c8be50aeb230800235dc82d3`,
S `a7846ec4fe0c441670dc602fd1acd5090ca29219618822244702272e1c533637`,
density `5677ba1f2199d06440164da557966f2c25e2d1a572f3cfb8c37e2a40c16ba29e`.
