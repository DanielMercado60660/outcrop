#!/usr/bin/env python3
"""Build the Marmousi2 (AGL Elastic Marmousi) arrays used by project 16-fwi-marmousi.

Inputs  (raw/, downloaded once from the official AGL page; re-downloaded only if missing):
    raw/MODEL_P-WAVE_VELOCITY_1.25m.segy.zip
    raw/MODEL_S-WAVE_VELOCITY_1.25m.segy.zip
    raw/MODEL_DENSITY_1.25m.segy.zip

Outputs (next to this script):
    vp.npy                 float32 [2801, 13601]  P velocity, m/s, 1.25 m grid, depth axis first
    vs.npy                 float32 [2801, 13601]  S velocity, m/s (0 in the water layer)
    rho.npy                float32 [2801, 13601]  density, kg/m^3 (SEG-Y stores g/cm^3; multiplied by 1000)
    vp_70x340_d50m.npy     float32 [70, 340]      block mean of vp, 40 x 40 samples per cell
    vp_140x680_d25m.npy    float32 [140, 680]     block mean of vp, 20 x 20 samples per cell
    vp_meta.json           grid / provenance / resolution metadata
    SHA256SUMS             sha256 of every file in this folder except README.md and SHA256SUMS

Deterministic: same inputs give byte-identical outputs. Needs only numpy (obspy is not required:
the SEG-Y files are plain IBM-float, fixed-length, so a small vectorised decoder is used; it was
checked bit-for-bit against obspy on traces 0, 3000, 6800 and 13600 of the P-velocity file).

Run:  .venv/bin/python -I data/marmousi2/make_marmousi2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

AGL_BASE = "https://www.agl.uh.edu/downloads/modeling_grids/"
SOURCES = {
    # key: (zip name, member name, sha256 of zip, sha256 of extracted SEG-Y)
    "vp": ("MODEL_P-WAVE_VELOCITY_1.25m.segy.zip", "MODEL_P-WAVE_VELOCITY_1.25m.segy",
           "366785977e33e3c9460b855ceba5b71b5ca32ecd46c2ef229b1ff43b5bfc23fc", None),
    "vs": ("MODEL_S-WAVE_VELOCITY_1.25m.segy.zip", "MODEL_S-WAVE_VELOCITY_1.25m.segy",
           "c2a4498e3442a1c401e82531f4f756a50932068e97baa78a3fb3f08fec10a913", None),
    "rho": ("MODEL_DENSITY_1.25m.segy.zip", "MODEL_DENSITY_1.25m.segy",
            "d4f4459b28457bdef23bc1eb1bcf409756ac479205b6bb9a9467c3718e9fc34e", None),
}

NZ, NX = 2801, 13601          # samples per trace (depth), number of traces (x)
D = 1.25                      # m, both axes
WATER_SAMPLES = 361           # samples 0..360 (z = 0 .. 450 m) are water (Vp 1500, Vs 0, rho 1.01)
FACTORS = (40, 20)            # block sizes -> 50 m and 25 m cells
F_CHECK_HZ = 10.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, want: str) -> Path:
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        url = AGL_BASE + name
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=600) as r, open(path, "wb") as f:
            f.write(r.read())
    got = sha256(path)
    if got != want:
        raise SystemExit(f"sha256 mismatch for {path}: {got} != {want}")
    return path


def ibm_to_float32(words_be_u32: np.ndarray) -> np.ndarray:
    """IBM System/360 single-precision float (big-endian uint32) -> float32."""
    u = words_be_u32.astype(np.uint32)
    sign = np.where(u >> 31, -1.0, 1.0)
    expo = ((u >> 24) & 0x7F).astype(np.int32) - 64
    mant = (u & 0x00FFFFFF).astype(np.float64) / float(1 << 24)
    return (sign * mant * np.power(16.0, expo)).astype(np.float32)


def read_model(zip_path: Path, member: str) -> tuple[np.ndarray, str]:
    with zipfile.ZipFile(zip_path) as z:
        buf = z.read(member)
    segy_sha = hashlib.sha256(buf).hexdigest()
    b = np.frombuffer(buf, dtype=np.uint8)
    bh = b[3200:3600].tobytes()
    dt = int.from_bytes(bh[16:18], "big", signed=True)     # bytes 3217-3218
    ns = int.from_bytes(bh[20:22], "big", signed=True)     # bytes 3221-3222
    fmt = int.from_bytes(bh[24:26], "big", signed=True)    # bytes 3225-3226
    if (dt, ns, fmt) != (1250, NZ, 1):
        raise SystemExit(f"{member}: unexpected binary header dt={dt} ns={ns} fmt={fmt}")
    tr_len = 240 + 4 * NZ
    if (b.size - 3600) != NX * tr_len:
        raise SystemExit(f"{member}: size {b.size} is not 3600 + {NX} x {tr_len}")
    tr = b[3600:].reshape(NX, tr_len)
    # trace header bytes 73-76 (sx) hold x in millimetres: 0, 1250, ..., 17_000_000
    sx = tr[:, 72:76].copy().view(">i4").ravel()
    if not np.array_equal(sx, np.arange(NX, dtype=np.int64) * 1250):
        raise SystemExit(f"{member}: traces are not x = 0..17000 m in 1.25 m steps")
    data = tr[:, 240:].copy().view(">u4").reshape(NX, NZ)
    return np.ascontiguousarray(ibm_to_float32(data).T), segy_sha   # [nz, nx]


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    nz, nx = (a.shape[0] - 1) // f, (a.shape[1] - 1) // f   # drop the last node (z=3500 m, x=17000 m)
    c = a[: nz * f, : nx * f].astype(np.float64)
    return c.reshape(nz, f, nx, f).mean(axis=(1, 3)).astype(np.float32)


def main() -> None:
    meta: dict = {
        "_source": {
            "model": "Marmousi2 / AGL Elastic Marmousi (Martin, Wiley & Marfurt 2006), 1.25 m modeling grids",
            "page": "https://www.agl.uh.edu/downloads/downloads.htm",
            "url_base": AGL_BASE,
            "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), see LICENSE and README_SEG_OPEN_DATA.txt",
            "attribution": ("Marmousi2 (AGL Elastic Marmousi) model, Copyright 2004 Allied Geophysical Laboratory, "
                            "University of Houston; G. S. Martin, R. Wiley and K. J. Marfurt (2006), The Leading Edge "
                            "25(2), 156-166, doi:10.1190/1.2172306; licensed CC BY 4.0."),
            "axis0": "depth z (down positive)",
            "axis1": "horizontal x",
            "origin_m": [0.0, 0.0],
            "water_layer": "yes: samples 0..360 (z = 0..450 m) are water, Vp 1500 m/s, Vs 0, rho 1010 kg/m^3, flat sea floor",
            "note": ("vp/vs/rho.npy are node samples: value (i,j) at z = i*1.25 m, x = j*1.25 m. For block-mean files "
                     "with factor f and cell size d = 1.25*f, cell (i,j) is the arithmetic mean of velocity over node "
                     "samples z in [i*d, (i+1)*d) x [j*d, (j+1)*d), i.e. f x f nodes; node centroid at "
                     "i*d + (d-1.25)/2. The last node row (z=3500 m) and column (x=17000 m) are dropped."),
        }
    }
    full = {}
    for key, (zname, member, zsha, _) in SOURCES.items():
        zpath = fetch(zname, zsha)
        arr, segy_sha = read_model(zpath, member)
        if key == "rho":
            arr = (arr.astype(np.float64) * 1000.0).astype(np.float32)   # g/cm^3 -> kg/m^3
        full[key] = arr
        out = HERE / f"{key}.npy"
        np.save(out, arr)
        meta[out.name] = {
            "shape": list(arr.shape), "dz_m": D, "dx_m": D, "dtype": "float32",
            "units": {"vp": "m/s", "vs": "m/s", "rho": "kg/m^3"}[key],
            "method": ("transpose of upstream SEG-Y (traces = x), IBM float -> float32, no resampling"
                       + ("; multiplied by 1000 (g/cm^3 -> kg/m^3)" if key == "rho" else "")),
            "raw_zip": f"raw/{zname}", "raw_zip_sha256": zsha,
            "segy_member": member, "segy_sha256": segy_sha,
            "min": float(arr.min()), "max": float(arr.max()),
        }

    vp = full["vp"]
    assert np.all(vp[:WATER_SAMPLES] == 1500.0) and not np.any(vp[WATER_SAMPLES] == 1500.0)
    meta["vp.npy"]["vmin_outside_water"] = float(vp[WATER_SAMPLES:].min())

    for f in FACTORS:
        d = D * f
        ds = block_mean(vp, f)
        name = f"vp_{ds.shape[0]}x{ds.shape[1]}_d{d:g}m.npy"
        np.save(HERE / name, ds)
        vmin = float(ds.min())
        meta[name] = {
            "shape": list(ds.shape), "dz_m": d, "dx_m": d, "dtype": "float32", "factor": f,
            "method": (f"crop to first {ds.shape[0]*f} x {ds.shape[1]*f} node samples (drop the last z row and last "
                       f"x column), then arithmetic mean of velocity (not slowness) over {f}x{f} blocks, float64 "
                       f"accumulate, cast to float32"),
            "extent_m": [ds.shape[0] * d, ds.shape[1] * d],
            "pure_water_rows": f"0..{WATER_SAMPLES // f - 1} (row {WATER_SAMPLES // f} mixes 1 water node with {f-1} sediment nodes)",
            "min": vmin, "max": float(ds.max()),
            "ppw_at_10Hz_min_velocity": round(vmin / (F_CHECK_HZ * d), 3),
            "ppw_at_10Hz_water_1500": round(1500.0 / (F_CHECK_HZ * d), 3),
            "fmax_Hz_for_4ppw_min_velocity": round(vmin / (4.0 * d), 3),
        }

    (HERE / "vp_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    files = sorted(p for p in HERE.rglob("*")
                   if p.is_file() and p.name not in ("README.md", "SHA256SUMS", ".DS_Store")
                   and "__pycache__" not in p.parts)
    lines = [f"{sha256(p)}  {p.relative_to(HERE).as_posix()}" for p in files]
    (HERE / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in meta.items() if k != "_source"}, indent=1))


if __name__ == "__main__":
    main()
