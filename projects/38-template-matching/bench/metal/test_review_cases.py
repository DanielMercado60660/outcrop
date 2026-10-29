"""Optional regression checks run on the GPU; no golden files used."""
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from outcrop_harness import call, metal_binary


PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def project_environment(monkeypatch):
    monkeypatch.setenv("OUTCROP_PROJECT", str(PROJECT))


def assert_pearson(data, templates, threshold=0.8):
    result = call("metal", "template_match", data=data, templates=templates,
                  threshold=threshold)
    t, s, m = templates.shape
    windows = data.shape[1] - m + 1
    expected = np.zeros((t, s, windows), dtype=np.float64)
    for ti in range(t):
        for si in range(s):
            y = templates[ti, si].astype(np.float64)
            y -= y.mean()
            for k in range(windows):
                x = data[si, k:k + m].astype(np.float64)
                x -= x.mean()
                xx, yy = np.sum(x * x), np.sum(y * y)
                if xx > 0 and yy > 0:
                    expected[ti, si, k] = np.sum(x * y) / np.sqrt(xx) / np.sqrt(yy)
    assert set(result) == {"correlation", "detections"}
    for value in result.values():
        assert value.shape == expected.shape
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
    np.testing.assert_allclose(result["correlation"], expected, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(result["detections"], (expected >= threshold).astype(np.float32))
    np.testing.assert_array_equal(
        result["detections"],
        (result["correlation"].astype(np.float64) >= threshold).astype(np.float32),
    )
    return result


@pytest.mark.parametrize("data_scale,template_scale", [
    (1e-23, 1e-23), (1e20, 1e20), (1e-23, 1e20), (1e20, 1e-23),
])
def test_extreme_amplitudes_metal(data_scale, template_scale):
    trace = np.array([0, 2, -1, 3, 0, 2, -1, 3], dtype=np.float64)
    data = (np.stack([trace, np.ones(8)]) * data_scale).astype(np.float32)
    wave = np.stack([trace[:4], trace[:4]])
    templates = (np.stack([wave, -wave, np.ones_like(wave)]) * template_scale).astype(np.float32)
    result = assert_pearson(data, templates)
    np.testing.assert_array_equal(result["correlation"][:, 1, :], 0)
    np.testing.assert_array_equal(result["correlation"][2], 0)


@pytest.mark.parametrize("bits", [
    [1, 2, 3, 1],                       # Smallest subnormal significands.
    [0x7f7fffff, 0xff7fffff, 0, 1],     # Both signs of the largest finite value.
    [0x7f7ffffd, 0x7f7ffffe, 0x7f7fffff, 0x7f7ffffe],  # Tiny changes on large DC.
])
def test_float32_endpoints_metal(bits):
    wave = np.array(bits, dtype=np.uint32).view(np.float32)
    negative = (np.array(bits, dtype=np.uint32) ^ np.uint32(0x80000000)).view(np.float32)
    data = np.tile(wave, 2)[None, :]
    templates = np.stack([wave, negative, np.full_like(wave, wave[0])])[:, None, :]
    result = assert_pearson(data, templates)
    np.testing.assert_array_equal(result["correlation"][2], 0)


def npy_bytes(header, version=1, payload=b""):
    encoded = header.encode("utf-8")
    length = struct.pack("<H" if version == 1 else "<I", len(encoded))
    return b"\x93NUMPY" + bytes([version, 0]) + length + encoded + payload


BAD_FILES = [
    b"", b"\x93NUMPY\x01\x00\x00",
    b"\x93NUMPY\x02\x00\x00\x00", b"\x93NUMPY\x03\x00\x00\x00\x00",
    b"\x93NUMPY\x01\x00\xff\xff",
    b"\x93NUMPY\x02\x00\xff\xff\xff\xff",
    b"\x93NUMPY\x03\x00\xff\xff\xff\xff",
    npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': )(2, 3), }"),
    npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (18446744073709551615, 2), }"),
    npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (4611686018427387904,), }"),
    npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (1, 3), }", payload=b"\0" * 4),
]


@pytest.mark.parametrize("name", ["data", "templates"])
@pytest.mark.parametrize("malformed", BAD_FILES)
def test_malformed_npy_returns_error_metal(tmp_path, name, malformed):
    ind, outd = tmp_path / "in", tmp_path / "out"
    ind.mkdir()
    outd.mkdir()
    np.save(ind / "data.npy", np.array([[1, 2, 3]], dtype=np.float32))
    np.save(ind / "templates.npy", np.array([[[1, 2]]], dtype=np.float32))
    (ind / "args.json").write_text(json.dumps({"threshold": 0.8}))
    (ind / f"{name}.npy").write_bytes(malformed)
    result = subprocess.run([str(metal_binary(PROJECT)), "template_match", str(ind), str(outd)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "error in template_match:" in result.stderr
    assert "panicked" not in result.stderr


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["<f4", "<f8"])
def test_valid_npy_versions_metal(tmp_path, version, dtype):
    ind, outd = tmp_path / "in", tmp_path / "out"
    ind.mkdir()
    outd.mkdir()
    for name, shape in [("data", (1, 3)), ("templates", (1, 1, 3))]:
        header = f"{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}"
        values = np.array([1, 2, 3], dtype=dtype)
        (ind / f"{name}.npy").write_bytes(npy_bytes(header, version, values.tobytes()))
    (ind / "args.json").write_text(json.dumps({"threshold": 0.8}))
    result = subprocess.run([str(metal_binary(PROJECT)), "template_match", str(ind), str(outd)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(np.load(outd / "correlation.npy"), 1, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(np.load(outd / "detections.npy"), np.ones((1, 1, 1)))
