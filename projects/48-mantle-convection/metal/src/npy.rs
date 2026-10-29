//! Bounds-checked local replacement for Inputs::f32("temperature").
//! Keep the shared reader's dtype conversion and C-order conventions without
//! passing untrusted header lengths, shape ranges, or products to that parser.

use outcrop_common::{Array, Result};
use std::path::Path;

pub fn read_temperature(path: &Path) -> Result<Array<f32>> {
    if !path.exists() {
        return Err("missing array input \"temperature\"".into());
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode(&bytes).map_err(|e| format!("{}: {e}", path.display()).into())
}

fn decode(bytes: &[u8]) -> Result<Array<f32>> {
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err("not a .npy file".into());
    }
    let version = bytes.get(6..8).ok_or("truncated .npy version")?;
    let (start, length): (usize, usize) = match version[0] {
        1 => {
            let raw = bytes.get(8..10).ok_or("truncated .npy header length")?;
            (10, u16::from_le_bytes(raw.try_into()?) as usize)
        }
        2 | 3 => {
            let raw = bytes.get(8..12).ok_or("truncated .npy header length")?;
            (12, u32::from_le_bytes(raw.try_into()?) as usize)
        }
        v => return Err(format!("unsupported .npy version {v}").into()),
    };
    let end = start.checked_add(length).ok_or(".npy header length overflow")?;
    let header = std::str::from_utf8(bytes.get(start..end).ok_or("truncated .npy header")?)?;
    // Match the existing reader's single-quoted field syntax and conversions.
    let field = |key: &str| -> Result<&str> {
        let key = format!("'{key}':");
        let start = header.find(&key).ok_or_else(|| format!("npy header missing {key}"))?
            + key.len();
        Ok(header.get(start..).ok_or("bad .npy field")?.trim_start())
    };
    let descr = field("descr")?.split('\'').nth(1).ok_or("bad descr")?;
    let width = match descr {
        "<f4" => 4,
        "<f8" => 8,
        d => return Err(format!("temperature: expected float array, got {d}").into()),
    };
    if field("fortran_order")?.starts_with("True") {
        return Err("Fortran-order arrays are not supported (use np.ascontiguousarray)".into());
    }
    let shape_field = field("shape")?;
    let open = shape_field.find('(').ok_or("bad shape")?;
    let close = shape_field.find(')').ok_or("bad shape")?;
    let inner = shape_field.get(open + 1..close).ok_or("bad shape")?;
    let shape = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let count = shape.iter().try_fold(1usize, |n, &s| {
        n.checked_mul(s).ok_or(".npy shape product overflow")
    })?;
    let size = count.checked_mul(width).ok_or(".npy data length overflow")?;
    let data_end = end.checked_add(size).ok_or(".npy data offset overflow")?;
    let raw = bytes.get(end..data_end).ok_or("truncated data")?;
    let mut data = Vec::new();
    data.try_reserve_exact(count)?;
    // Chunk lengths are exact; conversions are identical to Inputs::f32.
    for chunk in raw.chunks_exact(width) {
        data.push(if width == 4 {
            f32::from_le_bytes(chunk.try_into()?)
        } else {
            f64::from_le_bytes(chunk.try_into()?) as f32
        });
    }
    Ok(Array { shape, data })
}

#[cfg(test)]
mod tests {
    use super::decode;

    fn file(version: u8, header: &str, payload: &[u8]) -> Vec<u8> {
        let mut bytes = b"\x93NUMPY".to_vec();
        bytes.extend_from_slice(&[version, 0]);
        if version == 1 {
            bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        } else {
            bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        }
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn valid_versions_and_float_conversion() {
        for version in [1, 2, 3] {
            for dtype in ["<f4", "<f8"] {
                let header = format!(
                    "{{'descr': '{dtype}', 'fortran_order': False, 'shape': (1, 2), }}\n"
                );
                let values = [0.1f64, -1.25];
                let payload: Vec<u8> = values
                    .iter()
                    .flat_map(|&v| {
                        if dtype == "<f4" {
                            (v as f32).to_le_bytes().to_vec()
                        } else {
                            v.to_le_bytes().to_vec()
                        }
                    })
                    .collect();
                let bytes = file(version, &header, &payload);
                let a = decode(&bytes).unwrap();
                assert_eq!(a.shape, [1, 2]);
                assert_eq!(a.data, values.map(|v| v as f32));
                // Every incomplete version/length/header/payload must fail.
                for length in 0..bytes.len() {
                    assert!(
                        decode(&bytes[..length]).is_err(),
                        "v{version}, {dtype}, {length}"
                    );
                }
            }
        }
    }

    #[test]
    fn malformed_headers_and_overflow_return_errors() {
        let shapes = [
            ")(",
            "(1,",
            "(garbage,)",
            "(-1,)",
            "(18446744073709551615, 2)",
            "(18446744073709551615,)",
        ];
        for shape in shapes {
            let h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}\n");
            assert!(decode(&file(2, &h, &[])).is_err());
        }
        for header in [
            "{}",
            "{'descr': '<f4'}",
            "{'descr': <f4}",
            "{'descr': '<i4', 'fortran_order': False, 'shape': (0,), }",
            "{'descr': '<f4', 'fortran_order': True, 'shape': (0,), }",
        ] {
            assert!(decode(&file(1, header, &[])).is_err());
        }
        let mut bytes = file(3, "", &[]);
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&bytes).is_err());
        assert!(decode(&file(4, "", &[])).is_err());
        let mut invalid_utf8 = file(3, "é", &[]);
        invalid_utf8[12] = 0xff;
        assert!(decode(&invalid_utf8).is_err());
    }
}
