//! Local replacement for the shared float reader until its unchecked NPY
//! header slices and shape arithmetic are hardened. No GPU math happens here.

use outcrop_common::{Array, Result};
use std::{fs, path::Path};

/// Read the same little-endian, C-order f32/f64 arrays as Inputs::f32,
/// checking all file ranges and size arithmetic before decoding any values.
pub fn read_f32(input_dir: &Path, name: &str) -> Result<Array<f32>> {
    let path = input_dir.join(format!("{name}.npy"));
    if !path.exists() {
        return Err(format!("missing array input {name:?}").into());
    }
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_f32(&bytes, name)
}

fn decode_f32(bytes: &[u8], name: &str) -> Result<Array<f32>> {
    if bytes.len() < 10 || bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err("not a .npy file".into());
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => {
            let length = bytes.get(8..12).ok_or("truncated .npy header length")?;
            (
                u32::from_le_bytes([length[0], length[1], length[2], length[3]]) as usize,
                12usize,
            )
        }
        v => return Err(format!("unsupported .npy version {v}").into()),
    };
    let end = start.checked_add(header_len).ok_or(".npy header length overflow")?;
    let header = std::str::from_utf8(bytes.get(start..end).ok_or("truncated .npy header")?)?;
    // Match the shared reader's header syntax and dtype conversion semantics.
    let field = |key: &str| -> Result<&str> {
        let key = format!("'{key}':");
        let offset = header.find(&key).ok_or_else(|| format!("npy header missing {key}"))?;
        Ok(header[offset + key.len()..].trim_start())
    };
    let descr = field("descr")?.split('\'').nth(1).ok_or("bad descr")?;
    let element_size = match descr {
        "<f4" => 4usize,
        "<f8" => 8usize,
        d => return Err(format!("{name}: expected float array, got {d}").into()),
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
        .filter(|x| !x.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let count = shape
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .ok_or(".npy shape product overflow")?;
    let size = count.checked_mul(element_size).ok_or(".npy data size overflow")?;
    let data_end = end.checked_add(size).ok_or(".npy data offset overflow")?;
    let raw = bytes.get(end..data_end).ok_or("truncated .npy data")?;
    let mut data = Vec::new();
    data.try_reserve_exact(count)?;
    if element_size == 4 {
        data.extend(raw.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)));
    } else {
        data.extend(raw.as_chunks::<8>().0.iter().map(|&b| f64::from_le_bytes(b) as f32));
    }
    Ok(Array { shape, data })
}
