//! Local checked NPY reader until the frozen common reader validates bounds.
use outcrop_common::{Array, Result};

pub fn f32(name: &str) -> Result<Array<f32>> {
    let dir = std::env::args_os().nth(2).ok_or("missing input directory")?;
    let path = std::path::PathBuf::from(dir).join(format!("{name}.npy"));
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let invalid = || format!("{}: malformed or truncated NPY header", path.display());
    if bytes.get(..6) != Some(b"\x93NUMPY") { return Err(invalid().into()); }
    let major = *bytes.get(6).ok_or_else(invalid)?;
    let (start, length) = match major {
        1 => {
            let raw: [u8; 2] = bytes.get(8..10).ok_or_else(invalid)?.try_into()?;
            (10usize, u16::from_le_bytes(raw) as usize)
        }
        2 | 3 => {
            let raw: [u8; 4] = bytes.get(8..12).ok_or_else(invalid)?.try_into()?;
            (12usize, u32::from_le_bytes(raw) as usize)
        }
        v => return Err(format!("unsupported .npy version {v}").into()),
    };
    let end = start.checked_add(length).ok_or_else(invalid)?;
    let header = std::str::from_utf8(bytes.get(start..end).ok_or_else(invalid)?)?;
    let field = |key: &str| -> Result<&str> {
        let key = format!("'{key}':");
        let i = header.find(&key).ok_or_else(invalid)? + key.len();
        Ok(header[i..].trim_start())
    };
    let descr = field("descr")?.split('\'').nth(1).ok_or_else(invalid)?;
    let size = match descr {
        "<f4" => 4,
        "<f8" => 8,
        d => return Err(format!("{name}: expected float array, got {d}").into()),
    };
    let fortran = field("fortran_order")?;
    if fortran.starts_with("True") {
        return Err(format!("{}: Fortran-order arrays are not supported (use np.ascontiguousarray)", path.display()).into());
    }
    if !fortran.starts_with("False") { return Err(invalid().into()); }
    let shape_text = field("shape")?;
    let open = shape_text.find('(').ok_or_else(invalid)? + 1;
    let remainder = &shape_text[open..];
    let close = remainder.find(')').ok_or_else(invalid)?;
    let shape = remainder[..close].split(',').map(str::trim)
        .filter(|s| !s.is_empty()).map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let n = shape.iter().try_fold(1usize, |n, &s| n.checked_mul(s))
        .ok_or("NPY shape product overflow")?;
    let payload_size = n.checked_mul(size).ok_or("NPY payload length overflow")?;
    let data_end = end.checked_add(payload_size).ok_or("NPY payload length overflow")?;
    let raw = bytes.get(end..data_end).ok_or("truncated NPY data")?;
    let mut data = Vec::new();
    data.try_reserve_exact(n)?;
    for chunk in raw.chunks_exact(size) {
        let v = if size == 4 { f32::from_le_bytes(chunk.try_into()?) }
                else { f64::from_le_bytes(chunk.try_into()?) as f32 };
        data.push(v);
    }
    Ok(Array { shape, data })
}
