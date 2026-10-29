//! Checked local NPY reader. The frozen shared reader can panic on malformed headers.
use outcrop_common::{Array, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn field<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    for quote in ['\'', '"'] {
        let key = format!("{quote}{key}{quote}:");
        if let Some((_, value)) = header.split_once(&key) {
            return Ok(value.trim_start());
        }
    }
    Err(format!("NPY header missing {key}").into())
}

pub fn read_f32(path: &Path) -> Result<Array<f32>> {
    read_array(path).map_err(|e| format!("{}: {e}", path.display()).into())
}

fn read_array(path: &Path) -> Result<Array<f32>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut prefix = [0u8; 8];
    reader.read_exact(&mut prefix)?;
    if &prefix[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let mut length = [0u8; 4];
    let (header_len, start) = match prefix[6] {
        1 => {
            reader.read_exact(&mut length[..2])?;
            (u16::from_le_bytes([length[0], length[1]]) as usize, 10usize)
        }
        2 | 3 => {
            reader.read_exact(&mut length)?;
            (u32::from_le_bytes(length) as usize, 12usize)
        }
        version => return Err(format!("unsupported .npy version {version}").into()),
    };
    let offset = start.checked_add(header_len).ok_or("NPY header size overflow")?;
    if offset as u64 > file_len {
        return Err("truncated NPY header".into());
    }
    let mut header = Vec::new();
    header.try_reserve_exact(header_len)?;
    header.resize(header_len, 0);
    reader.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header)?;

    let dtype = field(header, "descr")?;
    let quote = dtype.chars().next().ok_or("bad NPY dtype")?;
    if quote != '\'' && quote != '"' {
        return Err("bad NPY dtype".into());
    }
    let dtype = dtype[1..].split(quote).next().ok_or("bad NPY dtype")?;
    let width = match dtype {
        "<f4" => 4usize,
        "<f8" => 8usize,
        _ => return Err(format!("expected float array, got {dtype}").into()),
    };
    let order = field(header, "fortran_order")?;
    if order.starts_with("True") {
        return Err("Fortran-order arrays are not supported (use np.ascontiguousarray)".into());
    }
    if !order.starts_with("False") {
        return Err("bad NPY fortran_order".into());
    }
    let shape_text = field(header, "shape")?.strip_prefix('(').ok_or("bad NPY shape")?;
    let (shape_text, _) = shape_text.split_once(')').ok_or("bad NPY shape")?;
    let mut shape = Vec::with_capacity(3);
    let mut count = 1usize;
    for dimension in shape_text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if shape.len() == 3 {
            return Err("expected a nonempty 3D array".into());
        }
        let dimension = dimension.parse::<usize>()?;
        count = count.checked_mul(dimension).ok_or("NPY shape size overflow")?;
        shape.push(dimension);
    }
    if shape.len() != 3 || shape.contains(&0) {
        return Err("expected a nonempty 3D array".into());
    }
    if count > u32::MAX as usize {
        return Err("grid is too large for Metal indexing".into());
    }
    let data_len = count.checked_mul(width).ok_or("NPY data size overflow")?;
    let end = offset.checked_add(data_len).ok_or("NPY file size overflow")?;
    if end as u64 > file_len {
        return Err("truncated NPY data".into());
    }
    let mut data = Vec::new();
    data.try_reserve_exact(count)?;
    // Decode bounded chunks rather than allocating another whole-file copy.
    let mut bytes = [0u8; 8192];
    while data.len() < count {
        let elements = (count - data.len()).min(bytes.len() / width);
        reader.read_exact(&mut bytes[..elements * width])?;
        for value in bytes[..elements * width].chunks_exact(width) {
            data.push(if width == 4 {
                f32::from_le_bytes([value[0], value[1], value[2], value[3]])
            } else {
                f64::from_le_bytes([
                    value[0], value[1], value[2], value[3],
                    value[4], value[5], value[6], value[7],
                ]) as f32
            });
        }
    }
    Ok(Array { shape, data })
}
