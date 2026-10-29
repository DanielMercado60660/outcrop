//! Bounded NPY input reader, local while the shared reader is frozen.
use outcrop_common::{Array, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn field<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let key = format!("'{key}':");
    let start = header.find(&key).ok_or("missing NPY header field")? + key.len();
    Ok(header[start..].trim_start())
}

pub fn read_f32(path: &Path) -> Result<Array<f32>> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    // read_exact checks every version's length field before any indexing.
    let (header_len, prefix_len) = match magic[6] {
        1 => {
            let mut bytes = [0u8; 2];
            file.read_exact(&mut bytes)?;
            (usize::from(u16::from_le_bytes(bytes)), 10usize)
        }
        2 | 3 => {
            let mut bytes = [0u8; 4];
            file.read_exact(&mut bytes)?;
            (usize::try_from(u32::from_le_bytes(bytes))?, 12usize)
        }
        version => return Err(format!("unsupported .npy version {version}").into()),
    };
    let offset = prefix_len
        .checked_add(header_len)
        .ok_or("NPY header too large")?;
    if u64::try_from(offset)? > file_len {
        return Err("truncated NPY header".into());
    }
    let mut header = Vec::new();
    header.try_reserve_exact(header_len)?;
    header.resize(header_len, 0u8);
    file.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header)?;
    let descr = field(header, "descr")?
        .split('\'')
        .nth(1)
        .ok_or("bad descr")?;
    let item_size = match descr {
        "<f4" => 4usize,
        "<f8" => 8usize,
        _ => return Err(format!("expected float array, got {descr}").into()),
    };
    let order = field(header, "fortran_order")?;
    if order.starts_with("True") {
        return Err("Fortran-order arrays are not supported (use np.ascontiguousarray)".into());
    }
    if !order.starts_with("False") {
        return Err("bad fortran_order".into());
    }
    let tuple = field(header, "shape")?.trim_start();
    let tuple = tuple.strip_prefix('(').ok_or("bad shape")?;
    let end = tuple.find(')').ok_or("bad shape")?;
    let mut shape = Vec::new();
    for dim in tuple[..end]
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
    {
        let dim = dim.parse::<usize>()?;
        shape.try_reserve(1)?;
        shape.push(dim);
    }
    let count = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or("NPY shape too large")?
    };
    let payload = count.checked_mul(item_size).ok_or("NPY payload too large")?;
    let end = offset.checked_add(payload).ok_or("NPY file too large")?;
    if u64::try_from(end)? > file_len {
        return Err("truncated NPY data".into());
    }

    // Allocate fallibly and decode in bounded chunks instead of retaining a
    // second full-sized copy of the input file. Trailing bytes are allowed,
    // matching the shared reader. f64 inputs are rounded to f32 as before.
    let mut data = Vec::new();
    data.try_reserve_exact(count)?;
    let mut block = [0u8; 8192];
    while data.len() < count {
        let values = (count - data.len()).min(block.len() / item_size);
        let bytes = &mut block[..values * item_size];
        file.read_exact(bytes)?;
        for chunk in bytes.chunks_exact(item_size) {
            let value = if item_size == 4 {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            } else {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32
            };
            data.push(value);
        }
    }
    Ok(Array { shape, data })
}
