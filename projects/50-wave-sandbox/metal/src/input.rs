//! Local, bounds-checked array input. The shared CLI still handles dispatch,
//! JSON scalars and output, but its NPY reader assumes well-formed headers.
use outcrop_common::{Array, Inputs, Result};
use std::ops::Deref;
use std::path::PathBuf;

pub struct CheckedInputs<'a> {
    scalars: &'a Inputs,
    dir: PathBuf,
}

impl Deref for CheckedInputs<'_> {
    type Target = Inputs;
    fn deref(&self) -> &Inputs { self.scalars }
}

struct Npy {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    descr: String,
    start: usize,
    count: usize,
}

impl<'a> CheckedInputs<'a> {
    pub fn new(scalars: &'a Inputs) -> Result<Self> {
        // Inputs hides its directory; the frozen CLI supplies it as argv[2].
        let dir = std::env::args_os().nth(2).ok_or("missing input directory")?.into();
        Ok(Self { scalars, dir })
    }

    fn array(&self, name: &str) -> Result<Npy> {
        let path = self.dir.join(format!("{name}.npy"));
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let parsed = (|| -> Result<_> {
            if bytes.len() < 10 || bytes.get(..6) != Some(b"\x93NUMPY") {
                return Err("not a .npy file".into());
            }
            let (hlen, start) = match bytes[6] {
                1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
                2 | 3 => {
                    let length = bytes.get(8..12).ok_or("truncated .npy version 2/3 prefix")?;
                    (usize::try_from(u32::from_le_bytes(length.try_into()?))?, 12usize)
                }
                v => return Err(format!("unsupported .npy version {v}").into()),
            };
            let start = start.checked_add(hlen).ok_or(".npy header length overflow")?;
            let header_start = if bytes[6] == 1 { 10 } else { 12 };
            let header = std::str::from_utf8(bytes.get(header_start..start).ok_or("truncated .npy header")?)?;
            let field = |key: &str| -> Result<&str> {
                let key = format!("'{key}':");
                let i = header.find(&key).ok_or_else(|| format!("npy header missing {key}"))?;
                Ok(header[i + key.len()..].trim_start())
            };
            let descr = field("descr")?.split('\'').nth(1).ok_or("bad descr")?.to_string();
            if field("fortran_order")?.starts_with("True") {
                return Err("Fortran-order arrays are not supported (use np.ascontiguousarray)".into());
            }
            let shape = field("shape")?;
            let begin = shape.find('(').ok_or("bad shape")?;
            let end = shape.find(')').ok_or("bad shape")?;
            let inner = shape.get(begin + 1..end).ok_or("bad shape")?;
            let shape = inner.split(',').map(str::trim).filter(|s| !s.is_empty())
                .map(str::parse::<usize>).collect::<std::result::Result<Vec<_>, _>>()?;
            let count = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(".npy shape product overflow")?;
            let size = match descr.as_str() {
                "<f4" | "<i4" => 4,
                "<f8" | "<i8" => 8,
                _ => return Err(format!("unsupported array dtype {descr}").into()),
            };
            let length = count.checked_mul(size).filter(|&n| n <= isize::MAX as usize)
                .ok_or(".npy data length overflow")?;
            let end = start.checked_add(length).ok_or(".npy data offset overflow")?;
            if end > bytes.len() { return Err("truncated .npy data".into()); }
            Ok((shape, descr, start, count))
        })().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Npy { bytes, shape: parsed.0, descr: parsed.1, start: parsed.2, count: parsed.3 })
    }

    pub fn f32(&self, name: &str) -> Result<Array<f32>> {
        let a = self.array(name)?;
        let data = match a.descr.as_str() {
            "<f4" => a.bytes[a.start..].as_chunks::<4>().0.iter().take(a.count)
                .map(|b| f32::from_le_bytes(*b)).collect(),
            "<f8" => a.bytes[a.start..].as_chunks::<8>().0.iter().take(a.count)
                .map(|b| f64::from_le_bytes(*b) as f32).collect(),
            d => return Err(format!("{name}: expected float array, got {d}").into()),
        };
        Ok(Array { shape: a.shape, data })
    }

    pub fn i64(&self, name: &str) -> Result<Array<i64>> {
        let a = self.array(name)?;
        let data = match a.descr.as_str() {
            "<i4" => a.bytes[a.start..].as_chunks::<4>().0.iter().take(a.count)
                .map(|b| i32::from_le_bytes(*b) as i64).collect(),
            "<i8" => a.bytes[a.start..].as_chunks::<8>().0.iter().take(a.count)
                .map(|b| i64::from_le_bytes(*b)).collect(),
            d => return Err(format!("{name}: expected integer array, got {d}").into()),
        };
        Ok(Array { shape: a.shape, data })
    }
}
