//! Minimal `.npy` (format 1.0/2.0/3.0) reader and writer: little-endian, C order.
//! Supported dtypes: f32 (<f4), f64 (<f8), i32 (<i4), i64 (<i8), u8 (|u1), bool (|b1).

use crate::Result;
use std::fs;
use std::path::Path;

/// An n-dimensional array in C (row-major) order.
#[derive(Clone, Debug, PartialEq)]
pub struct Array<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: Clone> Array<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape does not match data length");
        Array { shape, data }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Element types that can be stored in `.npy`.
pub trait Element: Copy + Sized {
    const DESCR: &'static [&'static str];
    const SIZE: usize;
    fn from_le(b: &[u8]) -> Self;
    fn to_le(self, out: &mut Vec<u8>);
    fn write_descr() -> &'static str {
        Self::DESCR[0]
    }
}

macro_rules! num_elem {
    ($t:ty, $size:expr, $($d:expr),+) => {
        impl Element for $t {
            const DESCR: &'static [&'static str] = &[$($d),+];
            const SIZE: usize = $size;
            fn from_le(b: &[u8]) -> Self { <$t>::from_le_bytes(b.try_into().unwrap()) }
            fn to_le(self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()) }
        }
    };
}
num_elem!(f32, 4, "<f4");
num_elem!(f64, 8, "<f8");
num_elem!(i32, 4, "<i4");
num_elem!(i64, 8, "<i8");
num_elem!(u8, 1, "|u1", "<u1");

impl Element for bool {
    const DESCR: &'static [&'static str] = &["|b1"];
    const SIZE: usize = 1;
    fn from_le(b: &[u8]) -> Self {
        b[0] != 0
    }
    fn to_le(self, out: &mut Vec<u8>) {
        out.push(self as u8)
    }
}

struct Header {
    descr: String,
    fortran: bool,
    shape: Vec<usize>,
    data_offset: usize,
}

fn parse_header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = bytes[6];
    let (hlen, start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        v => return Err(format!("unsupported .npy version {v}").into()),
    };
    let h = std::str::from_utf8(&bytes[start..start + hlen])?;
    let field = |key: &str| -> Result<String> {
        let k = format!("'{key}':");
        let i = h.find(&k).ok_or_else(|| format!("npy header missing {key}"))? + k.len();
        Ok(h[i..].trim_start().to_string())
    };
    let d = field("descr")?;
    let descr = d.split('\'').nth(1).ok_or("bad descr")?.to_string();
    let fortran = field("fortran_order")?.starts_with("True");
    let s = field("shape")?;
    let inner = &s[s.find('(').ok_or("bad shape")? + 1..s.find(')').ok_or("bad shape")?];
    let shape = inner
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(Header { descr, fortran, shape, data_offset: start + hlen })
}

/// Read a `.npy` file with element type `T`. Errors if the dtype does not match exactly.
pub fn read<T: Element + Clone>(path: impl AsRef<Path>) -> Result<Array<T>> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let h = parse_header(&bytes)?;
    if !T::DESCR.contains(&h.descr.as_str()) {
        return Err(format!("{}: dtype {} but expected {}", path.display(), h.descr, T::DESCR[0]).into());
    }
    if h.fortran {
        return Err(format!("{}: Fortran-order arrays are not supported (use np.ascontiguousarray)", path.display()).into());
    }
    let n: usize = h.shape.iter().product();
    let raw = &bytes[h.data_offset..];
    if raw.len() < n * T::SIZE {
        return Err(format!("{}: truncated data", path.display()).into());
    }
    let data = raw.chunks_exact(T::SIZE).take(n).map(T::from_le).collect();
    Ok(Array { shape: h.shape, data })
}

/// The dtype string (e.g. "<f4") of a `.npy` file, without reading its data.
pub fn dtype(path: impl AsRef<Path>) -> Result<String> {
    let mut f = fs::File::open(path.as_ref())?;
    let mut buf = vec![0u8; 4096];
    let n = std::io::Read::read(&mut f, &mut buf)?;
    Ok(parse_header(&buf[..n])?.descr)
}

/// Write `data` with `shape` as a version 1.0 `.npy` file.
pub fn write<T: Element>(path: impl AsRef<Path>, shape: &[usize], data: &[T]) -> Result<()> {
    assert_eq!(shape.iter().product::<usize>(), data.len(), "shape does not match data length");
    let shape_s = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", T::write_descr(), shape_s);
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * T::SIZE);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &x in data {
        x.to_le(&mut out);
    }
    fs::write(path.as_ref(), out).map_err(|e| format!("{}: {e}", path.as_ref().display()).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("outcrop-npy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.npy");
        write(&p, &[2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let a: Array<f32> = read(&p).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data[5], 6.0);
        write(&p, &[], &[7i64]).unwrap();
        let b: Array<i64> = read(&p).unwrap();
        assert_eq!(b.shape, Vec::<usize>::new());
        assert_eq!(b.data, vec![7]);
        assert!(read::<f64>(&p).is_err());
    }
}
