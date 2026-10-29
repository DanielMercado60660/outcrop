use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};
use std::io::Read;

// Inputs::f32 uses the shared NPY parser, which assumes a complete header and
// an ordered shape tuple. Validate those assumptions before calling it.
fn check_npy_header(name: &str) -> Result<()> {
    let dir = std::env::args_os().nth(2).ok_or("missing input directory")?;
    let path = std::path::Path::new(&dir).join(format!("{name}.npy"));
    let mut file = std::fs::File::open(&path)?;
    let mut bytes = [0u8; 4096];
    let len = file.read(&mut bytes)?;
    let bytes = &bytes[..len];
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{name}: invalid or truncated NPY header").into());
    }
    let (start, hlen) = match bytes[6] {
        1 => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 if bytes.len() >= 12 => (
            12usize,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        ),
        2 | 3 => return Err(format!("{name}: truncated NPY header length").into()),
        v => return Err(format!("{name}: unsupported NPY version {v}").into()),
    };
    let end = start.checked_add(hlen).ok_or("NPY header length overflow")?;
    let header = std::str::from_utf8(bytes.get(start..end).ok_or("truncated NPY header")?)?;
    let shape_key = "'shape':";
    let shape = header.find(shape_key).map(|i| &header[i + shape_key.len()..])
        .ok_or("NPY header missing shape")?;
    let begin = shape.find('(').ok_or("bad NPY shape")? + 1;
    let end = shape.find(')').ok_or("bad NPY shape")?;
    let dims = shape.get(begin..end).ok_or("bad NPY shape")?;
    let mut count = 1usize;
    for dim in dims.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        count = count.checked_mul(dim.parse::<usize>()?).ok_or("NPY shape overflow")?;
    }
    count.checked_mul(8).ok_or("NPY data size overflow")?;
    Ok(())
}

fn propagate(input: &Inputs, output: &mut Outputs) -> Result<()> {
    for name in ["epsilon_r", "electric", "hx", "hz"] {
        check_npy_header(name)?;
    }
    let eps = input.f32("epsilon_r")?;
    let e = input.f32("electric")?;
    let hx = input.f32("hx")?;
    let hz = input.f32("hz")?;
    if e.shape.len() != 2 || e.shape[0] == 0 || e.shape[1] == 0 {
        return Err("electric must have a nonempty two-dimensional shape".into());
    }
    if eps.shape != e.shape || hx.shape != e.shape || hz.shape != e.shape {
        return Err("field and permittivity shapes must match".into());
    }
    let nx = u32::try_from(e.shape[0])?;
    let nz = u32::try_from(e.shape[1])?;
    let n = e.data.len();
    if n > u32::MAX as usize {
        return Err("grid is too large for Metal indexing".into());
    }
    let dx = input.num("dx")?;
    let dz = input.num("dz")?;
    let dt = input.num("dt")?;
    let steps = input.int("steps")?;
    if dx <= 0.0 || dz <= 0.0 || dt < 0.0 || steps < 0 {
        return Err("spacings must be positive; dt and steps must be nonnegative".into());
    }
    if steps == 0 {
        output.f32("electric", &e.shape, &e.data)?;
        output.f32("hx", &e.shape, &hx.data)?;
        output.f32("hz", &e.shape, &hz.data)?;
        return Ok(());
    }
    let cx = (299_792_458.0 * dt / dx) as f32;
    let cz = (299_792_458.0 * dt / dz) as f32;
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/fdtd.metal"))?;
    let magnetic = gpu.pipeline(&library, "update_magnetic")?;
    let electric = gpu.pipeline(&library, "update_electric")?;
    let be = gpu.buffer_from(&e.data);
    let bhx = gpu.buffer_from(&hx.data);
    let bhz = gpu.buffer_from(&hz.data);
    let beps = gpu.buffer_from(&eps.data);
    // Bound command-buffer size for long simulations.
    let mut remaining = steps;
    while remaining > 0 {
        let count = remaining.min(128);
        gpu.batch(|batch| {
            for _ in 0..count {
                batch.dispatch(&magnetic, &[
                    Arg::Buf(&be), Arg::Buf(&bhx), Arg::Buf(&bhz),
                    Arg::U32(nx), Arg::U32(nz), Arg::F32(cx), Arg::F32(cz),
                ], [n, 1, 1])?;
                batch.dispatch(&electric, &[
                    Arg::Buf(&be), Arg::Buf(&bhx), Arg::Buf(&bhz), Arg::Buf(&beps),
                    Arg::U32(nx), Arg::U32(nz), Arg::F32(cx), Arg::F32(cz),
                ], [n, 1, 1])?;
            }
            Ok(())
        })?;
        remaining -= count;
    }
    output.f32("electric", &e.shape, &gpu.read::<f32>(&be, n))?;
    output.f32("hx", &e.shape, &gpu.read::<f32>(&bhx, n))?;
    output.f32("hz", &e.shape, &gpu.read::<f32>(&bhz, n))?;
    Ok(())
}

fn main() {
    outcrop_common::run(&[("propagate", propagate)]);
}
