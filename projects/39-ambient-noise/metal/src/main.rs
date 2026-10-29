use outcrop_common::{gpu::Arg, Array, Gpu, Inputs, Outputs, Result};
use std::path::Path;

fn read_array(path: &Path) -> Result<Array<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" { return Err("invalid npy header".into()); }
    let (start, len) = match bytes[6] {
        1 => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 if bytes.len() >= 12 => (12, u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize),
        _ => return Err("invalid npy version or truncated header".into()),
    };
    let end = start.checked_add(len).filter(|&x| x <= bytes.len()).ok_or("truncated npy header")?;
    let h = std::str::from_utf8(&bytes[start..end])?;
    let field = |key: &str| -> Result<&str> {
        let k = format!("'{key}':");
        let pos = h.find(&k).ok_or("missing npy field")? + k.len();
        Ok(h[pos..].trim_start())
    };
    let descr = field("descr")?.split('\'').nth(1).ok_or("invalid npy dtype")?;
    let size = match descr { "<f4" => 4usize, "<f8" => 8, _ => return Err("expected float array".into()) };
    if field("fortran_order")?.starts_with("True") { return Err("Fortran-order arrays are not supported".into()); }
    let s = field("shape")?;
    let open = s.find('(').ok_or("invalid npy shape")?;
    let close = s.find(')').filter(|&i| i > open).ok_or("invalid npy shape")?;
    let shape: Vec<usize> = s[open+1..close].split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::parse).collect::<std::result::Result<_, _>>()?;
    let count = shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b)).ok_or("npy shape overflow")?;
    let length = count.checked_mul(size).ok_or("npy data size overflow")?;
    let raw = bytes.get(end..end.checked_add(length).ok_or("npy data size overflow")?).ok_or("truncated npy data")?;
    let data = raw.chunks_exact(size).map(|v| if size == 4 { f32::from_le_bytes([v[0],v[1],v[2],v[3]]) } else { f64::from_le_bytes([v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]]) as f32 }).collect();
    Ok(Array { shape, data })
}

fn cutoff(eps: f64) -> f32 {
    // For an f32 magnitude m, m > floor_f32(eps) iff f64(m) > eps.
    // In particular, do not round a Python cutoff upward across a retained bin.
    let x = eps as f32;
    if (x as f64) <= eps || x.is_nan() { return x; }
    if x == 0.0 { return -f32::from_bits(1); }
    if x > 0.0 { f32::from_bits(x.to_bits() - 1) } else { f32::from_bits(x.to_bits() + 1) }
}

fn ambient_noise(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let dir = std::env::args_os().nth(2).ok_or("missing input directory")?;
    let a = read_array(&Path::new(&dir).join("a.npy"))?;
    let b = read_array(&Path::new(&dir).join("b.npy"))?;
    if a.shape.len() != 2 || a.shape != b.shape || a.shape[0] == 0 || a.shape[1] == 0 { return Err("a and b must have equal nonempty (windows, samples) shapes".into()); }
    let (windows, n) = (a.shape[0], a.shape[1]);
    let dt = inp.num("dt")?; let low = inp.num("low")?; let high = inp.num("high")?; let eps = cutoff(inp.num("eps")?);
    if !dt.is_finite() || dt <= 0.0 || n > 65535 || windows > u32::MAX as usize / n { return Err("invalid sampling interval or array dimensions".into()); }
    let bins = n / 2 + 1;
    // Python scalars retain f64 precision here, including inclusive band edges.
    let mask: Vec<u32> = (0..bins).map(|k| { let f = k as f64 / (n as f64 * dt); u32::from(f >= low && f <= high) }).collect();
    let gpu = Gpu::new()?; let lib = gpu.compile(include_str!("../kernels/ambient.metal"))?;
    let ba = gpu.buffer_from(&a.data); let bb = gpu.buffer_from(&b.data); let bm = gpu.buffer_from(&mask);
    let wa = gpu.buffer_zeroed::<f32>(windows * n); let wb = gpu.buffer_zeroed::<f32>(windows * n);
    let corr = gpu.buffer_zeroed::<f32>(windows * n); let stack = gpu.buffer_zeroed::<f32>(n);
    let stack_kernel = gpu.pipeline(&lib, "stack_windows")?;
    if n.is_power_of_two() && windows <= u32::MAX as usize / n / 3 {
        let init = gpu.pipeline(&lib, "fft_init")?; let stage = gpu.pipeline(&lib, "fft_stage")?;
        let whiten = gpu.pipeline(&lib, "fft_whiten")?; let finish = gpu.pipeline(&lib, "fft_finish")?;
        let x = gpu.buffer_zeroed::<f32>(windows * n * 6); let y = gpu.buffer_zeroed::<f32>(windows * n * 6);
        gpu.batch(|batch| {
            batch.dispatch(&init, &[Arg::Buf(&ba),Arg::Buf(&bb),Arg::Buf(&x),Arg::U32(n as u32),Arg::U32(windows as u32)], [n, windows, 1])?;
            let mut src = &x; let mut dst = &y;
            for s in 0..n.trailing_zeros() {
                batch.dispatch(&stage, &[Arg::Buf(src),Arg::Buf(dst),Arg::U32(n as u32),Arg::U32(windows as u32),Arg::U32(s),Arg::U32(0)], [n, windows * 2, 1])?;
                std::mem::swap(&mut src, &mut dst);
            }
            batch.dispatch(&whiten, &[Arg::Buf(src),Arg::Buf(dst),Arg::Buf(&bm),Arg::U32(n as u32),Arg::U32(windows as u32),Arg::F32(eps)], [n, windows, 1])?;
            std::mem::swap(&mut src, &mut dst);
            for s in 0..n.trailing_zeros() {
                batch.dispatch(&stage, &[Arg::Buf(src),Arg::Buf(dst),Arg::U32(n as u32),Arg::U32(windows as u32),Arg::U32(s),Arg::U32(1)], [n, windows * 3, 1])?;
                std::mem::swap(&mut src, &mut dst);
            }
            batch.dispatch(&finish, &[Arg::Buf(src),Arg::Buf(&wa),Arg::Buf(&wb),Arg::Buf(&corr),Arg::U32(n as u32),Arg::U32(windows as u32)], [n, windows, 1])?;
            batch.dispatch(&stack_kernel, &[Arg::Buf(&corr),Arg::Buf(&stack),Arg::U32(n as u32),Arg::U32(windows as u32)], [n, 1, 1])
        })?;
    } else {
        let sa = gpu.buffer_zeroed::<f32>(windows * bins * 2); let sb = gpu.buffer_zeroed::<f32>(windows * bins * 2);
        let forward = gpu.pipeline(&lib, "whiten_spectrum")?; let inverse = gpu.pipeline(&lib, "inverse_and_correlate")?;
        gpu.batch(|batch| {
            batch.dispatch(&forward, &[Arg::Buf(&ba),Arg::Buf(&bb),Arg::Buf(&bm),Arg::Buf(&sa),Arg::Buf(&sb),Arg::U32(n as u32),Arg::U32(windows as u32),Arg::F32(eps)], [bins, windows, 1])?;
            batch.dispatch(&inverse, &[Arg::Buf(&sa),Arg::Buf(&sb),Arg::Buf(&wa),Arg::Buf(&wb),Arg::Buf(&corr),Arg::U32(n as u32),Arg::U32(windows as u32)], [n, windows, 1])?;
            batch.dispatch(&stack_kernel, &[Arg::Buf(&corr),Arg::Buf(&stack),Arg::U32(n as u32),Arg::U32(windows as u32)], [n, 1, 1])
        })?;
    }
    out.f32("whitened_a", &a.shape, &gpu.read::<f32>(&wa, windows * n))?;
    out.f32("whitened_b", &a.shape, &gpu.read::<f32>(&wb, windows * n))?;
    out.f32("correlations", &a.shape, &gpu.read::<f32>(&corr, windows * n))?;
    out.f32("stack", &[n], &gpu.read::<f32>(&stack, n))?;
    Ok(())
}
fn main() { outcrop_common::run(&[("ambient_noise", ambient_noise)]); }
