use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

fn tesseroid_gravity(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let coordinates = inp.f32("coordinates")?;
    let tesseroids = inp.f32("tesseroids")?;
    let density = inp.f32("density")?;
    if coordinates.shape.len() != 2 || coordinates.shape[1] != 3 {
        return Err("coordinates must have shape (N, 3)".into());
    }
    if tesseroids.shape.len() != 2 || tesseroids.shape[1] != 6 {
        return Err("tesseroids must have shape (M, 6)".into());
    }
    let n = coordinates.shape[0];
    let m = tesseroids.shape[0];
    if density.shape != [m] { return Err("density must have shape (M,)".into()); }
    let gz = match inp.str("field")?.as_str() {
        "potential" => 0,
        "g_z" => 1,
        _ => return Err("unknown field".into()),
    };
    let radial = inp.flag("radial_adaptive")?;
    if n == 0 || m == 0 { return out.f32("field", &[n], &vec![0.0; n]); }
    let n_u32 = u32::try_from(n)?;
    let m_u32 = u32::try_from(m)?;
    // Form offsets in f64 from exact float32 inputs before converting to radians.
    let pair_floats = n.checked_mul(m).and_then(|x| x.checked_mul(9)).ok_or("input too large")?;
    // The shader computes flattened pair offsets using uint arithmetic.
    if pair_floats > u32::MAX as usize {
        return Err("pair buffer exceeds Metal 32-bit indexing limit".into());
    }
    let mut pairs = Vec::with_capacity(pair_floats);
    let deg = std::f64::consts::PI / 180.0;
    for obs in coordinates.data.chunks_exact(3) {
        for (cell, &rho) in tesseroids.data.chunks_exact(6).zip(&density.data) {
            for k in 0..4 {
                let origin = if k < 2 { obs[0] } else { obs[1] };
                pairs.push(((cell[k] as f64 - origin as f64) * deg) as f32);
            }
            pairs.push((cell[4] as f64 - obs[2] as f64) as f32);
            pairs.push((cell[5] as f64 - obs[2] as f64) as f32);
            pairs.push((obs[1] as f64 * deg) as f32);
            pairs.push(obs[2]);
            pairs.push(rho);
        }
    }
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/tesseroid.metal"))?;
    let kernel = gpu.pipeline(&library, "tesseroid_gravity")?;
    let input = gpu.buffer_from(&pairs);
    let output = gpu.buffer_zeroed::<f32>(n);
    let status = gpu.buffer_zeroed::<u32>(n);
    gpu.dispatch(&kernel, &[Arg::Buf(&input), Arg::Buf(&output), Arg::U32(n_u32), Arg::U32(m_u32), Arg::U32(gz), Arg::U32(radial as u32), Arg::Buf(&status)], [n, 1, 1])?;
    if let Some((index, code)) = gpu.read::<u32>(&status, n).into_iter().enumerate().find(|(_, code)| *code != 0) {
        let reason = if code == 1 { "subdivision stack exhausted" } else { "float32 subdivision precision exhausted" };
        return Err(format!("observation {index}: {reason}; an observation on or extremely close to a source boundary may not admit a finite subdivision").into());
    }
    out.f32("field", &[n], &gpu.read::<f32>(&output, n))
}

fn main() {
    outcrop_common::run(&[("tesseroid_gravity", tesseroid_gravity)]);
}
