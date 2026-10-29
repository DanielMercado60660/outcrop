//! Two-dimensional polygon gravity, evaluated on the Metal device.
use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

fn polygon_gravity(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let vertices = inp.f32("vertices")?;
    let observations = inp.f32("observations")?;
    let density = inp.num("density")?;
    if vertices.shape.len() != 2 || vertices.shape[1] != 2 || vertices.shape[0] < 3 {
        return Err("vertices must have shape (n, 2), n >= 3".into());
    }
    if observations.shape.len() != 2 || observations.shape[1] != 2 {
        return Err("observations must have shape (m, 2)".into());
    }
    let n = vertices.shape[0];
    let m = observations.shape[0];
    let n32 = u32::try_from(n)?;
    let m32 = u32::try_from(m)?;
    if density == 0.0 || m == 0 {
        return out.f32("gz", &[m], &vec![0.0; m]);
    }
    // Use a translated origin and f64 to compute the polygon winding.
    let x0 = vertices.data[0] as f64;
    let z0 = vertices.data[1] as f64;
    let mut area2 = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let x1 = vertices.data[2 * i] as f64 - x0;
        let z1 = vertices.data[2 * i + 1] as f64 - z0;
        let x2 = vertices.data[2 * j] as f64 - x0;
        let z2 = vertices.data[2 * j + 1] as f64 - z0;
        area2 += x1 * z2 - z1 * x2;
    }
    let scale = (2.0 * 6.67430e-11 * 1.0e5 * density * area2.signum()) as f32;
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/gravity.metal"))?;
    let kernel = gpu.pipeline(&library, "polygon_gravity")?;
    let vertices_buf = gpu.buffer_from(&vertices.data);
    let obs_buf = gpu.buffer_from(&observations.data);
    let result_buf = gpu.buffer_zeroed::<f32>(m);
    gpu.dispatch(&kernel, &[Arg::Buf(&vertices_buf), Arg::Buf(&obs_buf), Arg::Buf(&result_buf), Arg::U32(n32), Arg::U32(m32), Arg::F32(scale)], [m, 1, 1])?;
    let result: Vec<f32> = gpu.read(&result_buf, m);
    out.f32("gz", &[m], &result)
}

fn main() {
    outcrop_common::run(&[("polygon_gravity", polygon_gravity)]);
}
