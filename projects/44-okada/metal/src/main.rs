use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

fn okada(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let xyz = inp.f32("coordinates")?;
    if xyz.shape.len() != 2 || xyz.shape[1] != 3 { return Err("coordinates must have shape (N, 3)".into()); }
    let n = xyz.shape[0];
    let slip = inp.f32("slip")?;
    if slip.data.len() != 3 { return Err("slip must have three elements".into()); }
    let bounds = inp.f32("bounds")?;
    if bounds.data.len() != 4 { return Err("bounds must have four elements".into()); }
    let source = inp.str("source")?;
    if source != "point" && source != "finite_rectangle" { return Err("unknown source".into()); }
    if source == "point" && xyz.data.chunks_exact(3).any(|p| p[2] != 0.0) { return Err("point observations must be at the surface".into()); }
    if n == 0 {
        out.f32("displacement", &[0, 3], &[])?;
        out.f32("horizontal_derivatives", &[0, 3, 2], &[])?;
        return Ok(());
    }
    let gpu = Gpu::new()?;
    let lib = if source == "point" { gpu.compile(include_str!("../kernels/okada.metal"))? } else { gpu.compile(include_str!("../kernels/rectangle.metal"))? };
    let pipeline = gpu.pipeline(&lib, if source == "point" { "point_surface" } else { "rectangle" })?;
    let bx = gpu.buffer_from(&xyz.data);
    let bs = gpu.buffer_from(&slip.data);
    let bu = gpu.buffer_zeroed::<f32>(n * 3);
    let bd = gpu.buffer_zeroed::<f32>(n * 6);
    let depth = inp.num("depth")? as f32;
    let dip = inp.num("dip_deg")?;
    let alpha = inp.num("alpha")? as f32;
    if source == "point" {
        gpu.dispatch(&pipeline, &[Arg::Buf(&bx), Arg::Buf(&bs), Arg::Buf(&bu), Arg::Buf(&bd), Arg::F32(depth), Arg::F32(dip as f32), Arg::F32(alpha), Arg::U32(n as u32)], [n, 1, 1])?;
    } else {
        let (mut sd, mut cd) = dip.to_radians().sin_cos();
        if cd.abs() < 1e-6 { sd = sd.signum(); cd = 0.0; }
        let bb = gpu.buffer_from(&bounds.data);
        gpu.dispatch(&pipeline, &[Arg::Buf(&bx), Arg::Buf(&bs), Arg::Buf(&bu), Arg::Buf(&bd), Arg::F32(depth), Arg::F32(sd as f32), Arg::F32(cd as f32), Arg::F32(alpha), Arg::Buf(&bb), Arg::U32(n as u32)], [n, 1, 1])?;
    }
    out.f32("displacement", &[n, 3], &gpu.read::<f32>(&bu, n * 3))?;
    out.f32("horizontal_derivatives", &[n, 3, 2], &gpu.read::<f32>(&bd, n * 6))?;
    Ok(())
}

fn main() { outcrop_common::run(&[("okada", okada)]); }
