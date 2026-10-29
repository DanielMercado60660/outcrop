//! One-dimensional layered-earth magnetotelluric forward response.
use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};

fn mt1d(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let rho = inp.f32("resistivity")?;
    let thickness = inp.f32("thickness")?;
    let frequency = inp.f32("frequency")?;
    if rho.shape.len() != 1 || rho.data.is_empty() || thickness.shape.len() != 1
        || thickness.data.len() != rho.data.len() - 1 || frequency.shape.len() != 1
    {
        return Err("expected 1D arrays with L > 0 and L-1 thicknesses".into());
    }
    if rho.data.iter().chain(&thickness.data).chain(&frequency.data)
        .any(|&x| !x.is_finite() || x <= 0.0)
    {
        return Err("inputs must be finite and positive".into());
    }
    let n = frequency.data.len();
    let layers = u32::try_from(rho.data.len())?;
    let count = u32::try_from(n)?;
    if n == 0 {
        out.f32("apparent_resistivity", &[0], &[])?;
        out.f32("phase", &[0], &[])?;
        return Ok(());
    }
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/mt1d.metal"))?;
    let kernel = gpu.pipeline(&library, "mt1d")?;
    let rho_buf = gpu.buffer_from(&rho.data);
    let thickness_buf = gpu.buffer_from(&thickness.data);
    let frequency_buf = gpu.buffer_from(&frequency.data);
    let apparent_buf = gpu.buffer_zeroed::<f32>(n);
    let phase_buf = gpu.buffer_zeroed::<f32>(n);
    gpu.dispatch(&kernel, &[
        Arg::Buf(&rho_buf), Arg::Buf(&thickness_buf), Arg::Buf(&frequency_buf),
        Arg::Buf(&apparent_buf), Arg::Buf(&phase_buf), Arg::U32(layers), Arg::U32(count),
    ], [n, 1, 1])?;
    out.f32("apparent_resistivity", &[n], &gpu.read::<f32>(&apparent_buf, n))?;
    out.f32("phase", &[n], &gpu.read::<f32>(&phase_buf, n))?;
    Ok(())
}

fn main() {
    outcrop_common::run(&[("mt1d", mt1d)]);
}
