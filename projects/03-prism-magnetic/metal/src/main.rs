use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

mod buffers;

fn prism_magnetic(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let xyz = inp.f32("coordinates")?;
    let prisms = inp.f32("prisms")?;
    let magnetization = inp.f32("magnetization")?;
    let field = inp.f32("inducing_field")?;
    if xyz.shape.len() != 2 || xyz.shape[1] != 3 {
        return Err("coordinates must have shape (N, 3)".into());
    }
    if prisms.shape.len() != 2 || prisms.shape[1] != 6 {
        return Err("prisms must have shape (P, 6)".into());
    }
    let n = xyz.shape[0];
    let p = prisms.shape[0];
    if magnetization.shape != [p, 3] || field.shape != [3] {
        return Err("magnetization or inducing_field has an invalid shape".into());
    }
    // An empty source sum is exactly zero; an empty observation array still
    // produces all four correctly shaped output files without a GPU launch.
    if n == 0 || p == 0 {
        let zeros = vec![0.0f32; n];
        for name in ["b_e", "b_n", "b_u", "total_field"] {
            out.f32(name, &[n], &zeros)?;
        }
        return Ok(());
    }
    let n32 = u32::try_from(n)?;
    let p32 = u32::try_from(p)?;
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/prism.metal"))?;
    let kernel = gpu.pipeline(&lib, "prism_magnetic")?;
    let coords = buffers::from_slice(&gpu, &xyz.data, "coordinates")?;
    let bounds = buffers::from_slice(&gpu, &prisms.data, "prisms")?;
    let mags = buffers::from_slice(&gpu, &magnetization.data, "magnetization")?;
    let direction = buffers::from_slice(&gpu, &field.data, "inducing_field")?;
    let be = buffers::allocate(&gpu, n, "b_e")?;
    let bn = buffers::allocate(&gpu, n, "b_n")?;
    let bu = buffers::allocate(&gpu, n, "b_u")?;
    let total = buffers::allocate(&gpu, n, "total_field")?;
    gpu.dispatch(
        &kernel,
        &[
            Arg::Buf(&coords),
            Arg::Buf(&bounds),
            Arg::Buf(&mags),
            Arg::Buf(&direction),
            Arg::Buf(&be),
            Arg::Buf(&bn),
            Arg::Buf(&bu),
            Arg::Buf(&total),
            Arg::U32(n32),
            Arg::U32(p32),
        ],
        [n, 1, 1],
    )?;
    for (name, buffer) in [
        ("b_e", &be),
        ("b_n", &bn),
        ("b_u", &bu),
        ("total_field", &total),
    ] {
        out.f32(name, &[n], &gpu.read::<f32>(buffer, n))?;
    }
    Ok(())
}

fn main() {
    outcrop_common::run(&[("prism_magnetic", prism_magnetic)]);
}
